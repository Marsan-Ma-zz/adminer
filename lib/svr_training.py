#-*- coding:utf-8 -*-
# basic
import os
import copy
import json                       # json encoder/decoder
import operator                   # for sorting dict
import timeit                     # for benchmark
import random 
import math
import pickle
from operator import itemgetter   # for sorting list of list
from datetime import datetime   

# plot
import matplotlib
import matplotlib.pyplot as plt   # plot plate
import mpld3                      # plot to web
# import matplotlib.cm as cm
from matplotlib import cm
from matplotlib.font_manager import FontProperties
font = FontProperties(fname=r"/usr/share/fonts/truetype/LiHeiPro.ttf", size=11)   # chinese font for matplotlib
# from mpl_toolkits.mplot3d.axes3d import Axes3D
from mpl_toolkits.mplot3d import axes3d
from scipy.stats import sem

# SVR
import numpy as np
from sklearn import preprocessing # for data normalize
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import make_scorer, r2_score
from sklearn.cross_validation import train_test_split
from sklearn import svm

from sklearn.ensemble import RandomForestRegressor

# marsan
from lib import adhub_enum as num
from lib import svr_plot as plot
from lib import adhub_data as dat
from lib import svr_predict as pred
# from lib import svr_tasks as svt

# overwrite GridSearchCV for celery workers
if num.en_celery:
  from lib import grid_search_mp as grid_mp
  GridSearchCV.fit = grid_mp.fit

ROOT = '/home/marsan/workspace/adminer'
#=================================================
#   Scorer
#================================================= 
def rmse_scorer(y_true, y_pred):
  return rmse(y_pred, y_true)

def mix_scorer(y_true, y_pred):
  # y_true = map(lambda d: math.exp(min(100,d)), y_true)
  # y_pred = map(lambda d: math.exp(min(100,d)), y_pred)
  y_true = np.array(y_true).astype(np.float)
  y_pred = np.array(y_pred).astype(np.float)
  # print "RMSE - ", '%2.3f' % rmse(y_pred, y_true)
  r2 = r2_score(y_true, y_pred)
  # print "R2 - ", '%2.3f' % r2
  return r2

def mean_score(scores):
  """Print the empirical mean score and standard error of the mean."""
  return ("Mean score: {0:.3f} (+/-{1:.3f})").format(np.mean(scores), sem(scores))

def rmse(targets, predictions):
  return np.sqrt(((predictions - targets) ** 2).mean())

def list_flatten(nested_list):
  return [item for sublist in nested_list for item in sublist];


#================================
#  Grid Search
#================================
def grid_search(X_samples, Y_samples, gs_params, cv=3, debug=False):
  Ts = timeit.default_timer()
  if (num.kernel == 'linear'):
    gs_svr = GridSearchCV(svm.LinearSVR(), gs_params, n_jobs=6, cv=cv) #, refit=False) 
  elif (num.kernel == 'rfr'):
    gs_svr = GridSearchCV(RandomForestRegressor(), gs_params, n_jobs=6, cv=cv) #, refit=False) 
  else:
    gs_svr = GridSearchCV(svm.SVR(kernel=num.kernel, cache_size=1000), gs_params, n_jobs=6, cv=cv) #, refit=False)
  if num.en_celery:
    gs_svr.fit(X_samples, Y_samples, en_celery=True)
  else:
    gs_svr.fit(X_samples, Y_samples)
  print "[Best Params] %s %.3f @ %s" % (gs_svr.best_params_, gs_svr.best_score_, str(timeit.default_timer() - Ts))
  return gs_svr

def vec_recover(scaler, X, Y):
  S = np.insert(X, 0, Y, axis=1)
  Z = scaler.inverse_transform(S)
  if num.do_exp:
    Zo = map(lambda d: math.exp(d), Z[:,0])
    Zo = np.array(Zo).astype(np.float)
  else: 
    Zo = Z[:,0]
  return Zo

def crop_curve(y_pred, y_real, tmin=None, tmax=None):
  y_real_tmp = []
  y_pred_tmp = []
  for i in np.arange(len(y_real)):
    if (((not tmin) | (y_real[i] > tmin)) & ((not tmax) | (y_real[i] < tmax))):
      y_real_tmp.append(y_real[i])
      y_pred_tmp.append(y_pred[i])
  y_pred = np.array(y_pred_tmp).astype(np.float)
  y_real = np.array(y_real_tmp).astype(np.float)
  return y_pred, y_real


def show_result(gs_svr, scaler, X_train, y_train, X_test, y_test, en_plot=False, tmin=None, tmax=None):
  # predict
  Yt = gs_svr.predict(X_train)
  Yv = gs_svr.predict(X_test)
  # reverse scaling
  Yt_rec = vec_recover(scaler, X_train, Yt)
  Yv_rec = vec_recover(scaler, X_test, Yv)
  y_train_rec = vec_recover(scaler, X_train, y_train)
  y_test_rec = vec_recover(scaler, X_test, y_test)
  # crop segment for observation
  if (tmin or tmax):
    Yt_rec, y_train_rec = crop_curve(Yt_rec, y_train_rec, tmin, tmax)
    Yv_rec, y_test_rec = crop_curve(Yv_rec, y_test_rec, tmin, tmax)
  # calculate score
  print "[Train] Rsqr=", r2_score(y_train_rec, Yt_rec), ', RMSE=', rmse(y_train_rec, Yt_rec)
  print "[Verify] Rsqr=", r2_score(y_test_rec, Yv_rec), ', RMSE=', rmse(y_test_rec, Yv_rec)
  # plot
  if en_plot:
    plot.plot_predict(y_train_rec, Yt_rec)
    plot.plot_predict(y_test_rec, Yv_rec)
    # plot.plot_svr_grid(gs_svr)
  
#================================
#  Smart Grid Search
#================================
def svr_smart_search(path, X_samples, Y_samples, scaler, seg=4, filename_prefix='', remote=False):
  best_score = -1000
  best_params = {}
  params = {
    # svm
    'c_min': -2, 'c_max': 4, 
    'g_min': -4, 'g_max': 2, 
    # random forest
    'split_min': 10, 'split_max': 30,
    'leaf_min': 10, 'leaf_max': 30,
  }
  try_cnt = 0
  if num.simple: seg = num.simple_seg
  while True:
    # choose grid search parameters
    if (num.kernel == 'linear'): 
      gs_params = {
        'C': np.logspace(params['c_min'], params['c_max'], seg)
      }
    elif (num.kernel == 'rbf'):
      gs_params = {
        'C': np.logspace(params['c_min'], params['c_max'], seg),
        'gamma': np.logspace(params['g_min'], params['g_max'], seg),
      }
    elif (num.kernel == 'rfr'):
      gs_params = {
        'n_estimators': num.n_estimators,
        'min_samples_split': range(params['split_min'], params['split_max'], seg),
        'min_samples_leaf': range(params['leaf_min'], params['leaf_max'], seg),
      }
    gs_svr, X_train, y_train, X_test, y_test = svr_main(gs_params, X_samples, Y_samples, scaler, export_params=True)
    if (num.kernel == 'linear'):
      print "[SmartSearch @ %s] try=%i, score=%.3f, C=(%.3f, %.3f)\n" % (dat.curtime(), try_cnt, gs_svr.best_score_, params['c_min'], params['c_max'])
    elif (num.kernel == 'rbf'):
      print "[SmartSearch @ %s] try=%i, score=%.3f, C=(%.3f, %.3f), gamma=(%.3f, %.3f)\n" % (dat.curtime(), try_cnt, gs_svr.best_score_, params['c_min'], params['c_max'], params['g_min'], params['g_max'])
    elif (num.kernel == 'rfr'):
      print "[SmartSearch @ %s] try=%i, score=%.3f, split=(%i, %i), leaf=(%i, %i)" % (dat.curtime(), try_cnt, gs_svr.best_score_, params['split_min'], params['split_max'], params['leaf_min'], params['leaf_max'])
    print '-----------------------------'
    # update params
    if (gs_svr.best_score_ > best_score):  # get better, keep model
      try_cnt = 0
      params = update_params_range(gs_svr.grid_scores_)
      best_score, best_params = gs_svr.best_score_, gs_svr.best_params_
      best_gs_svr, best_X_train, best_X_test, best_y_train, best_y_test = gs_svr, X_train, X_test, y_train, y_test
    elif (try_cnt >= 3): # search ended, export best model
      if (num.kernel == 'linear'): 
        filename = "r2_%i_c_%i" % (best_score*100, best_params['C'])
      elif (num.kernel == 'rbf'): 
        filename = "r2_%i_c_%i_g_%i" % (best_score*100, best_params['C'], best_params['gamma']*10000)
      elif (num.kernel == 'rfr'):
        filename = "r2_%i_s_%i_l_%i" % (best_score*100, best_params['min_samples_split'], best_params['min_samples_leaf'])
      if filename_prefix: filename = "%s_%s" % (filename_prefix, filename)
      filename = "svr_%s.gs" % filename
      # bp = {'C': [best_params['C']]}
      # if (num.kernel == 'rbf'): bp['gamma'] = [best_params['gamma']]
      # best_gs_svr = grid_search(X_samples, Y_samples, bp)
      if remote:
        return best_gs_svr, scaler, filename
      else:
        pickle.dump([best_gs_svr, best_X_train, best_X_test, best_y_train, best_y_test, scaler], open("%s/%s" % (path, filename), "w"))
        return filename
    else: # keep trying
      try_cnt += 1
      params = update_params_range(gs_svr.grid_scores_)

def update_params_range(scores, p_rate=2):
  # print scores
  score_sorted = sorted(scores, key=lambda k: k[1], reverse=True)
  score_sorted = score_sorted[:(len(score_sorted)/p_rate)+1]
  # SVM
  if 'C' in score_sorted[0][0]: 
    cs = map(lambda s: s[0]['C'], score_sorted)
    if (num.kernel == 'rbf'): 
      gs = map(lambda s: s[0]['gamma'], score_sorted)
      return {'c_min': math.log(min(cs), 10), 'c_max': math.log(max(cs), 10), 'g_min': math.log(min(gs), 10), 'g_max': math.log(max(gs), 10)}
    else:
      return {'c_min': math.log(min(cs), 10), 'c_max': math.log(max(cs), 10)}
  # Random Forest
  elif 'n_estimators' in score_sorted[0][0]: 
    split = map(lambda s: s[0]['min_samples_split'], score_sorted)
    leaf = map(lambda s: s[0]['min_samples_leaf'], score_sorted)
    return {'split_min': min(split), 'split_max': max(split)+1, 'leaf_min': min(leaf), 'leaf_max': max(leaf)+1}
  

    
def svr_main(gs_params, X_samples, Y_samples, scaler, filename=None, export_params=False, en_plot=False):
  # training & verification
  Ts = timeit.default_timer()
  X_train, X_test, y_train, y_test = train_test_split(X_samples, Y_samples)
  print '[train_test_split]', np.shape(X_train), np.shape(y_train), np.shape(X_test), np.shape(y_test)
  gs_svr = grid_search(X_train, y_train, gs_params=gs_params)
  
  # plot result
  show_result(gs_svr, scaler, X_train, y_train, X_test, y_test, en_plot)
  print "[Time] ", timeit.default_timer() - Ts, '@', dat.curtime()
  if export_params: return gs_svr, X_train, y_train, X_test, y_test


#================================
#  Confidence Interval curve
#================================
def train_confidence_interval_curve(package, models, plot=False):
  # models = svt.init_div_models(package, threshold=0.5)
  convs = num.convs
  path = "%s/%s/conf" % (ROOT, package)
  if os.path.isdir(path): 
    print "[Error] folder %s already exists!" % path
  os.system("mkdir %s" % path)
  for btype in num.btypes:
    X = {}
    y = {}
    for c in convs:
      X[c] = []
      y[c] = []
    raw = dat.get_raw_sample(btype, char='training').filter(num_of_neighbors__ne=None).limit(100)
    print "[train_confidence_interval_curve] start %s, %i to go @ %s" % (btype, raw.count(), dat.curtime())
    for idx, ad in enumerate(raw):
      nbor = ad.num_of_neighbors  #conf.find_ads_alike(ad).count()
      resp = pred.predict_1_sample_ensemble(models[btype], ad, convs)
      for out in convs:
        if resp[out]:
          X[out].append(nbor)
          y[out].append(resp[out] - getattr(ad, out))
      if (idx % 50 == 0): print "[train_confidence_interval_curve] %s/%i @ %s" % (btype, idx, dat.curtime())
    # Fit regression model
    model_rbf = {}
    model_lin = {}
    for out in convs:
      Xm = X[out]
      ym = y[out]
      svr_rbf = svm.SVR(kernel='rbf', C=1e3, gamma=0.1)
      svr_lin = svm.SVR(kernel='linear', C=1e3)
      model_rbf[out] = svr_rbf.fit(Xm, ym)
      model_lin[out] = svr_lin.fit(Xm, ym)
      y_rbf = model_rbf[out].predict(Xm)
      y_lin = model_lin[out].predict(Xm)
      if plot:
        plt.scatter(Xm, ym, c='k', label='data')
        plt.hold('on')
        plt.plot(Xm, y_rbf, c='g', label='RBF model')
        plt.plot(Xm, y_lin, c='r', label='Linear model')
        plt.xlabel('neightbors_num')
        plt.ylabel('error')
        plt.title('confidence interval curve')
        plt.legend()
        plt.show()
    # pickle result
    filename = "%s/%s_conf.pickle" % (path, btype)
    pickle.dump([model_rbf, model_lin, X, y], open(filename, "w"))



