#-*- coding:utf-8 -*-

# basic
import os, timeit, copy, time
import math, json, operator
import pickle
import numpy as np
import multiprocessing as mp
from operator import itemgetter   # for sorting list of list
import datetime 
from decimal import *

import matplotlib
import matplotlib.pyplot as plt

# marsan
from lib import schema as db
from lib import adhub_enum as num
from lib import adhub_data as dat
from lib import svr_plot as plot

mp_out = mp.Queue()
#=================================================
#   small tasks
#================================================= 
def cal_score(spent, impression, click, opt_target, min_imp=0, min_clk=0, min_spent=100, fix_outlier=True):
  # for confidence range (mixed with N/A as zero, cannot use num.cint)
  if isinstance(impression, dict): impression = impression['median']
  if isinstance(click, dict): click = click['median']
  # general
  if (fix_outlier & ((spent < min_spent) | (impression < min_imp) | (click < min_clk))):
    return 0 if (opt_target == 'ctr') else float(100000) #"inf")  # forbidden case
  else:
    print '[cal_score]', opt_target, impression, spent
    if (opt_target in ['cpm', 'ocpm']):
      if (float(impression) == 0):
        score = 0
      else:
        score = float(spent) * 1000 / float(impression)
    elif (opt_target == 'cpc'):
      if (float(click) == 0):
        score = 0
      else:
        score = float(spent) / float(click)
    elif (opt_target == 'ctr'):
      if (float(impression) == 0):
        score = 0
      else:
        score = float(click) / float(impression)
    return score

def pick_peak(X, Z, score_target, fix_acts={}):
  S = []
  # calculate score
  for idx, z in enumerate(Z):
    if (score_target in num.btypes):
      score = cal_score(X[idx].spent, z['lifetime_impressions'], z['lifetime_clicks'], score_target)
    else:
      score = z[score_target]
    # check if satisfy lower bound
    # if fix_acts: print "fix_acts = ", fix_acts
    for k,v in fix_acts.iteritems():
      v_float = float(v)
      if (z[k] < v_float): score = np.nan
    # remove impossible case
    if (X[idx].min_age > X[idx].max_age): 
      score = np.nan
      # print "impossible case: min_age=%i, max_age=%i" % (X[idx].min_age, X[idx].max_age)
    S.append(score)
  # find best
  print "[score_target] %s - (%.2f / %.2f)" % (score_target, np.nanmin(S), np.nanmax(S))
  try:
    best_idx = np.nanargmin(S) if (score_target in ['ocpm', 'cpm', 'cpc']) else np.nanargmax(S)
    return X[best_idx], Z[best_idx]
  except ValueError:
    print "[ValueError] whole solution space not qualified!"
    return None, None


#=================================================
#   Confidence interval
#================================================= 
def confidence_range(samples, scale=None, outlier=None, en_plot=False):
  if outlier:
    rto = int(len(samples)*outlier)
    samples = sorted(samples)[rto:-rto]
  bmin = int(min(samples))
  bmax = int(max(samples))
  if en_plot:
    rng = range(bmin, bmax, scale) if scale else range(bmin, bmax)
    plt.hist(samples, bins=rng)
    plt.show()
  return bmin, bmax


def rf_estimate_dist(rf_model, scaler, s):
  results = []
  dummy_len = len(s)
  s_norm = scaler.transform(s)
  results_raw = [tree.predict(s_norm[1:]) for tree in rf_model.best_estimator_.estimators_]
  for r in results_raw:
    r_dummy = r + [0]*(dummy_len)
    rr = scaler.inverse_transform(r_dummy)[0]
    results.append(rr)
  return results


def samples_to_dist_info(samples, outliers=[0.05, 0.1, 0.2], en_plot=False):
  info = {'median': np.median(samples)}
  scale = int(info['median']/50)
  for ol in outliers:
    bmin, bmax = confidence_range(samples, scale=scale, outlier=ol)
    cf = (1-2*ol)*100
    info['range_%i' % cf] = {'min': bmin, 'max': bmax}
    # print "%i %% confidence in range (%.2f, %.2f)." % (cf, bmin, bmax)
  if en_plot: confidence_range(samples, scale=scale, en_plot=en_plot)
  return info
    

def demo_confidence_interval(btype='ocpm', pack='div_2015_0525_1618'):
  # random sample
  raw = dat.get_raw_sample(btype)
  sample = raw[np.random.random_integers(1,raw.count())]
  s = dat.fetch_record(sample, 'lifetime_impressions')
  s = np.array(s).astype(np.float)
  # predict
  filename = './%s/top/result/new_ocpm_lifetime_impressions.gs' % pack
  rf_model, X_train, X_test, y_train, y_test, scaler = pickle.load(open(filename))
  samples = rf_estimate_dist(rf_model, scaler, s)
  info = samples_to_dist_info(samples)
  print info

#=================================================
#   main flow
#================================================= 
def predict_1_sample(model, sample, convs=None, cint=None):
  z = {}
  if not convs: convs = num.convs 
  for t in convs:
    if ((t+'_svr') not in model): continue  # predict only if model exists
    s = dat.fetch_record(sample, t)
    s[0] = 0  # prevent null been feed to transform
    s = np.array(s).astype(np.float)
    if cint:
      print "[predict_1_sample] ################## predict with confidence interval !!"
      z[t] = rf_estimate_dist(model[t+'_svr'], model[t+'_scaler'], s)
    else:
      try:
        s_norm = model[t+'_scaler'].transform(s)
      except:
        print '[s]', s[:100]
      s_norm[0] = model[t+'_svr'].predict(s_norm[1:])
      z[t] = model[t+'_scaler'].inverse_transform(s_norm)[0]
      z[t] = 0 if (z[t] < 0) else z[t]  # remove negative value
  return z

def predict_1_sample_ensemble(model, sample, convs=None, cint=False, price_adjust=False):
  result = {}
  if not convs: convs = num.convs 
  for t in convs: result[t] = []
  # predict through available models
  for c in ['top', 'client_industry', 'client_name', 'objective']:
    div = 'top' if (c == 'top') else getattr(sample, c)
    if (div in model):
      for k,v in predict_1_sample(model[div], sample, convs, cint).items():
        s = model[div][k + '_score']**2
        result[k].append([v, s])
  # ensemble final results with weighting
  for t in convs:
    if cint:
      merged_samples = []
      for rt in result[t]:
        weight = int(rt[1]*1000)
        rtc = len(rt[0])
        merged_samples += [rt[0][np.random.randint(rtc)] for i in range(0, weight)]
      if (len(merged_samples) == 0):
        result[t] = {'median': 'N/A'}
      else:
        result[t] = samples_to_dist_info(merged_samples)
    else:
      w_sum = sum(map(lambda k: k[1], result[t]))
      fix = sum(map(lambda k: k[0]*k[1]/w_sum, result[t]))
      result[t] = fix
    # print t, result[t], fix
  # price adjustment
  if price_adjust:
    de = sample.sdate or datetime.datetime.now()
    ds = de - datetime.timedelta(days=7)
    rpts = db.daily_trends._get_collection().aggregate([
      { "$match": { 
        'bid_type': sample.bid_type, 
        'segname' : 'page_types', 
        'segment' : sample.page_types,
        'objective' : sample.objective,
        'onDate' : {"$gte" : ds, "$lte" : de} } 
      }, 
      { "$group": {
          "_id": 1,
          "impressions": { "$sum": '$impressions'},
          "clicks": { "$sum": '$clicks'},
          "spent": { "$sum": '$spent'},
        }
      }
    ])['result']
    if (len(rpts) == 0):
      print "[WARNING] price adjustment disabled since no reference price available."
    else:
      rpt = rpts[0]
      ref = {}
      ref['ecpc'] = float(rpt['spent']) / float(rpt['clicks']) if (rpt['clicks'] > 0) else None
      ref['ecpm'] = float(rpt['spent'])*1000 / float(rpt['impressions']) if (rpt['impressions'] > 0) else None
      print "[Aggregate Reference] ecpm:%.2f/ecpc:%.2f" % (ref['ecpm'], ref['ecpc']), rpts
      for s in ['ecpm', 'ecpc']:
        ceil = ref[s]*2
        floor = ref[s]*0.5
        if (result['lifetime_'+s] > ceil): 
          print "[price_adjust] force %s %.1f -> %.1f" % (s, result['lifetime_'+s], ceil)
          result['lifetime_'+s] = ceil
        elif (result['lifetime_'+s] < floor):
          print "[price_adjust] force %s %.1f -> %.1f" % (s, result['lifetime_'+s], floor)
          result['lifetime_'+s] = floor
  return result


def predict_1d_samples(model, sample, label, opt_target=None, opt_action=None, fix_acts=None, chart_target='opt', is_thread=False):
  X = []  # input
  Z = []  # output
  S = []  # chart
  space = num.all_range[label]
  for xi in space:
    x = copy.deepcopy(sample)
    setattr(x, label, xi)
    z = predict_1_sample_ensemble(model, x)
    X.append(x)
    Z.append(z)
    if (chart_target == 'opt'):
      s = cal_score(x.spent, z['lifetime_impressions'], z['lifetime_clicks'], opt_target, fix_outlier=False)
    else:
      s = z[chart_target]
    S.append(s)
  # find best
  xb, zb = pick_peak(X, Z, opt_action, fix_acts)
  # generate chart data
  if chart_target:
    if (isinstance(space[1], basestring)): space = np.arange(len(space))
    z_label = opt_target if (chart_target == 'opt') else chart_target
    chart = plot.plot_grid_2d(space, S, X_label=label, Z_label=z_label, json_data=True)
    out = {'label': label, 'xb': xb, 'zb': zb, 'chart': chart}
  else:
    out = {'label': label, 'xb': xb, 'zb': zb, 'chart': None}
  # output
  if is_thread:
    mp_out.put(out)
  else:
    return out

#=================================================
#   multi-process find_best_set
#================================================= 
def find_best_set_mp(model, sample, opt_target, opt_action, fix_vals=None, fix_acts=None, chart_target=None):
  X = []
  Z = []
  charts = {}
  processes = []
  Ts = timeit.default_timer()
  for label in num.all_labels:
    if label in fix_vals: continue
    p = mp.Process(target=predict_1d_samples, args=(model, sample, label, opt_target, opt_action, fix_acts, chart_target, True))
    p.start()
    processes.append(p)
  for p in processes:
    p.join(10)
  results = [mp_out.get() for p in processes]
  for r in results:
    if r['xb']: # X might return None, if all solution not qualified by fix_acts
      X.append(r['xb'])
      Z.append(r['zb'])
    if chart_target: charts[r['label']] = r['chart']
  if X:
    X_best, Z_best = pick_peak(X, Z, opt_action)
    print "[find_best_set] multi-process all-done. (%.1fs secs)" % (timeit.default_timer() - Ts)
    return X_best, Z_best, charts
  else:
    return None, None, charts

