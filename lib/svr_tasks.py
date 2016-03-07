#-*- coding:utf-8 -*-

# basic
import os, timeit, copy, glob, re
import math, json, operator, pickle
from operator import itemgetter   # for sorting list of list
from datetime import datetime   
from xpinyin import Pinyin
import multiprocessing as mp
from decimal import *

# plot
import matplotlib
import matplotlib.pyplot as plt   # plot plate

# SVR
import numpy as np
from sklearn.metrics import r2_score

# marsan
from lib import schema as db      # database schema
from lib import adhub_enum as num
from lib import adhub_data as dat
from lib import svr_plot as plot
from lib import svr_predict as pred
from lib import svr_training as train
from lib import svr_confidence as conf

py = Pinyin()
ROOT = '/home/marsan/workspace/adminer'
#=================================================
#   Small Tasks
#================================================= 
def get_fb_obj(creative_id):
  creative = db.creatives.objects(creative_id = creative_id).first()
  object_story_id = creative.object_story_id if creative else ''
  if object_story_id:
    osid = object_story_id.split('_')
    object_story_link = "https://www.facebook.com/%s/posts/%s" % (osid[0], osid[1])
  else:
    object_story_link = None
  return object_story_link

def sample_overwrite(sample, overwrite):
  new_sample = copy.deepcopy(overwrite)
  for k,v in sample.iteritems():
    if k not in new_sample: new_sample[k] = v
  new_sample = db.adgroups(**new_sample)
  return new_sample

def cal_verify_r2(gs_svr, scaler, X_test, y_test):
  Yv = gs_svr.predict(X_test)
  Yv_rec = train.vec_recover(scaler, X_test, Yv)
  y_test_rec = train.vec_recover(scaler, X_test, y_test)
  r2 = r2_score(y_test_rec, Yv_rec)
  return r2

#=================================================
#   Model loader
#================================================= 
def load_result_mp(bt, cv, fname, q, dv=None):
  svr, scaler, score = svr_load_result(fname)
  out = {'bt': bt, 'cv': cv, 'svr': svr, 'scaler': scaler, 'dv': dv, 'score': score}
  q.put(out)

def init_div_models(pack, prefix='new', threshold=0.5, silence=True, opt_target=None, opt_action=None): # load all divided models
  Ts = timeit.default_timer()
  # check if merged pickle exist
  merged_fname = '%s/%s/merged_%s.pickle' % (ROOT, pack, int(threshold*100))
  if os.path.isfile(merged_fname):
    models = pickle.load(open(merged_fname))
    num_of_models = len(models)
    print "[Initial] load models through merged file in %.2f secs" % (timeit.default_timer() - Ts)
    return models
  # if no merged pickle, do it now.
  models = {'cpc': {}, 'cpm': {}, 'ocpm': {}}
  queue = mp.Queue()
  processes = []
  # skip models not used
  btypes = [opt_target] if opt_target else num.btypes
  convs = [opt_action] if opt_action else num.convs
  # load existing models
  for bt in btypes:
    for dv, ca in num.divs:
      for cv in convs:
        if (ca == 'top'):
          fname = '%s/%s/%s/%s_%s_%s.gs' % (ROOT, pack, ca, prefix, bt, cv)
        else:
          fname = '%s/%s/%s/result_%s/%s_%s_%s.gs' % (ROOT, pack, ca, dv, prefix, bt, cv)
        # print fname
        if os.path.isfile(fname):
          p = mp.Process(target=load_result_mp, args=(bt, cv, fname, queue, dv))
          p.start()
          processes.append(p)
  num_of_models = 0
  for idx, p in enumerate(processes):
    out = queue.get()
    if (out['dv'] not in models[out['bt']]): # initialize
      models[out['bt']][out['dv']] = {}
    if (out['score'] > threshold):
      models[out['bt']][out['dv']][out['cv']+'_svr'] = out['svr']
      models[out['bt']][out['dv']][out['cv']+'_scaler'] = out['scaler']
      models[out['bt']][out['dv']][out['cv']+'_score'] = out['score']
      num_of_models += 1
      if not silence: print "[%i] model loaded: %s/%s/%s/%.3f" % (num_of_models, out['bt'], out['dv'], out['cv'], out['score'])
  for p in processes: p.join()
  print "[Initial] load %i models done in %.2f secs" % (num_of_models, timeit.default_timer() - Ts)
  # print models['cpc'].keys(), "\n\n", models['cpm'].keys(), "\n\n", models['ocpm'].keys()
  pickle.dump(models, open(merged_fname, "w"))
  return models

#=================================================
#   Main processes
#================================================= 
def adhub_train_all_model(path, industry=None, client=None, objective=None):
  mp_pool = mp.Pool(5)
  mp_raw = []
  for target in num.btypes: 
    for out in num.convs:
      if num.en_celery_turbo:
        mp_raw.append([path, target, out, (industry or ''), (client or ''), (objective or '')])
      else:
        adhub_train_model(path, target, out, industry=industry, client=client, objective=objective)
  if num.en_celery_turbo: mp_pool.map(adhub_train_model_star, mp_raw)

def adhub_train_model_star(star):
  adhub_train_model(*star)

def adhub_train_model(path, target, out, industry=None, client=None, objective=None):
  if (industry == ''): industry = None
  if (client == ''): client = None
  if (objective == ''): objective = None
  if not num.en_celery_turbo:
    print "\n==========================================================================="
    print "   [Training] %s_%s start @ %s" % (target, out, unicode(datetime.now())[0:19])
    print "===========================================================================\n"
  if not os.path.isdir("%s" % path): os.system("mkdir %s" % path)
  # if not os.path.isdir("%s/result" % path): os.system("mkdir %s/result" % path)
  X_samples, Y_samples, scaler = dat.data_prepare(target, out, outlier=num.outlier, industry=industry, client=client, objective=objective)
  if (len(Y_samples) < 100):
    print "[SKIP] training skipped due to too few samples in %s-%s" % (target, out)
  else:
    filename_prefix = "%s_%s" % (target, out)
    filename = train.svr_smart_search(path, X_samples, Y_samples, scaler, filename_prefix=filename_prefix)
    print "[%s Search Complete] save result in %s.gs" % (target, filename)
    print "==========================================================================="
    # os.system("ln -s %s/%s %s/result/new_%s_%s.gs" % (ROOT, filename, ROOT, target, out))
    os.system("cd %s; ln -s %s new_%s_%s.gs; cd ..;" % (path, filename, target, out))


def get_recommends(target, model, opt_target, opt_action, fix_vals, fix_acts, chart_target=None):
  data = {} # output buffer
  for k,v in fix_vals.iteritems():
    if k not in ['gender', 'page_types']: v = float(v)
    if v in ['-', '']: v = None
    setattr(target, k, v)  # overwrite fix_vals
  # mp jobs
  X_best, Z_best, charts = pred.find_best_set_mp(model, target, opt_target, opt_action, fix_vals, fix_acts, chart_target)
  if not X_best: return {'err': 'all solutions not qualified by fix_acts!'}
  if num.cint:
    pred_target = pred.predict_1_sample_ensemble(model, target, cint=True)
    Z_best = pred.predict_1_sample_ensemble(model, X_best, cint=True)
  else:
    pred_target = pred.predict_1_sample_ensemble(model, target)
  print '[pred_target]', pred_target
  # similars = conf.find_ads_alike(target, opt_action=opt_action)
  # data['fb_story'] = get_fb_obj(target.creative_id)
  # post-process
  data['pred'] = {
    'cint' : num.cint,
    'targ' : {
      'spent'   : target.spent,
      'score'   : "%.2f" % pred.cal_score(target.spent, pred_target['lifetime_impressions'], pred_target['lifetime_clicks'], opt_target),
    },
    'best' : {
      'spent'   : X_best.spent,
      'score'   : "%.2f" % pred.cal_score(X_best.spent, Z_best['lifetime_impressions'], Z_best['lifetime_clicks'], opt_target),
    },
    'charts'    : charts
  }
  # find_ads_alike
  Ts = timeit.default_timer()
  # data['pred']['similars']  = similars.limit(10).to_json()
  # data['pred']['nonn']      = similars.count()
  print "[find_ads_alike] cost %.1f secs." % (timeit.default_timer() - Ts)
  # score history
  if getattr(target, 'oid'):
    data['pred']['orig'] = {
      'spent'   : target.spent,
      'score'   : "%.2f" % pred.cal_score(target.spent, target.lifetime_impressions, target.lifetime_clicks, opt_target),
    }
  # conversions
  for t in num.convs: 
    if getattr(target, 'oid'): data['pred']['orig'][t] = getattr(target, t)
    if ((t+'_svr') in model['top']):
      if num.cint:
        data['pred']['targ'][t] = pred_target[t]
        data['pred']['best'][t] = Z_best[t]
      else:
        data['pred']['targ'][t] = "%i" % pred_target[t] 
        data['pred']['best'][t] = "%i" % Z_best[t]
    else:
      data['pred']['targ'][t] = 'N/A'
      data['pred']['best'][t] = 'N/A'
  # best input
  for t in num.all_labels: 
    data['pred']['best'][t] = getattr(X_best, t)
    changed = False
    if t in ['max_bid', 'spent']:
      changed = (abs(float(getattr(X_best, t)) - float(getattr(target, t))) > 10)
    else:
      changed = (getattr(X_best, t) != getattr(target, t))
    if changed:
      data['pred']['change'] = {'atr': t, 'targ': getattr(target, t), 'best': getattr(X_best, t)}
  return data


#=================================================
#   plot results
#================================================= 
def svr_load_result(filename, plot_result=False, export_params=False, tmin=None, tmax=None):
  gs_svr, X_train, X_test, y_train, y_test, scaler = pickle.load(open(filename))
  # print "[load_result] %i train samples, %i test samples" % (len(y_train), len(y_test))
  # print "[load] %s (params: %s, score: %.2f)" % (filename.split('/')[-1], gs_svr.best_params_, gs_svr.best_score_)
  score = cal_verify_r2(gs_svr, scaler, X_test, y_test)
  if plot_result: train.show_result(gs_svr, scaler, X_train, y_train, X_test, y_test, en_plot=plot_result, tmin=tmin, tmax=tmax)
  if export_params: 
    return gs_svr, scaler, X_train, y_train, X_test, y_test
  else:
    return gs_svr, scaler, score


#=================================================
#   ensemble performance comparison
#================================================= 
def raw_sort_delete_outlier(opt_target, conv, outlier=num.outlier, industry=None, client=None, objective=None):
  Ts = timeit.default_timer()
  raw = dat.get_raw_sample(opt_target, industry=industry, client=client, objective=objective)
  raw = sorted(raw, key=lambda k: getattr(k, conv))  #[TODO] can't solve by SQL order_by, because mongoengine sorting 32MB limit?
  raw_cnt = len(raw)
  outlier = int(raw_cnt * outlier)
  raw = raw[outlier : (raw_cnt - outlier)]
  # print "total %i samples, drop %i outliers, %i samples used, cost %.2f secs." % (raw_cnt, 2*outlier, raw_cnt - 2*outlier, timeit.default_timer() - Ts)
  return raw

def ensemble_performances(q, raw, model, opt_action, conv, ensemble=True):
  y = []
  for sample in raw:
    if ensemble:
      result = pred.predict_1_sample_ensemble(model, sample, [conv])
    else:
      result = pred.predict_1_sample(model['top'], sample, [conv])
    p = result[opt_action]
    r = getattr(sample, opt_action)
    if p: y.append([p, r])
  q.put(y)

def compare_ensemble_performances(raw, package, opt_target, opt_action, model=None, en_plot=True, conv=None, ensemble=True):
  result = {}
  # initialize
  if (raw == None): 
    print "no raw_data given, use default ..."
    raw = raw_sort_delete_outlier(opt_target, conv)
  if (model == None): 
    print "no model given, use default ..."
    model = init_div_models(package, threshold=0.5, opt_target=opt_target, opt_action=opt_action)[opt_target]
  result['raw_cnt'] = len(raw)
  # predict - mp launch
  Ts = timeit.default_timer()
  q = mp.Queue()
  processes = []
  for i in range(result['raw_cnt']/1000+1):
    sraw = raw[1000*i : 1000*(i+1)]
    p = mp.Process(target=ensemble_performances, args=(q, sraw, model, opt_action, conv, ensemble))
    p.start()
    processes.append(p)
  # predict - mp collect
  y = []
  for idx, p in enumerate(processes):
    out = q.get()
    y += out
  for p in processes: p.join()
  # sort
  y_sorted = sorted(y, key=lambda k: k[1])
  # print 'y_sorted = ', len(y_sorted)
  y_sorted = np.array(y_sorted).astype(np.float)
  y_pred, y_real = y_sorted[:,0], y_sorted[:,1]
  # performance
  result['r2'] = r2_score(y_real, y_pred)
  result['rmse'] = train.rmse(y_real, y_pred)
  result['time'] = (timeit.default_timer() - Ts)
  if en_plot: plot.plot_predict(y_real, y_pred)
  # print "[compare_ensemble] %i samples, R2=%.4f, RMSE=%.2f, done in %.2f secs" % (result['raw_cnt'], result['r2'], result['rmse'], result['time'])
  return result

#=================================================
#   verticals performance comparison
#================================================= 
def show_ensembled_verticals_performances(path, out, verticals, threshold=None):
  models = init_div_models(path, threshold=0)
  timestamp = datetime.now().strftime("%Y_%m%d_%H%M")
  os.system("mkdir %s/evp_%s" % (ROOT, timestamp))
  all_performances = {}
  for vertical, space in verticals:
    performances = ensembled_verticals_performances(path, out, vertical, space, models, threshold)
    # print performances
    fig, axes = plt.subplots(1, 3, figsize=(18, 4))
    for idx, target in enumerate(['ocpm', 'cpc', 'cpm']):
      # chosen_performances = [k for k in performances[target] if k[1][out]]
      sorted_performances = sorted(performances[target].items(), key=lambda k: k[1][out]['r2_esb'])
      X = [c for c, ps in sorted_performances]
      Y_r2     = [float(ps[out]['r2']) for c, ps in sorted_performances]
      Y_r2_top = [float(ps[out]['r2_top']) for c, ps in sorted_performances]
      Y_r2_esb = [float(ps[out]['r2_esb']) for c, ps in sorted_performances]
      Y_cnt    = [float(ps[out]['raw_cnt']) for c, ps in sorted_performances]
      plot.plot_r2_esb(axes, X, Y_cnt, Y_r2, Y_r2_top, Y_r2_esb, idx, target, vertical)
    fig.tight_layout()
    plt.show()
    filename = "evp_%s.perf" % vertical
    pickle.dump(performances, open("%s/evp_%s/evp_%s.perf" % (ROOT, timestamp, vertical), "w"))
    all_performances[vertical] = performances
  return all_performances

def ensembled_verticals_performances(root, out, vertical, space, models, threshold=None):
  performances = {'ocpm':{}, 'cpc':{}, 'cpm':{}}
  for c in space:
    if (vertical == 'clients'): 
      cin = py.get_pinyin(c.decode('utf8'))
      path = "%s/%s/result_%s" % (root, vertical, cin)
    else:
      path = "%s/%s/result_%s" % (root, vertical, c)
    for target in num.btypes:
      model = models[target]
      # predict
      for f in glob.glob("%s/svr_%s_%s_*" % (path, target, out)):
        # select raw data
        if (vertical == 'clients'):
          rawf = raw_sort_delete_outlier(target, out, client=c)
        elif (vertical == 'industries'):
          rawf = raw_sort_delete_outlier(target, out, industry=c)
        elif (vertical == 'objectives'):
          rawf = raw_sort_delete_outlier(target, out, objective=c)
        if (len(rawf) < 100): continue
        # independent model performance
        Ts = timeit.default_timer()
        gs_svr, X_train, X_test, y_train, y_test, scaler = pickle.load(open(f))
        r2 = cal_verify_r2(gs_svr, scaler, X_test, y_test)
        # ensemble model performance
        results_top = compare_ensemble_performances(rawf, None, target, out, model=model, en_plot=False, conv=out, ensemble=False)
        results_ensemble = compare_ensemble_performances(rawf, None, target, out, model=model, en_plot=False, conv=out)
        perf = {
          'r2' : max(0, r2),
          'r2_top' : max(0, results_top['r2']),
          'r2_esb' : max(0, results_ensemble['r2']),
          'train': len(y_train),
          'test': len(y_test),
          'raw_cnt' : len(rawf),
        }
        print "[ensemble] for %s/%s/%s, ind/top/esp=%.4f/%.4f/%.4f among %i samples, in %.2f secs." % (c, target, out, r2, results_top['r2'], results_ensemble['r2'], len(rawf), timeit.default_timer() - Ts)
        if threshold: 
          if (r2 < threshold): continue
        if (vertical == 'clients'): c = cin
        if c in performances[target]:
          performances[target][c][out] = perf
        else:
          performances[target][c] = {out: perf}
  return performances
                      

#=================================================
#   verticals performance comparison
#================================================= 
def show_verticals_performances(path, out, verticals, threshold=None):
  for v, s in verticals:
    vertical, space = v, s
    performances = readfile_verticals_performances(path, vertical, space, threshold)
    # print performances
    fig, axes = plt.subplots(1, 3, figsize=(18, 4))
    for idx, target in enumerate(['ocpm', 'cpc', 'cpm']):
      # chosen_performances = [k for k in performances[target] if k[1][out]]
      sorted_performances = sorted(performances[target].items(), key=lambda k: k[1][out]['train'])
      X = [c for c, ps in sorted_performances]
      Y_r2 = [float(ps[out]['r2']) for c, ps in sorted_performances]
      Y_train = [int(ps[out]['train']) for c, ps in sorted_performances]
      plot.plot_r2_nos(axes, X, Y_r2, Y_train, idx, target, vertical)
    fig.tight_layout()
    plt.show()
  return performances

def readfile_verticals_performances(root, vertical, space, threshold=None):
  performances = {'ocpm':{}, 'cpc':{}, 'cpm':{}}
  for c in space:
    if (vertical == 'clients'): c = py.get_pinyin(c.decode('utf8'))
    path = "%s/%s/result_%s" % (root, vertical, c)
    for target in num.btypes:
      for out in num.convs:
        for f in glob.glob("%s/svr_%s_%s_*" % (path, target, out)):
          gs_svr, X_train, X_test, y_train, y_test, scaler = pickle.load(open(f))
          r2 = cal_verify_r2(gs_svr, scaler, X_test, y_test)
          fname = f.split('/')[-1]
          client = c.decode('utf8')
          perf = {
            'r2': r2, #re.search('r2_(.+?)_', fname).group(1),
            'train': len(y_train),
            'test': len(y_test),
            # 'client': client,
            # 'bid_type': target,
            # 'fname': fname
          }
          if threshold: 
            if (r2 < threshold): continue
          if c in performances[target]:
            performances[target][c][out] = perf
          else:
            performances[target][c] = {out: perf}
  return performances
                      

#=================================================
#   bid_price prediction results
#================================================= 
test_cases = [
    ['ocpm', 'lifetime_clicks', 'POST_ENGAGEMENT', 'trend_cpc_7d'],
    ['ocpm', 'conv_mobile_app_install', 'MOBILE_APP_INSTALLS', 'trend_cp_mobin_7d'],
    ['ocpm', 'conv_like', 'PAGE_LIKES', 'trend_cp_like_7d'],
    ['ocpm', 'conv_page_engagement', 'POST_ENGAGEMENT', 'trend_cp_enga_7d'],
]

def bid_price_prediction_performance(pack, bid_type='ocpm'):
  models = init_div_models(pack, prefix='new', threshold=0.5, silence=True) # init_models()
  model = models[bid_type]
  raw = dat.get_raw_sample(bid_type).filter(lifetime_clicks__gt=100)
  print "[raw] %i samples" % raw.count()
  results = []
  cats = {}
  for cat in ['board', 'markup', 'pred', 'predup', 'hybrid', 'hybridup', 'real']:
    cats[cat] = {'spent': 0, 'profit': 0, 'debt': 0}
  # board = {'spent': 0, 'profit': 0, 'debt': 0}
  # markup = {'spent': 0, 'profit': 0, 'debt': 0}
  # mpred = {'spent': 0, 'profit': 0, 'debt': 0}
  # mpredup = {'spent': 0, 'profit': 0, 'debt': 0}
  # real = {'spent': 0, 'profit': 0, 'debt': 0}
  for r in raw:
    res = {}
    y_pred = pred.predict_1_sample_ensemble(model, r)
    # if ((y_pred['lifetime_clicks'] > 0) & (y_pred['lifetime_impressions'] > 0) & (r.lifetime_clicks > 0)):
    if (r.lifetime_impressions > 0): # (r.lifetime_clicks > 0):
      res['board_ecpm'] = Decimal(r.trend_cpm_7d)
      res['board_ecpc'] = Decimal(r.trend_cpc_7d)
      res['pred_ecpm'] = Decimal(y_pred['lifetime_ecpm'])
      res['pred_ecpc'] = Decimal(y_pred['lifetime_ecpc'])
      res['real_ecpm'] = Decimal(r.lifetime_ecpm)
      res['real_ecpc'] = Decimal(r.lifetime_ecpc)
      res['markup_ecpm'] = res['board_ecpm']*Decimal(1.4)
      res['markup_ecpc'] = res['board_ecpc']*Decimal(1.4)
      res['predup_ecpm'] = res['pred_ecpm']*Decimal(1.2)
      res['predup_ecpc'] = res['pred_ecpc']*Decimal(1.2)
      # [Hybrid]
      for s in ['ecpm', 'ecpc']:
        if (res['pred_'+s] > res['board_'+s]*Decimal(2)): 
          res['hybrid_'+s] = res['board_'+s]*Decimal(2)
        elif (res['pred_'+s] < res['board_'+s]*Decimal(0.5)): 
          res['hybrid_'+s] = res['board_'+s]*Decimal(0.5)
        else:
          res['hybrid_'+s] = res['pred_'+s]
      res['hybridup_ecpm'] = res['hybrid_ecpm']*Decimal(1.2)
      res['hybridup_ecpc'] = res['hybrid_ecpc']*Decimal(1.2)
      results.append(res)
      # profits
      btp = 'ecpm' if r.bid_type in [2,6,7] else 'ecpc'
      cats['real']['spent'] += r.spent
      # loop
      for cat in ['board', 'markup', 'pred', 'predup', 'hybrid', 'hybridup']:
        cats[cat]['spent'] += r.spent * (res['%s_%s' % (cat, btp)]/res['real_%s' % btp])
        diff = r.spent * ((res['%s_%s' % (cat, btp)] - res['real_%s' % btp])/res['real_%s' % btp])
        if (diff > 0):
          cats[cat]['profit'] += diff
        else:
          cats[cat]['debt'] -= diff
      # # board
      # board['spent'] += r.spent * (res['board_%s' % btp]/res['real_%s' % btp])
      # if (res['board_%s' % btp] >= res['real_%s' % btp]):
      #   board['profit'] += r.spent * ((res['board_%s' % btp] - res['real_%s' % btp])/res['real_%s' % btp])
      # else:
      #   board['debt'] += r.spent * ((res['real_%s' % btp] - res['board_%s' % btp])/res['real_%s' % btp])
      # # markup = board*mrto
      # markup['spent'] += r.spent * (res['markup_%s' % btp]/res['real_%s' % btp])
      # if (res['markup_%s' % btp] >= res['real_%s' % btp]):
      #   markup['profit'] += r.spent * ((res['markup_%s' % btp] - res['real_%s' % btp])/res['real_%s' % btp])
      # else:
      #   markup['debt'] += r.spent * ((res['real_%s' % btp] - res['markup_%s' % btp])/res['real_%s' % btp])
      # # prediction
      # mpred['spent'] += r.spent * (res['pred_%s' % btp]/res['real_%s' % btp])
      # if (res['pred_%s' % btp] >= res['real_%s' % btp]):
      #   mpred['profit'] += r.spent * ((res['pred_%s' % btp] - res['real_%s' % btp])/res['real_%s' % btp])
      # else:
      #   mpred['debt'] += r.spent * ((res['real_%s' % btp] - res['pred_%s' % btp])/res['real_%s' % btp])
      # # prediction markup
      # mpredup['spent'] += r.spent * (res['predup_%s' % btp]/res['real_%s' % btp])
      # if (res['predup_%s' % btp] >= res['real_%s' % btp]):
      #   mpredup['profit'] += r.spent * ((res['predup_%s' % btp] - res['real_%s' % btp])/res['real_%s' % btp])
      # else:
      #   mpredup['debt'] += r.spent * ((res['real_%s' % btp] - res['predup_%s' % btp])/res['real_%s' % btp])
  print "-----[%s/%s]----------------------------" % (pack, bid_type)
  print "[Real] Spent:%.1fw" % (cats['real']['spent']/Decimal(1e4))
  for cat in ['board', 'markup', 'pred', 'predup', 'hybrid', 'hybridup']:
    print "[%s] Spent:%.1fw / Profit: %.1fw / Debt: %.1fw / Earn: %.1fw" % (cat, cats[cat]['spent']/Decimal(1e4), cats[cat]['profit']/Decimal(1e4), cats[cat]['debt']/Decimal(1e4), (cats[cat]['profit'] - cats[cat]['debt'])/Decimal(1e4))
  # print "[Board] Spent:%1.f / Profit: %1.f / Debt: %1.f / Earn: %1.f" % (board['spent'], board['profit'], board['debt'], (board['profit'] - board['debt']))
  # print "[Markup] Spent:%1.f / Profit: %1.f / Debt: %1.f / Earn: %1.f" % (markup['spent'], markup['profit'], markup['debt'], (markup['profit'] - markup['debt']))
  # print "[Pred] Spent:%1.f / Profit: %1.f / Debt: %1.f / Earn: %1.f" % (mpred['spent'], mpred['profit'], mpred['debt'], (mpred['profit'] - mpred['debt']))
  # print "[Predup] Spent:%1.f / Profit: %1.f / Debt: %1.f / Earn: %1.f" % (mpredup['spent'], mpredup['profit'], mpredup['debt'], (mpredup['profit'] - mpredup['debt']))
  print "----------------------------------------"
  # plot results
  for cv in ['ecpm', 'ecpc']:
    sorted_results = sorted(results, key=lambda k: k['real_%s' % cv])
    outlier = 0.05
    s_lower = int(len(sorted_results)*outlier)
    s_upper = int(len(sorted_results)*(1-outlier))
    samples = sorted_results[s_lower:s_upper]
    y_pred = []
    y_board = []
    y_real = []
    for s in samples:
      # y_pred.append(s['pred_%s' % cv])
      y_pred.append(s['hybrid_%s' % cv])
      y_board.append(s['board_%s' % cv])
      y_real.append(s['real_%s' % cv])
    y_pred = np.array(y_pred).astype(np.float)
    y_board = np.array(y_board).astype(np.float)
    y_real = np.array(y_real).astype(np.float)
    print "[board_%s] R2=" % cv, r2_score(y_board, y_real), ', RMSE=', train.rmse(y_board, y_real)
    plot.plot_predict(y_real, y_board, sort_both=True)
    print "[pred_%s] R2=" % cv, r2_score(y_pred, y_real), ', RMSE=', train.rmse(y_pred, y_real)
    plot.plot_predict(y_real, y_pred, sort_both=True)

