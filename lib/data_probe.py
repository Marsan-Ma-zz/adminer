#-*- coding:utf-8 -*-

# basic
import os, timeit, copy
import math, json, operator, pickle
from operator import itemgetter   # for sorting list of list
from datetime import datetime   

# SVR
import numpy as np
from sklearn import metrics
from sklearn.ensemble import ExtraTreesRegressor

# marsan
from lib import schema as db      # database schema
from lib import adhub_enum as num
from lib import adhub_data as dat
from lib import svr_plot as plot
from lib import svr_predict as pred
from lib import svr_training as train

# nlp
import jieba.analyse

#===========================================
#   Label data qualify
#===========================================
def sample_atr_avg(samples, atr):
  vals = map(lambda i: getattr(i, atr), samples)
  vals = [v for v in vals if v > 0]
  vals_avg = (float(sum(vals)) / len(vals)) if len(vals) else -1
  return vals_avg

def qualify_labels():
  raw = db.adgroups.objects(valid=True,
            lifetime_unique_ctr__lte=0.05, 
            lifetime_unique_clicks__gte=50,
            lifetime_unique_clicks__lte=1e4, lifetime_unique_impressions__lte=1e6, 
            all_days__gte=3, month__gte=9)
  for t in num.categories:
    for c in num.convs:
      Ts = timeit.default_timer()
      targets = raw.filter(cr_category=t).only(c)
      others = raw.filter(cr_category__nin=[None, t]).only(c)
      target_avg = sample_atr_avg(targets, c)
      others_avg = sample_atr_avg(others, c)
      ratio = (target_avg / others_avg) if others_avg else -1
      print "%s / %s / target: %.2f (%i) / others: %.2f (%i) / [ratio: %.5f] / %.2f secs" % \
          (t, c, target_avg, targets.count(), others_avg, others.count(), ratio, timeit.default_timer() - Ts)

#===========================================
#   Feature selection by training one feature per model
#===========================================
ATR_VECS = [
      "random.random()",
      "s.max_bid", # if s.max_bid else 0,         
      "s.min_age",
      "s.max_age",
      "s.spent",
      "s.weekday",
      "s.month",
      # s.year - 2012,
      "s.all_days",
      # global environment
      "s.global_spent_page_type_3d",
      "s.global_spent_page_type_7d",
      "s.global_spent_page_type_28d",
      "s.global_imps_page_type_3d",
      "s.global_imps_page_type_7d",               
      "s.global_imps_page_type_28d",              
      "s.global_clks_page_type_3d",               
      "s.global_clks_page_type_7d",               
      "s.global_clks_page_type_28d",
      "estimated_reach",
      #
      "sel_conv(s, out)",
      #
      "num.get_ages(s.min_age, s.max_age)",
      "num.genders[s.gender]",
      "num.page_types[s.page_types]",
      "num.industry[s.client_industry]",
      "num.objectives[s.objective]",
      "num.get_story_action(s.cr_story)",
      "num.get_categories(s.cr_category)",
      "num.get_client_name(s.client_name)",
      "num.months[s.month]",
      "num.weekdays[s.weekday]",
]

def feature_select_by_training(out='lifetime_ecpm'):
  for bt in ['ocpm', 'cpc', 'cpm']:
    results = []
    raw = dat.get_raw_sample(bt, char='training')
    for fxt in ATR_VECS:
      X_samples, Y_samples, scaler = dat.data_prepare(bt, out, raw=raw, fxt=fxt)
      best_gs_svr, scaler, filename = train.svr_smart_search('', X_samples, Y_samples, scaler, remote=True)
      # print "=====[FEATURE_SELECT]===== %s/%s by %s: best_score= %.4f" % (bt, out, fxt, best_gs_svr.best_score_)
      results.append([bt, out, fxt, best_gs_svr.best_score_])
    print "=====[FEATURE_SELECT] Results for %s/%s by %i samples =====" % (bt, out, raw.count())
    for r in results:
      print "[FEATURE_SELECT] %s/%s by %s: best_score= %.4f" % (r[0], r[1], r[2], r[3])


#===========================================
#   Feature selection
#===========================================
ATRS = ['genders', 'page_types', 'industry', 'objectives', 'story_actions', 'categories', 'client_names', 'auto_tags', 'interests']
def cal_important_features(batch=10, threshold=1e-4):
  X_samples, Y_samples, scaler = dat.data_prepare('ocpm', 'lifetime_ecpm', outlier=0.05)
  tot_goot_atrs = {}
  for a in ATRS[5:]: tot_goot_atrs[a] = {}
  for i in np.arange(1,batch+1):
    Ts = timeit.default_timer()
    model = ExtraTreesRegressor(n_jobs=6)
    model.fit(X_samples, Y_samples)
    print "Totally %i features." % len(model.feature_importances_)
    print "[Labels] %i categories, %i interests, %i client_names, %i auto_tags" % (num.categories_len, num.interests_len, num.client_names_len, num.auto_tags_len)
    good_atrs = show_important_features(model.feature_importances_, threshold)
    for a in reversed(ATRS[5:]):
      for b in good_atrs[a]:
        if b in tot_goot_atrs[a]:
          tot_goot_atrs[a][b] += 1
        else:
          tot_goot_atrs[a][b] = 1
    print "%i batch finished in %.1f secs." % (i, (timeit.default_timer() - Ts))
    print "------------------------------------------------"
  # show performances
  for atr in reversed(ATRS[5:]):
    print "-------[%s]-----------------------" % atr
    for j in np.arange(1,batch+1):
      good_keys = [k for k,v in tot_goot_atrs[atr].items() if (v >= j)]
      print "%i keys occurs > %i times." % (len(good_keys), j)
  return tot_goot_atrs

def show_important_features(feature_importances, threshold):
  # collect
  idx_step = 7
  idxs = {}
  for a in ATRS:
    idxs[a] = idx_step
    if isinstance(getattr(num, a), dict):
      idx_step += len(getattr(num, a).values()[0])
    else:
      idx_step += len(getattr(num, a))
  print map(lambda b: "%s: %i" % (b, idxs[b]), ATRS)
  # print primitive labels
  print "[primitive importances] \n%s" % feature_importances[:7]
  for a in ATRS[:5]:
    if isinstance(getattr(num, a), dict):
      print "[%s importances] \n%s" % (a, feature_importances[idxs[a]:idxs[a]+len(getattr(num, a).values()[0])])
    else:
      print "[%s importances] \n%s" % (a, feature_importances[idxs[a]:idxs[a]+len(getattr(num, a))])

  # select minor labels
  good_idxs = [idx for idx,t in enumerate(feature_importances) if (t > threshold)]
  good_atrs = {}
  for a in ATRS[5:]: good_atrs[a] = []
  for idx in good_idxs:
    for a in reversed(ATRS[5:]):
      if (idx >= idxs[a]):
        # print idx, idxs[a], len(getattr(num, a))
        i = getattr(num, a)[idx-idxs[a]]
        good_atrs[a].append(i)
        break
  return good_atrs

#===========================================
#   Main Flow
#===========================================
if __name__ == '__main__':
  feature_select_by_training(out='lifetime_ecpm')
  feature_select_by_training(out='lifetime_ecpc')
# print "[Start] import @ " + str(datetime.now())
# # qualify_labels()
# cal_important_features()
# print "[Done] import @ " + str(datetime.now())


