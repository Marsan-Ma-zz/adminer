#-*- coding:utf-8 -*-

# basic
import os, timeit, copy, time
import math, json, operator
import numpy as np
import multiprocessing as mp
from operator import itemgetter   # for sorting list of list
from datetime import datetime 

# marsan
from lib import schema as db
from lib import adhub_enum as num
from lib import adhub_data as dat
from lib import svr_plot as plot
from lib import svr_predict as pred

mp_out = mp.Queue()

#=================================================
#   small tasks
#================================================= 
def find_ads_alike(ad, d_age=3, d_spent=5000, d_days=5, max_num=None, opt_action=None):
  # print ad._data
  if (ad.client_industry == ''): ad.client_industry = None
  neighbors = db.adgroups.objects(
    valid=True,
    # equal
    bid_type=ad.bid_type,
    client_industry=ad.client_industry,
    # page_types=ad.page_types,
    # range
    min_age__gte=ad.min_age-d_age,
    min_age__lte=ad.min_age+d_age,
    max_age__gte=ad.max_age-d_age,
    max_age__lte=ad.max_age+d_age,
    spent__gte=ad.spent-d_spent,
    spent__lte=ad.spent+d_spent,
    all_days__gte=ad.all_days-d_days,
    all_days__lte=ad.all_days+d_days,
  )
  if max_num:
    if (neighbors.count() > max_num):
        neighbors = neighbors.filter(gender=ad.gender)
    if (neighbors.count() > max_num):
        neighbors = neighbors.filter(objective=ad.objective)
    if (neighbors.count() > max_num):
        neighbors = neighbors.filter(month=ad.month)
    if (neighbors.count() > max_num):
        neighbors = neighbors[:max_num]
  if opt_action:
    neighbors = neighbors.exclude('img_pixels', 'cr_story', 'cr_message', 'all_ctrs', 'all_clicks', 'all_impressions', 'all_unique_ctrs', 'all_unique_clicks', 'all_unique_impressions')
    neighbors = neighbors.order_by("-"+opt_action)
  return neighbors

def get_avg_neighbors_error_rate(model, ad, convs=None):
  nbors = find_ads_alike(ad)
  nbors_cnt = nbors.count()
  if not convs: convs = num.convs
  ers = {}
  for c in convs: ers[c] = []
  for n in nbors:
    result = pred.predict_1_sample_ensemble(model, n, convs)
    for c in convs:
      resp = result[c]
      real = getattr(n, c)
      erate = ((resp - real)/max(1, real))**2
      ers[c].append(erate)
  # sort & remove outlier
  for c in convs:
    ers[c] = sorted(ers[c])
    ers_cnt = len(ers[c])
    ers[c] = ers[c][int(ers_cnt*0.1):int(ers_cnt*0.9)]
    ers[c] = np.average(ers[c])
  return ers

#=================================================
#   small tasks
#================================================= 
# def find_ads_alike(ad, d_age=3, d_spent=1000, d_days=3):
#   x = copy.deepcopy(ad)

#   if (neighbors.count() > max_num):
#       neighbors = neighbors.filter(gender=ad.gender)
#   if (neighbors.count() > max_num):
#       neighbors = neighbors.filter(objective=ad.objective)
#   if (neighbors.count() > max_num):
#       neighbors = neighbors.filter(month=ad.month)
#   if (neighbors.count() > max_num):
#       neighbors = neighbors[:max_num]
#   return neighbors

