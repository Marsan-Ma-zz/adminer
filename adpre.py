#-*- coding:utf-8 -*-

# web framework + server
import os, bottle, cherrypy, imp
import decimal, datetime
from decimal import *
import socket
from bottle import route, run, template, response
from bottle import get, post, request, debug
from bottle import static_file

# import libraries
import timeit
import copy
import json
import getpass
import numpy as np
from multiprocessing import Process, Queue

# marsan
from lib import schema as db        # database schema
from lib import adhub_enum as num
from lib import svr_tasks as svt    # scripts for SVR
from lib import adhub_data as dat
from lib import svr_predict as pred

PACK = 'rbf_2015_2m4_w_trend2' #'rfr_2015_q1'
ROOT = '/home/marsan/workspace/adminer/'
PFX = 'new'
models = {'cpc': {}, 'cpm': {}, 'ocpm': {}}

if os.path.islink("%s/server" % ROOT): os.remove("%s/server" % ROOT)
os.system("ln -s %s/%s %s/server;" % (ROOT, PACK, ROOT))
from server.adhub_data import fetch_record as fec
dat.fetch_record = fec

#===========================================
#   Initialize
#===========================================
def load_result_mp(bt, cv, fname, q, dv=None):
  svr, scaler, score = svt.svr_load_result(fname)
  out = {'bt': bt, 'cv': cv, 'svr': svr, 'scaler': scaler, 'dv': dv, 'score': score}
  q.put(out)

# load all models
def init_models():
  Ts = timeit.default_timer()
  queue = Queue()
  processes = []
  for bt in num.btypes:
    for cv in num.convs:
      # fname = '%sresult_gte9m_ccs_usd/%s_%s_%s.gs' % (ROOT, PFX, bt, cv)
      fname = '%s/result/%s_%s_%s.gs' % (ROOT, PFX, bt, cv)
      if os.path.isfile(fname):
        p = Process(target=load_result_mp, args=(bt, cv, fname, queue))
        p.start()
        processes.append(p)
  for p in processes:
    out = queue.get()
    models[out['bt']][out['cv']+'_svr'] = out['svr']
    models[out['bt']][out['cv']+'_scaler'] = out['scaler']
  for p in processes: p.join()
  print "[Initial] load models done in %.2f secs" % (timeit.default_timer() - Ts)
  # print models['cpc'].keys(), "\n\n", models['cpm'].keys(), "\n\n", models['ocpm'].keys()


#===========================================
#   Small Tasks
#===========================================
# patch JSON encoder cannot serialize numpy array
class CustumJsonEncoder(json.JSONEncoder):
  def default(self, obj):
    if isinstance(obj, np.ndarray):
      return obj.tolist() # or map(int, obj)
    elif isinstance(obj, decimal.Decimal):
      return float(obj)
    # try:
    return json.JSONEncoder.default(self, obj)
    # except:
    #   print "[CustumJsonEncoder TypeError]", obj.__class__

def custum_json_dump(raw):
  return json.dumps(raw, cls=CustumJsonEncoder)

def reconstruct_sample(params):
  atrs = {}
  print "[params]", params
  for a in ["client_name", "client_industry", 'gender', 'page_types', 'objective']:
    if (a in params): atrs[a] = params[a]
  for a in ['min_age', 'max_age', 'weekday', 'month', 'all_days', 'spent', 'max_bid', 'bid_type']:
    if (a in params): atrs[a] = float(params[a])
  if 'cr_category' in params:
    atrs['cr_category'] = params['cr_category'].lower().split('/')
  #
  dday = {'now': datetime.datetime.now()}
  for d in [3, 7, 28]:
    dday["%id" % d] = dday['now'] - datetime.timedelta(days=d)
    rpts = db.daily_trends._get_collection().aggregate([
      { "$match": { 
        'bid_type' : atrs['bid_type'], 
        'segname'  : 'page_types',
        'segment'  : atrs['page_types'],
        'onDate' : {"$gte" : dday["%id" % d]},
      }}, 
      { "$group": {
          "_id": atrs['bid_type'],
          "impressions": { "$sum": '$impressions'},
          "clicks": { "$sum": '$clicks'},
          "spent": { "$sum": '$spent'},
        }
      }
    ])['result']
    if (len(rpts) > 0):
      rpt = rpts[0]
      atrs['global_spent_page_type_%id' % d] = rpt['spent']
      atrs['global_imps_page_type_%id' % d] = rpt['impressions']
      atrs['global_clks_page_type_%id' % d] = rpt['clicks']
      atrs['trend_cpm_%id' % d] = float(rpt['spent'])*1000 / float(rpt['impressions']) if (rpt['impressions'] > 0) else None
      atrs['trend_cpc_%id' % d] = float(rpt['spent']) / float(rpt['clicks']) if (rpt['clicks'] > 0) else None
      atrs['lifetime_ecpm'] = 0
      atrs['lifetime_ecpc'] = 0
    for cv in ["mobile_app_install", "like", "page_engagement", "offsite_conversion", "link_click", "video_view", "video_play", "app_engagement", "receive_offer"]:
      rpts = db.daily_trends._get_collection().aggregate([
        { "$match": { 
          'bid_type' : atrs['bid_type'], 
          'segname'  : 'page_types',
          'segment'  : atrs['page_types'],
          'onDate'   : {"$gte" : dday["%id" % d]},
          cv         : {"$gte" : 0},
        }}, 
        { "$group": {
            "_id": 1,
            "spent": { "$sum": '$spent'},
            "cv": { "$sum": "$%s" % cv},
          }
        }
      ])['result']
      key = {
        'mobile_app_install'     : 'cp_mobin',
        'like'                   : 'cp_like', 
        'page_engagement'        : 'cp_enga',
        'offsite_conversion'     : 'cp_offcon',
        'link_click'             : 'cp_linkc',
        'video_view'             : None,
        'video_play'             : None, 
        'app_engagement'         : None,
        'receive_offer'          : None,
      }[cv]
      if (len(rpts) == 0):
        atrs["trend_%s_%id" % (key, d)] = None
      else:
        rpt = rpts[0]
        atrs["trend_%s_%id" % (key, d)] = float(rpt['spent']) / float(rpt['cv']) if ((rpt['spent'] > 0) & (rpt['cv'] > 0)) else None
  # fix dummy
  for cv in ["cp_mobin", "cp_like", "cp_enga", "cp_offcon", "cp_linkc"]:  
    atrs["trend_%s_3d" % cv] = atrs["trend_%s_3d" % cv] or atrs["trend_%s_7d" % cv] or atrs["trend_%s_28d" % cv]
    atrs["trend_%s_7d" % cv] = atrs["trend_%s_7d" % cv] or atrs["trend_%s_28d" % cv]
  print "[New sample]", atrs
  target = db.adgroups(**atrs)
  return target
  
#===========================================
#   Post API
#===========================================
@get('/')
def demo():
  return template('%s/view/demo.tpl' % ROOT)

@get('/favicons/:path#.+#')
def server_static(path):
  return static_file(path, root=ROOT+'/assets/favicons/')

@get('/images/:path#.+#')
def get_images(path):
  return static_file(path, root=ROOT+'/assets/images/')

@get('/assets/:path#.+#')
def get_assets(path):
  return static_file(path, root=ROOT+'/assets/')

@post('/get_sample')
def get_sample():
  Ts = timeit.default_timer()
  raw = dat.get_raw_sample(request.json['opt_target'], industry=request.json['industry'])
  target = raw[np.random.random_integers(1,raw.count())]
  print "[SPENT_0] %.2f" % target.spent
  print "[get_sample] request done in %.2f secs." % (timeit.default_timer() - Ts)
  return target.to_json()

@post('/get_recomment')
def get_recomment():
  Ts = timeit.default_timer()
  params = request.json
  opt_target = num.bid_types_n2s[int(params['opt_target'])]
  opt_action = params['opt_action']
  try:
    target = db.adgroups.objects(adgroup_id=long(params['target_id'])).first()
  except:
    target = reconstruct_sample(params)
  data = svt.get_recommends(target, models[opt_target], opt_target, opt_action, params['fix_vals'], params['fix_acts'], params['chart_target'])
  data_json = custum_json_dump(data)
  print "[get_recomment] request done in %.2f secs." % (timeit.default_timer() - Ts)
  return data_json

@post('/get_bid_price')
def get_bid_price():
  params = request.json
  data = {}
  try:
    ad = db.adgroups.objects(adgroup_id=long(params['target_id'])).first()
  except:
    ad = reconstruct_sample(params)
  y_pred = pred.predict_1_sample_ensemble(models[num.bid_types_n2s[ad.bid_type]], ad, price_adjust=True)
  data = {
    'trend_cpm_3d': ad.trend_cpm_3d,
    'trend_cpc_3d': ad.trend_cpc_3d,
    'trend_cpm_7d': ad.trend_cpm_7d,
    'trend_cpc_7d': ad.trend_cpc_7d,
    'trend_cpm_28d': ad.trend_cpm_28d,
    'trend_cpc_28d': ad.trend_cpc_28d,
    'lifetime_ecpm': ad.lifetime_ecpm,
    'lifetime_ecpc': ad.lifetime_ecpc,
    'pred_ecpm': y_pred['lifetime_ecpm'], 
    'pred_ecpc': y_pred['lifetime_ecpc'],
    'raw_imps': ad.lifetime_impressions,
    'raw_clks': ad.lifetime_clicks,
    'spent': ad.spent,
  }
  data_json = custum_json_dump(data)
  return data_json

#===========================================
#   Server
#===========================================
if __name__ == '__main__':
  models = svt.init_div_models(PACK, prefix=PFX, threshold=0.5, silence=True) # init_models()

  hostname = socket.gethostname()
  if (getpass.getuser() == 'marsan'):
    bottle.debug(True)  # mainly for disable template caching, DO NOT use in production!
    bottle.run(server='cherrypy', host='%s.azure.funptw' % hostname, port=6781, reloader=True, debug=True)
  else:   # startup by supervisor as root
    # bottle.debug(True)  # mainly for disable template caching, DO NOT use in production!
    bottle.run(server='cherrypy', host='%s.azure.funptw' % hostname, port=6780, reloader=True, debug=True)
