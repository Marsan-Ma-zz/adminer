#-*- coding:utf-8 -*-

import os, timeit, copy
import time, pickle

from datetime import datetime
from xpinyin import Pinyin

from lib import adhub_enum as num
from lib import svr_tasks as svt
# from cjobs import train_model_worker

py = Pinyin()
ROOT = '/home/marsan/workspace/adminer'

#===========================================
#   Divisions
#===========================================
def train_all_verticals():
  timestamp = datetime.now().strftime("%Y_%m%d_%H%M")
  os.system("mkdir %s/div_%s" % (ROOT, timestamp))
  os.system("cp lib/adhub_data.py %s/div_%s" % (ROOT, timestamp)) # backup for reference
  open('%s/div_%s/__init__.py' % (ROOT, timestamp), 'w').write('')
  #-----[Top Model]------------------------------------------
  path = "%s/div_%s/top" % (ROOT, timestamp)
  svt.adhub_train_all_model(path)
  # -----[objectives divide]-------------------------------------
  os.system("mkdir %s/div_%s/objectives" % (ROOT, timestamp))
  for c in num.div_objectives:
    path = "%s/div_%s/objectives/result_%s" % (ROOT, timestamp, c)
    svt.adhub_train_all_model(path, objective=c)
  #-----[industries divide]-------------------------------------
  os.system("mkdir %s/div_%s/industries" % (ROOT, timestamp))
  for c in num.div_industries:
    path = "%s/div_%s/industries/result_%s" % (ROOT, timestamp, c)
    svt.adhub_train_all_model(path, industry=c)
  #-----[clients divide]-------------------------------------
  os.system("mkdir %s/div_%s/clients" % (ROOT, timestamp))
  for c in num.div_clients:
    cin = py.get_pinyin(c.decode('utf8'))
    path = "%s/div_%s/clients/result_%s" % (ROOT, timestamp, cin)
    svt.adhub_train_all_model(path, client=c)

if __name__ == '__main__':
  # svt.adhub_train_all_model()
  train_all_verticals()
  


