#-*- coding:utf-8 -*-

import os, timeit, copy

from datetime import datetime
# from xpinyin import Pinyin

from lib import schema as db      # database schema
from lib import adhub_enum as num
from lib import adhub_data as dat
# from lib import svr_training as svr_train

from sklearn.cross_validation import _fit_and_score

# for sharing large inputs
import redis, pickle
red = redis.StrictRedis(host='YOUR_REDIS_HOST', port='YOUR_REDIS_PORT') #, db=0)

#----------------------------
#   Celery
#----------------------------
from celery import Celery
app = Celery('tasks')
app.config_from_object('celeryconfig')


#----------------------------
#   Jobs
#----------------------------
# tiny job for test
@app.task
def add(x, y):
  return x + y

# # train model
# @app.task
# def train_model_worker(target, out, industry=None, client=None, objective=None):
#   X_samples, Y_samples, scaler = dat.data_prepare(target, out, outlier=num.outlier, industry=industry, client=client, objective=objective)
#   if (len(Y_samples) < 100): 
#     return None
#   else:
#     filename_prefix = "%s_%s" % (target, out)
#     return svr_train.svr_smart_search(X_samples, Y_samples, scaler, filename_prefix=filename_prefix, remote=True)

# run one train
@app.task
def fas_mp(base_estimator, key, scorer, train, test, verbose,
                  parameters, fit_params, return_parameters, error_score):
  samples = pickle.loads(red.get(key))
  return _fit_and_score(base_estimator, samples['X'], samples['y'], scorer,
                                train, test, verbose, parameters,
                                fit_params, return_parameters=True,
                                error_score=error_score)
