#-*- coding:utf-8 -*-

from abc import ABCMeta, abstractmethod
from collections import Sized

import time, pickle, random
import numpy as np

from sklearn.base import BaseEstimator, is_classifier, clone
from sklearn.cross_validation import _check_cv as check_cv
from sklearn.cross_validation import _fit_and_score
from sklearn.externals.joblib import Parallel, delayed
from sklearn.utils.validation import _num_samples, indexable
from sklearn.metrics.scorer import check_scoring
from sklearn.grid_search import ParameterGrid, _CVScoreTuple

from celery import group
import cjobs

# redis for storing static X, y
from datetime import datetime
import redis
red = redis.StrictRedis(host='YOUR_REDIS_HOST', port=6789) #, db=0)

def _fit(self, X, y, parameter_iterable, en_celery=False):
  """Actual fitting,  performing the search over parameters."""

  estimator = self.estimator
  cv = self.cv
  self.scorer_ = check_scoring(self.estimator, scoring=self.scoring)

  n_samples = _num_samples(X)
  X, y = indexable(X, y)

  if y is not None:
      if len(y) != n_samples:
          raise ValueError('Target variable (y) has a different number '
                           'of samples (%i) than data (X: %i samples)'
                           % (len(y), n_samples))
  cv = check_cv(cv, X, y, classifier=is_classifier(estimator))

  if self.verbose > 0:
      if isinstance(parameter_iterable, Sized):
          n_candidates = len(parameter_iterable)
          print("Fitting {0} folds for each of {1} candidates, totalling"
                " {2} fits".format(len(cv), n_candidates,
                                   n_candidates * len(cv)))

  base_estimator = clone(self.estimator)

  pre_dispatch = self.pre_dispatch

  if en_celery:
    out = []
    timestamp = timestamp = datetime.now().strftime("%Y%m%d%H%M%s")
    key = "sample_%s_%s" % (timestamp, int(round(random.random(), 8)*1e8))
    red.set(key, pickle.dumps({'X': X, 'y': y}))
    grp = group(cjobs.fas_mp.s(clone(base_estimator), key, self.scorer_,
                                train, test, self.verbose, parameters,
                                self.fit_params, return_parameters=True,
                                error_score=self.error_score)
            for parameters in parameter_iterable
            for train, test in cv)()
    out = grp.get()
    red.delete(key)
  else:
    out = Parallel(
        n_jobs=self.n_jobs, verbose=self.verbose,
        pre_dispatch=pre_dispatch
    )(
        delayed(_fit_and_score)(clone(base_estimator), X, y, self.scorer_,
                                train, test, self.verbose, parameters,
                                self.fit_params, return_parameters=True,
                                error_score=self.error_score)
            for parameters in parameter_iterable
            for train, test in cv)

  # Out is a list of triplet: score, estimator, n_test_samples
  n_fits = len(out)
  n_folds = len(cv)

  scores = list()
  grid_scores = list()
  for grid_start in range(0, n_fits, n_folds):
      n_test_samples = 0
      score = 0
      all_scores = []
      for this_score, this_n_test_samples, _, parameters in \
              out[grid_start:grid_start + n_folds]:
          all_scores.append(this_score)
          if self.iid:
              this_score *= this_n_test_samples
              n_test_samples += this_n_test_samples
          score += this_score
      if self.iid:
          score /= float(n_test_samples)
      else:
          score /= float(n_folds)
      scores.append((score, parameters))
      # TODO: shall we also store the test_fold_sizes?
      grid_scores.append(_CVScoreTuple(
          parameters,
          score,
          np.array(all_scores)))
  # Store the computed scores
  self.grid_scores_ = grid_scores

  # Find the best parameters by comparing on the mean validation score:
  # note that `sorted` is deterministic in the way it breaks ties
  best = sorted(grid_scores, key=lambda x: x.mean_validation_score,
                reverse=True)[0]
  self.best_params_ = best.parameters
  self.best_score_ = best.mean_validation_score

  if self.refit:
      # fit the best estimator using the entire dataset
      # clone first to work around broken estimators
      best_estimator = clone(base_estimator).set_params(
          **best.parameters)
      if y is not None:
          best_estimator.fit(X, y, **self.fit_params)
      else:
          best_estimator.fit(X, **self.fit_params)
      self.best_estimator_ = best_estimator
  return self



def fit(self, X, y=None, en_celery=False):
  return _fit(self, X, y, ParameterGrid(self.param_grid), en_celery=en_celery)

