# import libraries
import timeit
import numpy as np
import re
import datetime
from lib import schema as db      # database schema
from lib import adhub_enum as num
from lib import adhub_data as dat
import operator
import multiprocessing as mp


def date_shift(time_txt, days):
  time = datetime.datetime.strptime(time_txt, '%Y-%m-%d')
  time += datetime.timedelta(days=days)
  new_time = datetime.datetime.strftime(time, '%Y-%m-%d')
  return new_time


def fix_date_star(star):
  return fix_date(*star)

def fix_date(ts, tn):
  process_cnt = 0
  raw = db.adgroup_daily_convs.objects(ctime__gte=ts, ctime__lt=tn).timeout(False)
  raw_cnt = raw.count()
  print "[%s to %s] start, %i to go @ %s" % (ts, tn, raw_cnt, str(datetime.datetime.now())[11:19])
  for idx, r in enumerate(raw):
    if not ('time' in str(r.onDate.__class__)):
      r.update(set__onDate=r.onDate)
      process_cnt += 1
    # if (idx % 10000 == 0):
    #   print "[%s to %s] %i/%i done at %s" % (ts, tn, process_cnt, raw_cnt, str(datetime.datetime.now())[11:19])    
  print "[%s to %s] complete, %i/%i done @ %s" % (ts, tn, process_cnt, raw_cnt, str(datetime.datetime.now())[11:19])


def go_through_date(tstart, tend, func):
  p = mp.Pool(6)
  raws = []
  date = tstart
  while True:
    date_n = date_shift(date, 7)
    raws.append([date, date_n])
    date = date_n
    if (datetime.datetime.strptime(date_n, '%Y-%m-%d') > datetime.datetime.strptime(tend, '%Y-%m-%d')):
      break
  print raws
  p.map(fix_date_star, raws)


def add_object_type():
  for ot in num.object_types:
    raw_c = db.creatives.objects(object_type=ot)
    cids = map(lambda r: r.creative_id, raw_c.only('creative_id'))
    raw_a = db.adgroups.objects(creative_id__in=cids)
    raw_a.update(set__object_type=ot)
    adids = map(lambda r: r.adgroup_id, raw_a.only('adgroup_id'))
    db.adgroup_dailys.objects(adgroup_id__in=adids).update(set__object_type=ot)
    db.adgroup_daily_convs.objects(adgroup_id__in=adids).update(set__object_type=ot)
    print "[add_object_type] done %s" % ot

if __name__ == '__main__':
  # go_through_date('2014-01-01', '2015-06-01', fix_date_star)
  # add_object_type()
