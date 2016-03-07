#-*- coding:utf-8 -*-
import os, datetime, copy
import parse_live as parse

#===========================================
#   Main Flow
#===========================================
if __name__ == '__main__':
  firstdump = False
  if firstdump:
    dump_end    = str(datetime.datetime.now())[:10]
    dump_start  = '2015-01-01'
    checkdate   = dump_start
  else:
    dump_end    = str(datetime.datetime.now())[:10]
    dump_start  = parse.date_shift(dump_end, -28)
    checkdate   = parse.date_shift(dump_end, -14)
  
  # sample_briefing()
  print "[Start] import from %s to %s, checkdate=%s @ %s" % (dump_start, dump_end, checkdate, parse.show_current_time())

  #====[data from adhub]=====
  parse.adhub_dump(dump_start, dump_end, checkdate, reset=firstdump)
  parse.fix_currency(checkdate)
  parse.prejoin_aggregate(checkdate, fullcheck=firstdump)
  parse.update_user_os(checkdate)
  parse.sanitize(dump_start)  # cross-out garbage ads before do conv_aggregate

  # ====[data from facebook]=====
  parse.update_adgroups_tokens()
  page2owner = parse.setup_pages_owner()
  parse.get_creatives(page2owner, checkdate)
  ## update_neighbors_stats(checkdate, empty_only=True)

  # ====[dailytrend]=====
  parse.update_daily_trend(dump_start, dump_end)
  parse.update_daily_trend_top(dump_start, dump_end)
  parse.update_daily_trend_mobile(dump_start, dump_end)
  parse.update_daily_trend_otype(dump_start, dump_end)
  parse.update_daily_trend_global(dump_start, dump_end)
  
  # ====[adgroup extra info prejoin for training]=====
  for depth in [2, 1, 0]:
    parse.update_adgroup_trend(dump_start, depth)
  parse.update_adgroup_global_trend(dump_start, dump_end)
  parse.sample_briefing()

  #====[verify options]=====
  ### verify_adgroup_dailys()  
  ### verify_adgroup_dailys_perlog()
  ### verify_adgroup_dailys_extra_info()
  # verify_daily_trends_page_types()
  
