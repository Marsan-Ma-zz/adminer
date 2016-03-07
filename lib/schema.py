#-*- coding:utf-8 -*-
#==================================
#   [MongoEngine] http://mongoengine-odm.readthedocs.org/tutorial.html
#==================================
from mongoengine import *
from pymongo import MongoClient
import datetime

connect('YOUR_DATABASE_NAME', host='YOUR_MONGODB_HOST', port='YOUR_MONGODB_PORT')

#==================================
#   Campaign
#==================================
class campaigns(Document):
  oid                 = StringField(unique=True)
  # client info
  client_id           = StringField()
  client_name         = StringField()
  client_industry     = StringField()
  # main
  campaign_id         = StringField(unique=True)
  account_id          = StringField()
  campaign_status     = StringField()
  name                = StringField()
  daily_budget        = DecimalField()
  daily_imps          = IntField()
  lifetime_budget     = DecimalField()
  lifetime_imps       = IntField()
  budget_remaining    = DecimalField()
  start_time          = DateTimeField()
  end_time            = DateTimeField()
  # campaign_group_id   = StringField()   # null before 2014/7
  approved_time       = DateTimeField()
  buying_type         = StringField()
  rf_prediction_id    = StringField()   # null before 2014/7
  targeting           = DictField()
  bid_type            = StringField()
  bid_info            = StringField()
  utime               = DateTimeField()
  ctime               = DateTimeField()
  meta = {
    'indexes': [
      'oid',
      'utime',
      'ctime',
      'campaign_id',
    ]
  }


#==================================
#   AdGroup
#==================================
class adgroups(Document):
  oid                   = StringField(unique=True)
  # prejoin
  client_id             = StringField()
  client_name           = StringField()
  client_industry       = StringField()
  # main
  adgroup_id            = StringField(unique=True)
  ad_id                 = StringField()
  campaign_id           = StringField()
  account_id            = StringField()
  name                  = StringField()
  ad_status             = IntField()
  i_adgroup_status      = IntField()
  bid_type              = StringField()                        # V
  bid_info              = StringField()                     # VVV
  max_bid               = DecimalField()                    # V
  max_bid_ccf           = DecimalField()
  creative_id           = StringField()
  creative              = StringField()                     # V
  conversion_specs      = StringField()
  tracking_pixel_ids    = StringField()
  demolink_hash         = StringField()
  start_time            = DateTimeField()
  end_time              = DateTimeField()
  updated_time          = DateTimeField()
  disapprove_reason_descriptions  = StringField()
  link_url              = StringField()                     # V
  creative_hash         = StringField()                     
  targeting_hash        = StringField()                     
  tracking_specs        = StringField()
  descriptions          = StringField()                     # V
  my_max_bid            = StringField()  # custumer currency
  estimated_reach       = StringField()                     # V
  approved_time         = StringField()
  campaign_group_id     = StringField()
  adgroup_review_feedback   = StringField()
  failed_delivery_checks    = StringField()
  view_tags                 = StringField()                 # V
  objective                 = StringField()                 # VVV  '' means None, since Facebook add this field in 2014
  social_prefs              = StringField()                 # VVVVV
  event_status              = StringField()         
  stats_update_time         = StringField()
  engagement_audience       = StringField()                 # video view 才有, true並且view > 1000會自動幫你建retargeting audience
  adgroup_status            = StringField()
  # targeting flattened
  targeting                 = StringField()                     # VVVVV
  location                  = StringField()                     # V
  gender                    = StringField()                     # VVV
  gender_num                = IntField()                        # 0 as female, 1 as all, 2 as male
  min_age                   = IntField()                        # VVV
  max_age                   = IntField()                        # VVV
  interests                 = ListField()                       # VVV
  page_types                = StringField()                     # VVV
  user_os                   = StringField()     # mobile only
  # lifetime performance
  valid                                   = BooleanField(default=True)  # filter data usable or not
  good                                    = BooleanField(default=True)  # filter data usable or not
  spent                                   = DecimalField()
  # creatives
  account_raw               = StringField()
  tokens                    = ListField()
  object_story_id           = StringField()                     # VVV
  object_type               = StringField()                     # VVV
  cr_title                  = StringField()
  cr_body                   = StringField()
  cr_image_url              = StringField()
  cr_image_tags             = ListField()
  cr_link_url               = StringField()
  cr_story                  = StringField()
  cr_message                = StringField()
  cr_category               = ListField()
  cr_names                  = StringField()
  auto_tags                 = ListField()     # tags from cr_message
  img_pixels                = ListField()        
  # time info
  sdate                                   = DateTimeField()
  year                                    = IntField()
  month                                   = IntField()      # month number started at 2012/Jan
  weekday                                 = IntField()      # weekday of start_time
  s7days_uc_impressions                   = IntField()      # performance of staring 7 days
  s7days_uc_clicks                        = IntField()
  s7days_uc_ctr                           = DecimalField()
  s7days_spent                            = DecimalField()
  # daily info
  all_days                                = IntField()      # num of days where impression > 0
  all_ctrs                                = ListField()     
  all_clicks                              = ListField()
  all_impressions                         = ListField()
  all_unique_ctrs                         = ListField()
  all_unique_clicks                       = ListField()
  all_unique_impressions                  = ListField() 
  # confidence interval
  num_of_neighbors                        = IntField()
  # trend of 1 week before start
  trend_cpm_3d                            = DecimalField()
  trend_cpc_3d                            = DecimalField()
  trend_cp_mobin_3d                       = DecimalField()
  trend_cp_like_3d                        = DecimalField()
  trend_cp_enga_3d                        = DecimalField()
  trend_cp_offcon_3d                      = DecimalField()
  trend_cp_linkc_3d                       = DecimalField()
  trend_cp_vdoview_3d                     = DecimalField()
  trend_cp_vdoplay_3d                     = DecimalField()
  #
  trend_cpm_7d                            = DecimalField()
  trend_cpc_7d                            = DecimalField()
  trend_cp_mobin_7d                       = DecimalField()
  trend_cp_like_7d                        = DecimalField()
  trend_cp_enga_7d                        = DecimalField()
  trend_cp_offcon_7d                      = DecimalField()
  trend_cp_linkc_7d                       = DecimalField()
  trend_cp_vdoview_7d                     = DecimalField()
  trend_cp_vdoplay_7d                     = DecimalField()
  #
  trend_cpm_28d                           = DecimalField()
  trend_cpc_28d                           = DecimalField()
  trend_cp_mobin_28d                      = DecimalField()
  trend_cp_like_28d                       = DecimalField()
  trend_cp_enga_28d                       = DecimalField()
  trend_cp_offcon_28d                     = DecimalField()
  trend_cp_linkc_28d                      = DecimalField()
  trend_cp_vdoview_28d                    = DecimalField()
  trend_cp_vdoplay_28d                    = DecimalField()
  #
  global_spent_ta_3d                      = DecimalField()
  global_spent_ta_7d                      = DecimalField()
  global_spent_ta_28d                     = DecimalField()
  global_imps_ta_3d                       = DecimalField()
  global_imps_ta_7d                       = DecimalField()
  global_imps_ta_28d                      = DecimalField()
  global_clks_ta_3d                       = DecimalField()
  global_clks_ta_7d                       = DecimalField()
  global_clks_ta_28d                      = DecimalField()
  #
  global_spent_page_type_3d               = DecimalField()
  global_spent_page_type_7d               = DecimalField()
  global_spent_page_type_28d              = DecimalField()
  global_imps_page_type_3d                = DecimalField()
  global_imps_page_type_7d                = DecimalField()
  global_imps_page_type_28d               = DecimalField()
  global_clks_page_type_3d                = DecimalField()
  global_clks_page_type_7d                = DecimalField()
  global_clks_page_type_28d               = DecimalField()
  # global_spent_bid_type                   = DecimalField()
  # global_spent_objective                  = DecimalField()
  # trend of 1 month before start
  # conv actions
  lifetime_impressions                    = IntField()
  lifetime_clicks                         = IntField()
  lifetime_ecpm                           = DecimalField()    # spent / (lifetime_impressions / 1000)
  lifetime_ecpc                           = DecimalField()    # spent / lifetime_clicks
  conv_mobile_app_install                 = IntField()
  conv_like                               = IntField()
  conv_page_engagement                    = IntField()
  conv_offsite_conversion                 = IntField()
  conv_link_click                         = IntField()
  conv_video_view                         = IntField()
  conv_video_play                         = IntField()
  conv_app_engagement                     = IntField()
  conv_receive_offer                      = IntField()
  # lifetime
  lifetime_unique_impressions             = IntField()
  lifetime_unique_clicks                  = IntField()
  lifetime_unique_social_impressions      = IntField()
  lifetime_unique_social_clicks           = IntField()
  lifetime_total_unique_actions           = IntField()
  lifetime_total_unique_actions_1d_view   = IntField()
  lifetime_total_unique_actions_7d_view   = IntField()
  lifetime_total_unique_actions_28d_view  = IntField()
  lifetime_total_unique_actions_1d_click  = IntField()
  lifetime_total_unique_actions_7d_click  = IntField()
  lifetime_total_unique_actions_28d_click = IntField()
  lifetime_cost_per_unique_click          = DecimalField()
  lifetime_unique_ctr                     = DecimalField()
  lifetime_reach                          = IntField()
  lifetime_social_reach                   = IntField()
  utime                                   = DateTimeField()
  ctime                                   = DateTimeField()
  dummy                                   = StringField() # for test
  meta = {
    'indexes': [
      'oid',
      'utime',
      'ctime',
      'valid',
      'good',
      'adgroup_id',
      'campaign_id',
      'client_industry',
      'page_types',
      'min_age',
      'max_age',
      'spent',
      'all_days',
      'good',
      'user_os',
      'object_type',
      'creative_id',
    ]
  }


class adgroups_ulog(Document):
  account_id      = StringField()
  event_type      = StringField()
  actor_name      = StringField()
  event_time      = StringField()
  object_name     = StringField()
  object_id       = StringField()
  object_id_md5   = StringField()
  extra_data      = StringField()
  meta = {
    'indexes': [
      'event_type',
      'event_time',
      'object_id',
      'object_id_md5',
    ]
  }


#==================================
#   Creatives
#==================================
class creatives(Document):
  oid                 = StringField(unique=True)
  creative_id         = StringField()
  account_id          = StringField()
  mode                = IntField()
  title               = StringField()
  body                = StringField()
  image_hash          = StringField()
  image_url           = StringField()
  link_url            = StringField()
  name                = StringField()
  run_status          = StringField()
  preview_url         = StringField()
  count_current_adgroups = StringField()
  object_id           = StringField()
  story_id            = StringField()
  auto_update         = StringField()
  action_spec         = StringField()
  related_fan_page    = StringField()
  url_tags            = StringField()
  by_user_id          = StringField()
  previews            = StringField()
  actor_name          = StringField()
  object_story_id     = StringField()
  image_crops         = StringField()
  video_id            = StringField()
  actor_image_hash    = StringField()
  object_url          = StringField()
  actor_id            = StringField()
  object_store_url    = StringField()
  call_to_action_type = StringField()
  object_type         = StringField()
  utime               = DateTimeField()
  ctime               = DateTimeField()
  meta = {
    'indexes': [
      'oid',
      'utime',
      'ctime',
      # 'creative_id',
      'mode',
      'object_type',
    ]
  }


#==================================
#   AdGroup Daily Trend
#==================================
class daily_trends(Document):
  # info
  bid_type            = StringField()
  onDate              = DateTimeField()
  segname             = StringField()
  segment             = StringField()     # bid_type, industries
  objective           = StringField()
  # performances
  impressions         = IntField()
  clicks              = IntField()
  spent               = DecimalField()    # will change to US Cent, different than production db!
  spend               = DecimalField()    # for verify use, same as production db
  # convs
  mobile_app_install  = IntField()
  like                = IntField()
  page_engagement     = IntField()
  offsite_conversion  = IntField()
  link_click          = IntField()
  video_view          = IntField()
  video_play          = IntField()
  app_engagement      = IntField()
  receive_offer       = IntField()
  meta = {
    'indexes': [
      'bid_type',
      'onDate',
      'segname',
      'segment',
      'objective',
      ('bid_type', 'onDate', 'segment'),
    ]
  }


#==================================
#   AdGroup Daily Log
#==================================
class adgroup_dailys(Document):
  oid                             = StringField(unique=True)
  valid                           = BooleanField()
  # prejoin
  client_id                       = StringField()
  client_name                     = StringField()
  client_industry                 = StringField()
  bid_type                        = StringField()
  page_types                      = StringField()                     # VVV
  objective                       = StringField()
  user_os                         = StringField()                     # mobile only
  object_type                     = StringField()                     # VVV
  # pre-process
  latest                          = BooleanField(default=False) # true if latest record
  # id                              = StringField()
  adgroup_id                      = StringField()
  campaign_id                     = StringField()
  account_id                      = StringField()
  onDate                          = DateTimeField()
  impressions                     = IntField()
  clicks                          = IntField()
  spent                           = DecimalField()
  social_impressions              = IntField()
  social_clicks                   = IntField()
  social_spent                    = DecimalField()
  unique_impressions              = IntField()
  unique_clicks                   = IntField()
  social_unique_impressions       = IntField()
  social_unique_clicks            = IntField()
  actions                         = IntField()
  connections                     = IntField()
  etag                            = StringField()
  spend                           = DecimalField()
  unique_social_impressions       = IntField()
  unique_social_clicks            = IntField()
  total_actions                   = IntField()
  total_actions_1d_view           = IntField()
  total_actions_7d_view           = IntField()
  total_actions_28d_view          = IntField()
  total_actions_1d_click          = IntField()
  total_actions_7d_click          = IntField()
  total_actions_28d_click         = IntField()
  total_unique_actions            = IntField()
  total_unique_actions_1d_view    = IntField()
  total_unique_actions_7d_view    = IntField()
  total_unique_actions_28d_view   = IntField()
  total_unique_actions_1d_click   = IntField()
  total_unique_actions_7d_click   = IntField()
  total_unique_actions_28d_click  = IntField()
  cost_per_total_action           = DecimalField()
  cost_per_unique_click           = DecimalField()
  cpc                             = DecimalField()
  cpm                             = DecimalField()
  cpp                             = DecimalField()
  ctr                             = DecimalField()
  unique_ctr                      = DecimalField()
  reach                           = IntField()
  social_reach                    = IntField()
  frequency                       = DecimalField()
  lifetime_unique_impressions     = IntField()
  lifetime_unique_clicks          = IntField()
  lifetime_unique_social_impressions      = IntField()
  lifetime_unique_social_clicks           = IntField()
  lifetime_total_unique_actions           = IntField()
  lifetime_total_unique_actions_1d_view   = IntField()
  lifetime_total_unique_actions_7d_view   = IntField()
  lifetime_total_unique_actions_28d_view  = IntField()
  lifetime_total_unique_actions_1d_click  = IntField()
  lifetime_total_unique_actions_7d_click  = IntField()
  lifetime_total_unique_actions_28d_click = IntField()
  lifetime_cost_per_unique_click          = DecimalField()
  lifetime_unique_ctr                     = DecimalField()
  lifetime_reach                          = IntField()
  lifetime_social_reach                   = IntField()
  req_time                                = DateTimeField()
  on_date_utc                             = DateTimeField()
  utime                                   = DateTimeField()
  ctime                                   = DateTimeField()
  meta = {
    'indexes': [
      'valid',
      'bid_type',
      'page_types',
      'objective',
      'oid',
      'utime',
      'ctime',
      'latest',
      'campaign_id',
      'adgroup_id',
      # 'user_os',
      'object_type',
      'onDate',
      ('campaign_id', 'adgroup_id', 'onDate'),
    ]
  }


#==================================
#   AdGroup Daily Conversion Log
#==================================
class adgroup_daily_convs(Document):
  oid                           = StringField(unique=True)
  valid                         = BooleanField()
  # prejoin
  client_id                     = StringField()
  client_name                   = StringField()
  client_industry               = StringField()
  bid_type                      = StringField()
  page_types                    = StringField()                     # VVV
  objective                     = StringField()
  user_os                       = StringField()     # mobile only
  object_type                   = StringField()                     # VVV
  # pre-process
  latest                        = BooleanField(default=False) # true if latest record
  # main
  adgroup_id                    = StringField()
  campaign_id                   = StringField()
  onDate                        = DateTimeField()
  is_optimized                  = StringField()
  action_type                   = StringField()
  action_type_i                 = StringField()
  object_id                     = StringField()
  post_click_1d                 = IntField()
  post_click_7d                 = IntField()
  post_click_28d                = IntField()
  post_imp_1d                   = IntField()
  post_imp_7d                   = IntField()
  post_imp_28d                  = IntField()
  account_id                    = StringField()
  action_target_id              = StringField()
  inline_actions                = StringField()
  actions                       = StringField()
  actions_1d_view               = IntField()
  actions_7d_view               = IntField()
  actions_28d_view              = IntField()
  actions_1d_click              = IntField()
  actions_7d_click              = IntField()
  actions_28d_click             = IntField()
  actions_1d_view_by_convs      = IntField()
  actions_7d_view_by_convs      = IntField()
  actions_28d_view_by_convs     = IntField()
  actions_1d_click_by_convs     = IntField()
  actions_7d_click_by_convs     = IntField()
  actions_28d_click_by_convs    = IntField()
  action_values                 = DecimalField()
  action_values_1d_view         = DecimalField()
  action_values_7d_view         = DecimalField()
  action_values_28d_view        = DecimalField()
  action_values_1d_click        = DecimalField()
  action_values_7d_click        = DecimalField()
  action_values_28d_click       = DecimalField()
  action_values_1d_view_by_convs    = DecimalField()
  action_values_7d_view_by_convs    = DecimalField()
  action_values_28d_view_by_convs   = DecimalField()
  action_values_1d_click_by_convs   = DecimalField()
  action_values_7d_click_by_convs   = DecimalField()
  action_values_28d_click_by_convs  = DecimalField()
  unique_actions                    = IntField()
  total_actions_by_convs            = IntField()
  total_unique_actions_by_convs     = IntField()
  unique_actions_1d_view            = IntField()
  unique_actions_7d_view            = IntField()
  unique_actions_28d_view           = IntField()
  unique_actions_1d_click           = IntField()
  unique_actions_7d_click           = IntField()
  unique_actions_28d_click          = IntField()
  unique_actions_1d_click_by_convs  = IntField()
  unique_actions_7d_click_by_convs  = IntField()
  unique_actions_28d_click_by_convs = IntField()
  unique_actions_1d_view_by_convs   = IntField()
  unique_actions_7d_view_by_convs   = IntField()
  unique_actions_28d_view_by_convs  = IntField()
  cost_per_action_type              = IntField()
  req_time                          = DateTimeField()
  on_date_utc                       = DateTimeField()
  utime                                   = DateTimeField()
  ctime                                   = DateTimeField()
  meta = {
    'indexes': [
      'valid',
      'bid_type',
      'page_types',
      'objective',
      'oid',
      'utime',
      'ctime',
      'adgroup_id',
      'campaign_id',
      'onDate',
      'actions',
      'action_type',
      'action_values',
      'unique_actions',
      'total_actions_by_convs',
      'latest',
      'object_type',
      # 'user_os',
    ]
  }

