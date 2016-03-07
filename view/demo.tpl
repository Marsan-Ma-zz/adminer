% root = '/home/marsan/workspace/adminer/'
% include(root + 'view/header.tpl')
% actions = ['lifetime_unique_clicks', 'lifetime_unique_impressions', 'conv_mobile_app_install', 'conv_like', 'conv_page_engagement', 'conv_offsite_conversion', 'conv_link_click', 'conv_video_view', 'conv_video_play', 'conv_app_engagement', 'conv_receive_offer']

<body>
  <div id="page">
    <!-- banner & ctrl & summary -->
    <div id='banner'>
      <h2>[AdPredictor Demo]</h2>
      <p>
        Model:
        <select id='opt_target'>
          % for t, k in [['ocpm', 7], ['cpm', 2], ['cpc', 1]]:
            <option value="{{k}}">{{t}}</option>
          % end
        </select>

        Target:
        <select id='opt_action'>
          % for t in actions:
            <option value="{{t}}">{{t}}</option>
          % end
        </select>
        <button href="#" id="re_calculate">重新計算</button>
      </p>
      <h4 class='title_splitter'>
        取樣產業: 
        <select id='sample_industry'>
          % for d in ['all', 'gaming','retail','professionalservices','government', 'financialservices','automotive','travel','agency','EC','CPG','telecom', 'organizations','others','entertainmentmedia','education','technology']:
            <option value="{{d}}">{{d}}</option>
          % end
        </select>
        <button href="#" id="next_sample">重新取樣</button>
        <p id="client_name"></p>
        <a id='ad_link' target='_blank'>(看原始廣告)</a>
        <span id='nonn'></span>
      </h4>
      % include(root + 'view/adpre_rec_vals.tpl')
    </div>

    <!-- chart ctrl -->
    <h4 class='title_splitter'>
      做圖變數：
      <select id='chart_target'>
        % for d in actions:
          <option value="{{d}}" id='csel_{{d}}'>{{d}}</option>
        % end
        <option value="opt" id='chart_target_opt'>opt</option>
      </select>
    </h4>
    % include(root + 'view/adpre_ctrl_val.tpl', name='min_age', min='13', max='65', step='1', value='20')
    % include(root + 'view/adpre_ctrl_val.tpl', name='max_age', min='13', max='65', step='1', value='40')
    % include(root + 'view/adpre_ctrl_val.tpl', name='weekday', min='1', max='7', step='1', value='7')
    % include(root + 'view/adpre_ctrl_val.tpl', name='month', min='1', max='12', step='1', value='12')
    % include(root + 'view/adpre_ctrl_val.tpl', name='max_bid', min='10', max='1000', step='10', value='150')
    
    <div class='variables'>
      <div class='variables_input'>
        <label for="gender">gender</label>
        <select id='gender'>
          <option value="">0: All</option>
          <option value="1">1: Male</option>
          <option value="2">2: Female</option>
        </select>
      </div>
      <div id="chart_gender" class='chart'></div>
    </div>
    % include(root + 'view/adpre_ctrl_val.tpl', name='all_days', min='1', max='180', step='1', value='30')

    <div class='variables'>
      <div class='variables_input'>
        <label for="page_types">page_types</label>
        <select id='page_types'>
          <option value="">0: None</option>
          <option value="feed">1: feed</option>
          <option value="desktopfeed">2: desktopfeed</option>
          <option value="mobile">3: mobile</option>
          <option value="rightcolumn">4: rightcolumn</option>
          <option value="desktop">5: desktop</option>
          <option value="mobilefeed-and-external">6: mobilefeed-and-external</option>
          <option value="desktop-and-mobile-and-external">7: desktop-and-mobile-and-external</option>
          <option value="rightcolumn-and-mobile">8: rightcolumn-and-mobile</option>
        </select>
      </div>
      <div id="chart_page_types" class='chart'></div>
    </div>

    % include(root + 'view/adpre_ctrl_val.tpl', name='spent', min='100', max='150000', step='100', value='3000')
    
    <div class='variables'>
      <div class='variables_input'>
        <label for="bid_type">bid_type</label>
        <select id='bid_type'>
          <option value="1">0: CPC</option>
          <option value="2">1: CPM</option>
          <option value="5">2: MULTI_PREMIUM</option>
          <option value="6">3: RELATIVE_OCPM</option>
          <option value="7">4: ABSOLUTE_OCPM</option>
          <option value="9">5: CPA</option>
        </select>
      </div>
      <div id="chart_bid_type" class='chart'></div>
    </div>

    <img id="loadicon" src="https://s3-ap-northeast-1.amazonaws.com/piposay/images/loading_icon.gif" />
    
  </div>

  % include(root + 'view/adpre_style.tpl')
  % include(root + 'view/adpre_script.tpl')
</body>
