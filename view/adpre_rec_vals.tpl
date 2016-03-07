<table id='atr_summary' border=1>
  <tr class='bg_grey'>
    <th>Input</th>
    <th>History</th>
    <th><span class='txt_grey'>Predict</span> / <span class='txt_orange'>Overwrite</span></th>
    <th class='txt_green'>Recommend</th>
    <th>Accept</th>
  </tr>

  % for atr_name in ['min_age', 'max_age', 'weekday', 'month', 'all_days', 'gender', 'page_types', 'bid_type', 'spent', 'max_bid']:
    <tr id='atr_{{atr_name}}'>
      <td>{{atr_name}}</td>
      <td></td>
      <td><input class='txt_orange' id='fix_{{atr_name}}'/></td>
      <td class='txt_green'></td>
      <td><input type="checkbox"/></td>
    </tr>
  % end

  <tr>
    <td colspan="5" id='tbl_desc'>
      <p><b>[History]</b>: 取樣一支過去投放過的廣告<br></p>
      <p class='txt_orange'><b>[Overwrite]</b>: 覆寫History，作為Prediction的依據<br></p>
      <p class='txt_grey'><b>[Predict]</b>: 根據History+Overwrite的預測結果<br></p>
      <p class='txt_blue'><b>[Lock]</b>: 鎖定Recomment的成效底限<br></p>
      <p class='txt_green'><b>[Recommend]</b>: 根據Predict，建議更佳的設定<br></p>
    </td>
  </tr>

</table>

<table id='performance_summary' border=1>
  <tr class='bg_grey'>
    <th>Output</th>
    <th>History</th>
    <th class='txt_grey'>Predict</th>
    <th class='txt_blue'>Lock</th>
    <th class='txt_green'>Recommend</th>
  </tr>

  % for atr_name in ['lifetime_unique_impressions', 'lifetime_unique_clicks', 'conv_mobile_app_install', 'conv_like', 'conv_page_engagement', 'conv_offsite_conversion', 'conv_link_click', 'conv_video_view', 'conv_video_play', 'conv_app_engagement', 'conv_receive_offer', 'opt_result']:
    <tr id='atr_{{atr_name}}'>
      <td>{{atr_name}}</td>
      <td></td>
      <td class='txt_grey'></td>
      <td><input class='txt_blue' id='fix_{{atr_name}}' placeholder='-'/></td>
      <td class='txt_green'></td>
    </tr>
  % end
</table>

