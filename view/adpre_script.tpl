<script src="//ajax.googleapis.com/ajax/libs/jquery/1.9.1/jquery.min.js" ></script>
<script>
  // beautiful checkbox
  % include(root + 'assets/jquery-labelauty.js')
  $(":checkbox").labelauty({ label: false });

  //================================
  //  [Data Ctrl]
  //================================
  function printValue(sliderID, textbox) {
    var x = document.getElementById(textbox);
    var y = document.getElementById(sliderID);
    x.value = y.value;
  }
  function mpld3_load_lib(url, callback){
    var s = document.createElement('script');
    s.src = url;
    s.async = true;
    s.onreadystatechange = s.onload = callback;
    s.onerror = function(){console.warn("failed to load library " + url);};
    document.getElementsByTagName("head")[0].appendChild(s);
  }

  //================================
  //  [Plotter]
  //================================
  function mpld3_plot(chart_id, chart_json){
    if(typeof(mpld3) !== "undefined" && mpld3._mpld3IsLoaded){
      // already loaded: just create the figure
      !function(mpld3){
        mpld3.draw_figure(chart_id, chart_json);
      }(mpld3);
    }else if(typeof define === "function" && define.amd){
      // require.js is available: use it to load d3/mpld3
      require.config({paths: {d3: "https://mpld3.github.io/js/d3.v3.min"}});
      require(["d3"], function(d3){
        window.d3 = d3;
        mpld3_load_lib("https://mpld3.github.io/js/mpld3.v0.2.js", function(){
          mpld3.draw_figure(chart_id, chart_json);
        });
      });
    }else{
      // require.js not available: dynamically load d3 & mpld3
      mpld3_load_lib("https://mpld3.github.io/js/d3.v3.min.js", function(){
        mpld3_load_lib("https://mpld3.github.io/js/mpld3.v0.2.js", function(){
          mpld3.draw_figure(chart_id, chart_json);
        })
      });
    }  
  }

  //================================
  //  [Recommendation Demo]
  //================================
  var sample_industry = '';
  var target = {};
  var charts = {};
  var variables = ['min_age', 'max_age', 'weekday', 'month', 'max_bid', 
                  'spent', 'all_days', 'gender', 'bid_type', 'page_types'];
  var performances = ['lifetime_unique_clicks', 'lifetime_unique_impressions',
          'conv_mobile_app_install', 'conv_like', 'conv_page_engagement', 'conv_offsite_conversion', 'conv_link_click', 
          'conv_video_view', 'conv_video_play', 'conv_app_engagement', 'conv_receive_offer']

  function update_drag_bars(target){
    $(".variables input").each(function(idx){
      var id = $(this).attr('id');
      if (id.indexOf('_val') == -1){
        var va = target[id];
        // console.log(id + '/' + va);
        $(this).val(va);
        $(this).siblings("input").eq(0).val(va);  
      }
    });
    $(".variables select").each(function(idx){
      var id = $(this).attr('id');
      var va = target[id];
      // console.log(id + '/' + va);
      $(this).val(va);
    });
  }

  function update_summary_tables(target, best_set){
    variables.forEach(function(v){
      var var1 = '', var2 = '';
      val1 = (!target || (target[v] == '') || (target[v] == null)) ? '-' : target[v];
      if (!best_set) { $("#atr_" + v + " td:nth-child(2)").html(val1); }
      $("#atr_" + v + " td:nth-child(3) input").attr('placeholder', val1);
      val2 = (!best_set || (best_set[v] == '') || (best_set[v] == null)) ? '-' : best_set[v];
      $("#atr_" + v + " td:nth-child(4)").html(val2);
    });
    performances.forEach(function(v){
      var var1 = '', var2 = '';
      val1 = (!target || (target[v] == '') || (target[v] == null)) ? '-' : target[v];
      if (!best_set) { $("#atr_" + v + " td:nth-child(2)").html(val1); }
      $("#atr_" + v + " td:nth-child(3) input").attr('placeholder', val1);
      val2 = (!best_set || (best_set[v] == '') || (best_set[v] == null)) ? '-' : best_set[v];
      $("#atr_" + v + " td:nth-child(5)").html(val2);
    });
    if (!best_set) {
      $("#performance_summary td.txt_grey, #performance_summary td.txt_green, #performance_summary td.txt_orange").html('');
    }
  }

  function get_sample(){
    $("#loadicon").show();
    $('.labelauty').removeAttr('checked');
    $(".labelauty-unchecked-image, .labelauty-checked-image").hide();
    $("input.txt_orange").val('');
    var opt_target = $('#opt_target').val();
    sample_industry = $("#sample_industry").val();
    $.ajax({
      type: 'POST',
      url: "/get_sample",
      data: JSON.stringify({opt_target : opt_target, industry : sample_industry}),
      contentType: "application/json; charset=utf-8",
      dataType: 'json',
      success:function(resp) {
        target = resp;
        update_summary_tables(target, null);
        $("#client_name").html('['+target['client_industry']+'/'+target['objective']+'] '+target['client_name']);
        ['spent', 'bid_type', 'all_days'].forEach(function(v){
          $("#atr_"+v+" td:nth-child(3) input").val(target[v]);
        });
        // $("#atr_lifetime_unique_clicks td:nth-child(4) input").val(target['lifetime_unique_clicks']); // tmp
        update_drag_bars(target);
        get_recomment();
      },
    });
  }

  function update_charts(charts){
    $(".chart").empty();
    var fkeys = get_fix_vals(variables, 3);
    variables.forEach(function(v){
      if (Object.keys(fkeys).indexOf(v) == -1) {
        var chart_data = eval("charts." + v);
        if (chart_data != '') { mpld3_plot('chart_' + v, chart_data); }  
      }
    });
  }

  function get_fix_vals(items, slot){
    var fvals = {};
    // var fkeys = [];
    items.forEach(function(v){
      fv = $("#atr_" + v + " td:nth-child(" + slot + ") input").val();
      if (fv.length > 0) {
        fvals[v] = fv;
        // fkeys.push(v);
      }
    });
    return fvals;
  }


  function get_recomment(){
    $("#loadicon").show();
    $(".labelauty-unchecked-image, .labelauty-checked-image").hide();
    var t = $("#opt_target").val();
    $("#atr_opt_result td:nth-child(1)").html(t);
    $("#chart_target_opt").html(t);
    data = {
      opt_target: $('#opt_target').val(), 
      opt_action: $('#opt_action').val(), 
      chart_target: $('#chart_target').val(), 
      target_id: target.adgroup_id, 
      fix_vals: get_fix_vals(variables, 3),
      fix_acts: get_fix_vals(performances, 4),
    };
    $.ajax({
      type: 'POST',
      url: "/get_recomment",
      data: JSON.stringify(data),
      contentType: "application/json; charset=utf-8",
      dataType: 'json',
      success:function(resp) {
        if (resp.err) {
          alert(resp.err);
        } else {
          // console.log('[get_recomment] ajax completed.');
          update_summary_tables(target, resp.pred.best);
          // special field
          $("#atr_opt_result td:nth-child(2)").html(resp.pred.orig.score);
          $("#atr_opt_result td:nth-child(3)").html(resp.pred.targ.score);
          $("#atr_opt_result td:nth-child(5)").html(resp.pred.best.score);
          console.log(resp.pred.targ);
          console.log(resp.pred.best);
          performances.forEach(function(v){
            if (resp.pred.cint) {
              $("#atr_"+v+" td:nth-child(3)").html(parseInt(eval('resp.pred.targ.' + v + '.median')));  
              $("#atr_"+v+" td:nth-child(5)").html(parseInt(eval('resp.pred.best.' + v + '.median')));
            } else {
              $("#atr_"+v+" td:nth-child(3)").html(eval('resp.pred.targ.' + v));  
              $("#atr_"+v+" td:nth-child(5)").html(eval('resp.pred.best.' + v));
            }
            if (eval('resp.pred.targ.' + v) == 'N/A') {
              $("#csel_" + v).attr('disabled', 'true');
            } else {
              $("#csel_" + v).removeAttr('disabled');
            }
          });
          if (resp.fb_story) {
            $("#ad_link").attr("href", resp.fb_story).show();
          } else {
            $("#ad_link").hide();
          }
          // console.log(resp.pred.charts);
          if (resp.pred.charts) {
            update_charts(resp.pred.charts);  
          }
        }
        // other info
        $("#nonn").html(resp.pred.nonn);
        $("#loadicon").hide();
        $(".labelauty-unchecked-image, .labelauty-checked-image").show();
      },
    });
  }

  $(document).ready(function() {
    // get initial sample
    get_sample();
    // update sample
    $('#next_sample').on('click', function(){
      get_sample();
    });
    $('#re_calculate').on('click', function(){
      get_recomment();
    });
    $("#sample_industry").on('change', function(){
      get_sample();
    });
    $("#opt_target, #opt_action, #chart_target").on('change', function(){
      $("#atr_opt_result td:nth-child(1)").html($("#opt_target").val());
      get_recomment();
    });
    $(".labelauty-unchecked-image, .labelauty-checked-image").hide();
    $(".labelauty").click( function(){
      if( $(this).is(':checked') ) {
        var val = $(this).parent('td').siblings('td:nth-child(4)').html();
        $(this).parent('td').siblings('td:nth-child(3)').children('input').val(val);
      } else {
        $(this).parent('td').siblings('td:nth-child(3)').children('input').val('');
      }
      get_recomment();
    });


    // adjust value parameters & update prediction
    $(".variables input").each(function(idx){
      $(this).on('input', function(){
        var id = $(this).attr('id');
        var va = $(this).val();
        $(this).siblings("input").eq(0).val(va);
        target[id] = $(this).val();
        $("#fix_"+id).val(va);
      });
      $(this).on('mouseup', function(){
        get_recomment();
      });
    });
    // adjust label parameters & update prediction
    $(".variables select").each(function(idx){
      $(this).on('change', function(){
        var id = $(this).attr('id');
        var va = $(this).val();
        target[id] = $(this).val();
        get_recomment();
      });
    });
  });

</script>