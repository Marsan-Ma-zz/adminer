<style>
  /* beautiful checkbox */
  % include(root + 'assets/jquery-labelauty.css')
  /* ----- shared ----- */
  .txt_green  { color: green;     }
  .txt_blue   { color: darkblue;  }
  .txt_red    { color: red;       }
  .txt_orange { color: darkorange;}
  .txt_grey   { color: grey;}
  .bg_grey    { background-color: #eee; }
  /* ----- banner ----- */
  #banner h2 {
    margin: 5px 15px;
    display: inline-block;
  }
  #banner p {
    display: inline;
  }
  h4.title_splitter {
    margin: 10px;
    background-color: #eee;
    padding: 5px 10px;
    border: 1px solid #ccc;
  }
  #client_name {
    color: darkblue;
    font-weight: bold;
    text-transform: capitalize;
  }
  select {
    font-size: 15px;
    background-color: white;
  }
  button {
    font-size: 15px;
    font-weight: bold;
  }
  #atr_opt_result {
    background-color: #ddd;
  }
  #ad_link {
    display: none;
  }

  /* ----- variables ----- */
  #loadicon {
    position: fixed;
    top: 0;
    right: 20px;
    width: 200px;
    margin-left: -100px;
    z-index: 100;
  }
  .variables {
    border: 1px solid #ccc;
    width: 40%;
    margin: 10px;
    display: inline-block;  
  }
  .variables .variables_input {
    padding: 5px 30px;
    background-color: #eee;
  }
  .chart div {
    text-align: center;
    padding-bottom: 10px; 
  }
  /*.labelauty-unchecked-image, .labelauty-checked-image {display: none;}*/

  /* ----- summary table ----- */
  #atr_summary, #performance_summary {
    width: 40%;
    display: inline-block;
    border-collapse: collapse;
    border-spacing: 0;
    margin: 5px 10px;
  }
  #atr_summary td, #atr_summary th, #performance_summary td, #performance_summary th {
    padding: 4px 10px;
  }
  #tbl_desc p {
    line-height: 24px;
  }
  #atr_summary td input, #performance_summary td input {
    width: 100px;
  }
  #atr_max_bid td, #atr_opt_result td {
    font-weight: bold;
  }
  #fix_opt_result {
    display: none;
  }

  /* ----- chart ----- */
  .mpld3-figure {
    padding-left: 20px;
  }
  #loadicon, .mpld3-toolbar {display: none;}
  
</style>
