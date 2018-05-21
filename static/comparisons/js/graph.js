$(document).ready(function() {
  var trueip = Object.keys(comparisons)[0];
  var truedate = Object.keys(comparisons[trueip])[0];
  var selectedip = Object.keys(comparisons[trueip][truedate])[0];
  function drawchart() {
    $('#toptitle').text("Uknnown IP: " + trueip + " Unknown date: " + truedate + " Known IP: " + selectedip);
    dataseinmonth = Object.keys(comparisons[trueip][truedate][selectedip]);
    var sum = 0;
    data = []
    for( var i = 0; i < 24; i++ ){
      var arrx = []
      var arry = []
      for( var j = 0; j < dataseinmonth.length; j++ ){
        arrx.push(dataseinmonth[j])
        d = comparisons[trueip][truedate][selectedip][dataseinmonth[j]]
        if(d != -1)
        {
          arry.push(d[i][0])
        }
        //sum += parseInt( elmt[i], 10 ); //don't forget to add the base
      }
      var trace = {
        x: arrx,
        y: arry,
        name: i.toString(),
        type: 'bar'
      };
      data.push(trace)
    }
    var layout = {
      'xaxis': {'title': 'Days in known capture'},
      'yaxis': {'title': 'Classifier hour comparisons results'},
      'barmode': 'stack',
      'title': 'Similarity of unknown IP adress and date to known data'
    };
    Plotly.newPlot('trueIPDiv', data, layout);
  }
  $('#toptitle').text(trueip + " " + truedate + " " + selectedip);
  drawchart();
  function drawmenu(cList,dictofitems) {
    $.each(dictofitems, function (k, v) {
      var li = $('<li/>')
      .addClass('ui-menu-item')
      .attr('role', 'menuitem')
      .appendTo(cList);
      var aaa = $('<a/>')
      .addClass('ui-all')
      .text(k)
      .appendTo(li);
    });
  }
  var cList = $('#trueipdropdown ul')
  drawmenu(cList,comparisons);
  cList = $('#truedatadropdown ul')
  drawmenu(cList,comparisons[trueip]);
  cList = $('#knownipdropdown ul')
  drawmenu(cList,comparisons[trueip][truedate]);

  jQuery(".trueipitem").click(function(e){
    trueip = $(event.target).text();
    drawchart();
    e.preventDefault();
    });
    jQuery(".truedateitem").click(function(e){
      truedate = $(event.target).text();
      drawchart();
      e.preventDefault();
      });
    jQuery(".knownipitem").click(function(e){
      selectedip = $(event.target).text();
      drawchart();
      e.preventDefault();
      });

    
})
