file = urlParam('file');
  
var link = document.getElementById('link');

var a = document.createElement('a');
var linkText = document.createTextNode("Get Plot.ly charts");
a.appendChild(linkText);
a.href = "charts-plotly.html?file="+file;
link.appendChild(a)

google.charts.load('current', {'packages':['line', 'corechart']});
google.charts.setOnLoadCallback(function(){
  file = urlParam('file');
  loadFile(file);
});
// google.charts.setOnLoadCallback(initialize);

function drawChart(d,file_name) {
  
var button = document.getElementById('change-chart');
var generate_btn =  document.getElementById('generate-chart');
var chartDiv = document.getElementById('chart_div');
var loading = document.getElementById('loading');
loading.setAttribute("style","color:red");


var data = new google.visualization.DataTable();

keys = Object.keys(d[0]);
timeStampExists = false;
for (i = 0; i < keys.length; i++) {

  if (keys[i] == "timestamp") {
    // console.log(keys[i])
    data.addColumn('date', 'timestamp');
    timeStampExists = true;
  } else {
    // console.log(keys[i])
    data.addColumn('number', keys[i]);
  }
}

data_array = []
for (i = 0; i < d.length; i++){
  entry = []
  for (j = 0; j < keys.length; j++){
    if (keys[j] == "timestamp") {
      d[i]["timestamp"] = new Date(Date.parse(d[i]["timestamp"]));
      entry.push(d[i]["timestamp"]);
    } else {
      d[i][keys[j]] = parseFloat(d[i][keys[j]]);
      entry.push(d[i][keys[j]]);
    }
  }
  data_array.push(entry)
}


data.addRows(data_array)

var materialOptions = {
  chart: {
    title: file_name
  },
  width:  Math.round($(window).width()*0.8),
  height: Math.round($(window).height()*0.75),
  // series: 
  //   // Gives each series an axis name that matches the Y-axis below.
  //   0: {axis: 'Temps'},
  //   1: {axis: 'Daylight'}
  // },
  // axes: {
  //   // Adds labels to each axis; they don't have to match the axis names.
  //   y: {
  //     Temps: {label: 'Temps (Celsius)'},
  //     Daylight: {label: 'Daylight'}
  //   }
  // }
};

var classicOptions = {
  title: file_name,
  width:  Math.round($(window).width()*0.8),
  height: Math.round($(window).height()*0.75),
  // Gives each series an axis that matches the vAxes number below.
  // series: {
  //   0: {targetAxisIndex: 0},
  //   1: {targetAxisIndex: 1}
  // },
  // vAxes: {
  //   // Adds titles to each axis.
  //   0: {title: 'Temps (Celsius)'},
  //   1: {title: 'Daylight'}
  // },
  // hAxis: {
  //   ticks: [new Date(2014, 0), new Date(2014, 1), new Date(2014, 2), new Date(2014, 3),
  //           new Date(2014, 4),  new Date(2014, 5), new Date(2014, 6), new Date(2014, 7),
  //           new Date(2014, 8), new Date(2014, 9), new Date(2014, 10), new Date(2014, 11)
  //          ]
  // },
  // vAxis: {
  //   viewWindow: {
  //     max: 30
  //   }
  // }
};

function drawMaterialChart() {
  var materialChart = new google.charts.Line(chartDiv);
  materialChart.draw(data, materialOptions);
  button.innerText = 'Change to Classic';
  button.onclick = drawClassicChart;
}

function drawClassicChart() {
  var classicChart = new google.visualization.LineChart(chartDiv);
  classicChart.draw(data, classicOptions);
  button.innerText = 'Change to Material';
  button.onclick = drawMaterialChart;
}

drawMaterialChart();
loading.setAttribute("style","display:none");

function encode_as_link_svg() {
    // Add some critical information
    $("svg").attr({ version: '1.1' , xmlns:"http://www.w3.org/2000/svg"});

    var svg      = chartDiv.getElementsByTagName('svg')[0].outerHTML,
        b64      = btoa(svg),
        download = $("#downlaod-chart"),
        html     = download.html();
      
        // svg.attr({ version: '1.1' , xmlns:"http://www.w3.org/2000/svg"})

    var win = window.open();
    win.document.write("<a id='"+"downlaod-chart"+"' href-lang='image/svg+xml' href='data:image/svg+xml;base64,\n"+b64+"' download>Download SVG.If the link doesn't work try using Firefox</a>\n<br>\n<iframe src='data:image/svg+xml;base64,\n" + b64  + "' frameborder='0' style='border:0; top:0px; left:0px; bottom:0px; right:0px; width:80%; height:80%;' allowfullscreen></iframe>");

}
generate_btn.onclick = encode_as_link_svg;
}

function loadFile(file) {
  // var file = file_name;
  $.get(file, function(csv) {
      var data = $.csv.toObjects(csv);
      drawChart(data,file);
  });
}

function urlParam(name){
  var results = new RegExp('[\?&]' + name + '=([^&#]*)').exec(window.location.href);
  if (results==null) {
     return null;
  }
  return decodeURI(results[1]) || 0;
}