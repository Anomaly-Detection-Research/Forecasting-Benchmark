
$( document ).ready(function() {
    console.log( "ready!" );
    file = urlParam('file');
    console.log(file)
    var loading = document.getElementById('loading');
    var link = document.getElementById('link');

    var a = document.createElement('a');
    var linkText = document.createTextNode("Get Google Charts");
    a.appendChild(linkText);
    a.href = "charts.html?file="+file;
    link.appendChild(a)
    loading.setAttribute("style","color:red");
    
    function urlParam(name){
        var results = new RegExp('[\?&]' + name + '=([^&#]*)').exec(window.location.href);
        if (results==null) {
           return null;
        }
        return decodeURI(results[1]) || 0;
    }

    function loadFile(file) {
        $.get(file, function(csv) {
            var data = $.csv.toObjects(csv);
            plotData(data, file);
        });
    }
    
    loadFile(file)

    function plotData(data,file_name){
        console.log(data)
        x = []
        y_actual = []
        y_predicted = []

        for(i=0; i<data.length;i++){
            x.push(data[i].timestamp)
            y_actual.push(data[i].value)
            y_predicted.push(data[i].prediction)
        }
        var value = {
            x: x,
            y: y_actual,
            type: 'scatter',
          };
          
          var prediction = {
            x: x,
            y: y_predicted,
            type: 'scatter'
          };
          
          var data = [value, prediction];
          
          var layout = {
            title:file_name
          };
          
          Plotly.newPlot('chart_div', data, layout, {showSendToCloud: true});
          loading.setAttribute("style","display:none");
    }

});

