
$( document ).ready(function() {
    file = urlParam('file');
    var loading = document.getElementById('loading');
    var loading_all = document.getElementById('loading_all');
    var link = document.getElementById('link');

    var a = document.createElement('a');
    var linkText = document.createTextNode("Get Google Charts");
    a.appendChild(linkText);
    a.href = "charts.html?file="+file;
    a.classList.add("btn")
    a.classList.add("btn-primary")
    a.class = "btn btn-primary"
    link.appendChild(a)
    loading.setAttribute("style","color:red");
    loading_all.setAttribute("style","color:red");
    
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
            plotAllData(data, file);
        });
    }
    
    loadFile(file)

    function plotData(data,file_name){
        x = []
        y_actual = []
        y_predicted = []
        y_prediction_training = []

        for(i=0; i<data.length;i++){
            x.push(data[i].timestamp)
            y_actual.push(data[i].value)
            y_predicted.push(data[i].prediction)
        }

        var max_value = y_actual.reduce(function(a, b) {
          return Math.max(a, b);
        });
        // if(max_value > 1){
        //     max_value =max_value*0.8
        // }

        for(i=0; i<data.length;i++){
            y_prediction_training.push(parseFloat(data[i].prediction_training)*max_value)
        }

        var value = {
            x: x,
            y: y_actual,
            type: 'scatter',
            name:'value'
          };
          
          var prediction = {
            x: x,
            y: y_predicted,
            type: 'scatter',
            name:'prediction'
          };
          
          var prediction_training = {
            x: x,
            y: y_prediction_training,
            type: 'bar',
            marker: {
                color: '#F2824D',
                opacity: 0.25,
            },
            name:'prediction_training'
          };

          var data = [value, prediction, prediction_training];
          
          var layout = {
            title:file_name
          };
          
          Plotly.newPlot('chart_div', data, layout, {showSendToCloud: true});
          loading.setAttribute("style","display:none");
    }

    function plotAllData(data,file_name){
        x = []
        y_actual = []
        y_predicted = []
        y_warp_distance = []
        y_label = []
        y_threshold = []
        y_positive_detection = []
        y_prediction_training = []
        y_threshold_training = []

        for(i=0; i<data.length;i++){
            x.push(data[i].timestamp)
            y_actual.push(parseFloat(data[i].value))
            y_predicted.push(parseFloat(data[i].prediction))
            y_warp_distance.push(parseFloat(data[i].warp_distance))
            y_threshold.push(parseFloat(data[i].distance_threshold))
        }

        var max_value = y_actual.reduce(function(a, b) {
            return Math.max(a, b);
        });
        // if(max_value > 1){
        //     max_value =max_value*0.8
        // }

        for(i=0; i<data.length;i++){
            y_label.push(data[i].label*max_value)
            y_positive_detection.push(parseFloat(data[i].positive_detection)*max_value)
            y_prediction_training.push(parseFloat(data[i].prediction_training)*max_value)
            y_threshold_training.push(parseFloat(data[i].threshold_training)*max_value)
        }

        var value = {
            x: x,
            y: y_actual,
            type: 'scatter',
            name:'value'
          };
          
          var prediction = {
            x: x,
            y: y_predicted,
            type: 'scatter',
            name:'prediction'
          };

          var warp_distance = {
            x: x,
            y: y_warp_distance,
            type: 'scatter',
            name:'warp_distance'
          };

          var label = {
            x: x,
            y: y_label,
            type: 'bar',
            marker: {
                color: 'rgb(255,0,0)',
                opacity: 0.3,
            },
            name:'label'
          };

          var threshold = {
            x: x,
            y: y_threshold,
            type: 'scatter',
            name:'threshold'
          };

          var positive_detection = {
            x: x,
            y: y_positive_detection,
            type: 'scatter',
            name:'positive_detection'
          };

          var prediction_training = {
            x: x,
            y: y_prediction_training,
            type: 'bar',
            marker: {
                color: '#F2824D',
                opacity: 0.25,
            },
            name:'prediction_training'
          };

          var threshold_training = {
            x: x,
            y: y_threshold_training,
            type: 'bar',
            marker: {
                color: '#9467BD',
                opacity: 0.25,
            },
            name:'threshold_training'
          }
          
          var data = [value, prediction, warp_distance, label, threshold, positive_detection, prediction_training, threshold_training];
          
          var layout = {
            title:file_name
          };
          
          Plotly.newPlot('chart_div_all', data, layout, {showSendToCloud: true});
          loading_all.setAttribute("style","display:none");
    }

});

