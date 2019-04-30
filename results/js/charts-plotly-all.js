
$( document ).ready(function() {
    file = urlParam('file');
    var loading = document.getElementById('loading');
    loading.setAttribute("style","color:red");
    
    function urlParam(name){
        var results = new RegExp('[\?&]' + name + '=([^&#]*)').exec(window.location.href);
        if (results==null) {
           return null;
        }
        return decodeURI(results[1]) || 0;
    }

    function loadFile(file,model) {
        file = "data/"+model+"/"+file
        $.get(file, function(csv) {
            var data = $.csv.toObjects(csv);
            plotData(data, file,model);
        });
    }
    

    function plotData(data,file_name,model){
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
            name:'value'
          };
          
          var prediction = {
            x: x,
            y: y_predicted,
            type: 'scatter',
            name:'prediction'
          };
          
          var data = [value, prediction];
          
          var layout = {
            title:file_name
          };
          
          Plotly.newPlot(model, data, layout, {showSendToCloud: true});
          loading.setAttribute("style","display:none");
    }

    loadFile(file,"arma")
    loadFile(file,"arima")
    loadFile(file,"cnn")
    loadFile(file,"lstm")
    loadFile(file,"lstmcnn_kerascombinantion")
    loadFile(file,"lstmcnn")
    loadFile(file,"sherlock-lstmcnn")

});

