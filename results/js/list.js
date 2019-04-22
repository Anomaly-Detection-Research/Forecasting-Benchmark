function addItem(ul_id,item){
	var ul = document.getElementById(ul_id);
  var li = document.createElement("li");
  var a = document.createElement('a');
  var linkText = document.createTextNode(item.file);
  var outText = document.createTextNode(" | mse:"+item.mse+" | params:"+item.parameters);
  a.appendChild(linkText);
  a.href = "charts-plotly.html?file=data/"+ul_id+"/"+item.file;
  li.appendChild(a);
  li.appendChild(outText);
  ul.appendChild(li);
}

function generateList(ul_id,list){
  for(i=0; i<list.length-1; i++){
    addItem(ul_id,list[i])
  }

}


function loadFile(ul_id,file) {
  $.get(file, function(csv) {
      var list = $.csv.toObjects(csv);
      generateList(ul_id,list);
  });
}


loadFile("arma","./arma_list.csv")
loadFile("arima","./arima_list.csv")
loadFile("lstm","./lstm_list.csv")
loadFile("cnn","./cnn_list.csv")
loadFile("lstmcnn","./lstmcnn_list.csv")