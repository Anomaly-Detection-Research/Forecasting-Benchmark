// List All
function addAllItem(item){
  var tbody = document.getElementById("all");
  var tr = document.createElement("tr");
  var td_file = document.createElement("td");
  // var td_mse = document.createElement("td");
  // var td_params = document.createElement("td");

  var a = document.createElement('a');
  var linkText = document.createTextNode(item.file);
  a.appendChild(linkText);
  a.href = "charts-plotly-all.html?file="+item.file;
  a.target="_blank"
  td_file.appendChild(a)

  // var Text_mse = document.createTextNode(item.mse);
  // td_mse.appendChild(Text_mse)

  // var Text_params = document.createTextNode(item.parameters);
  // td_params.appendChild(Text_params)


  tr.appendChild(td_file)
  // tr.appendChild(td_mse)
  // tr.appendChild(td_params)

  tbody.appendChild(tr)
}

function generateAllList(list){
  for(i=0; i<list.length; i++){
    addAllItem(list[i])
  }
  $('#all-table').DataTable();
}

function alllist(){
    $.get("./arma_list.csv", function(csv) {
      var list = $.csv.toObjects(csv);
      generateAllList(list);
  });
}



alllist()

//List for each model
function addItem(ul_id,item){
  var tbody = document.getElementById(ul_id);
  var tr = document.createElement("tr");
  var td_file = document.createElement("td");
  var td_mse = document.createElement("td");
  var td_params = document.createElement("td");

  var a = document.createElement('a');
  var linkText = document.createTextNode(item.file);
  a.appendChild(linkText);
  a.href = "charts-plotly.html?file=data/"+ul_id+"/"+item.file;
  a.target="_blank"
  td_file.appendChild(a)

  var Text_mse = document.createTextNode(item.mse);
  td_mse.appendChild(Text_mse)

  var Text_params = document.createTextNode(item.parameters);
  td_params.appendChild(Text_params)


  tr.appendChild(td_file)
  tr.appendChild(td_mse)
  tr.appendChild(td_params)

  tbody.appendChild(tr)
}

function generateList(ul_id,list){
  for(i=0; i<list.length; i++){
    addItem(ul_id,list[i])
  }
  $('#'+ul_id+"-table").DataTable();
  
}


function loadFile(ul_id,file) {
  $.get(file, function(csv) {
      var list = $.csv.toObjects(csv);
      generateList(ul_id,list);
  });
}


loadFile("arma","./arma_list.csv",)
console.log("arma loaded")
loadFile("arima","./arima_list.csv")
console.log("arima loaded")
loadFile("lstm","./lstm_list.csv")
console.log("lstm loaded")
loadFile("cnn","./cnn_list.csv")
console.log("cnn loaded")
loadFile("lstmcnn","./lstmcnn_list.csv")
console.log("lstmcnn loaded")
loadFile("lstmcnn_kerascombinantion","./lstmcnn_kerascombinantion_list.csv")
console.log("lstmcnn_kerascombinantion loaded")
loadFile("sherlock-lstmcnn","./sherlock-lstmcnn_list.csv")
console.log("sherlock-lstmcnn loaded")


