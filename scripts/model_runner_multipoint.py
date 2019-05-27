import json
import pandas
import math
import sys
import os
import numpy as np
import re
import helpers
# import models.armaModel as arma
# import models.arima as arima
# import models.lstm as lstm
# import models.cnn as cnn
# import models.lstmcnn as lstmcnn
# import models.lstmcnn_kerascombinantion as lstmcnn_kerascombinantion

from models.mutipoint.modelFactory import modelFactory

# args
csv_input_directory = "../data"
csv_output_directory = "../results"

# parameters
defualt_training_ratio = 0.1
parameters = {
    # for all
    "training_ratio":defualt_training_ratio,
    "no_of_prediction_points":5,

    # for ARMA and ARIMA
    "ar_max":4,
    "d_max":1,
    "ma_max":4,

    # for LSTM and CNN
    "epochs":15, 
    "batch_size":1,
    "sequance_length":10,
    "DL1units":20, 
    "DL2units":15, 
    "DL3units":5,
    
    # for LSTM and LSTMCNN
    "lstmCells":10,

    # for CNN and LSTMCNN
    "CL1filters":1, 
    "CL1kernal_size":2, 
    "CL1strides":1, 
    "PL1pool_size":1, 
    
    # for LSTMCNN
    "lstmWeight":0.5,
    "cnnWeight":0.5,
    "CNNDL1units":20,
    "CNNDL2units":15,
    "CNNDL3units":5,
    "LSTMDL1units":20,
    "LSTMDL2units":15,
    "LSTMDL3units":5
}

# uncoment below line to set models and set getting_models = False
# models = ["arma"] #done
models = ["multi_lstm"]
# getting_models = True
getting_models = False

while(getting_models):
    print("Select model to run")
    print("1 - exit")
    print("2 - all")
    print("3 - multi_arma")
    print("4 - multi_arima")
    print("5 - multi_cnn")
    print("6 - multi_lstm")
    print("7 - multi_lstmcnn")
    print("8 - multi_lstmcnn_kerascombinantion")
   
    user_input = input("(Enter number) : ")
    try:
        val = int(user_input)
        if(val == 1):
            exit()
        elif(val == 2):
            models = ["multi_arma","multi_arima","multi_lstm","multi_cnn","multi_lstmcnn","multi_lstmcnn_kerascombinantion"]
            getting_models = False
        elif(val == 3):
            models = ["multi_arma"]
            getting_models = False
        elif(val == 4):
            models = ["multi_arima"]
            getting_models = False
        elif(val == 5):
            models = ["multi_cnn"]
            getting_models = False
        elif(val == 6):
            models = ["multi_lstm"]
            getting_models = False
        elif (val == 7):
            models = ["multi_lstmcnn"]
            getting_models = False
        elif (val == 8):
            models = ["multi_lstmcnn_kerascombinantion"]
            getting_models = False
        else:
            print("Invalid argument")
    except ValueError:
        print("Invalid argument")



# get all csv files in input directory
reg_x = re.compile(r'\.(csv)')
csv_input_files = []
for path, dnames, fnames in os.walk(csv_input_directory):
    csv_input_files.extend([os.path.join(path, f) for f in fnames if reg_x.search(f)])

csv_input_files.sort()
modelFactoryInstance = modelFactory()
for m in models:
    output_files = []
    output_files.append("file,mse,parameters")
    nan_output_files = []
    nan_output_files.append("file")
    helpers.dump_results(output_files, csv_output_directory,m)

    print("##### ["+m+"]"+ str(len(csv_input_files)) + " CSV input files to process #####")
    count = 1
    for f in csv_input_files:
        print("Processing ["+m+"]" + f)
        if f.split("/")[-1] == "electricity_utilization_specific_user":
            parameters["training_ratio"] = 0.58979536887
        else :
            parameters["training_ratio"] = defualt_training_ratio
        # fetching input file data
        dataframe_expect = pandas.read_csv(f)
        value = np.array(dataframe_expect['value'])
        timestamp = np.array(dataframe_expect['timestamp'])
        
        parameters["data"] = value

        # running model
        model = modelFactoryInstance.get_model(m,parameters)
        model.train()

        if(m=="multi_arma"): # ARMA model

            ar, ma, prediction = model.get_output()

            params = "ar=" + str(ar)
            params += ";ma=" + str(ma)
            params += ";training_ratio=" + str(parameters["training_ratio"])
            params += ";no_of_prediction_points="+str(parameters["no_of_prediction_points"])
        
        elif(m=="multi_arima"): # ARIMA model

            ar,d,ma,prediction = model.get_output()

            params = "ar=" + str(ar)
            params += ";d=" + str(d)
            params += ";ma=" + str(ma)
            params += ";training_ratio=" + str(parameters["training_ratio"])
            params += ";no_of_prediction_points="+str(parameters["no_of_prediction_points"])

        elif(m=="multi_lstm"): # LSTM model
            
            prediction = model.get_output()

            params = "lstmCells="+str(parameters["lstmCells"])
            params += ";DL1units="+str(parameters["DL1units"])
            params += ";DL2units="+str(parameters["DL2units"])
            params += ";DL3units="+str(parameters["DL3units"])
            params += ";epochs="+str(parameters["epochs"])
            params += ";batch_size="+str(parameters["batch_size"])
            params += ";training_ratio="+str(parameters["training_ratio"])
            params += ";sequance_length="+str(parameters["sequance_length"])
            params += ";no_of_prediction_points="+str(parameters["no_of_prediction_points"])
            
        elif(m=="multi_cnn"): # CNN model

            prediction = model.get_output()

            params = "CL1filters=" + str(parameters["CL1filters"])
            params += ";CL1kernal_size=" + str(parameters["CL1kernal_size"])
            params += ";CL1strides=" + str(parameters["CL1strides"])
            params += ";PL1pool_size=" + str(parameters["PL1pool_size"])
            params += ";DL1units=" + str(parameters["DL1units"])
            params += ";DL2units=" + str(parameters["DL2units"])
            params += ";DL3units=" + str(parameters["DL3units"])
            params += ";epochs=" + str(parameters["epochs"])
            params += ";batch_size=" + str(parameters["batch_size"])
            params += ";training_ratio=" + str(parameters["training_ratio"])
            params += ";sequance_length=" + str(parameters["sequance_length"])
            params += ";no_of_prediction_points="+str(parameters["no_of_prediction_points"])
            
        elif(m=="multi_lstmcnn"): # LSTMCNN model
            
            prediction = model.get_output()

            params = "lstmWeight="+str(parameters["lstmWeight"])
            params += ";cnnWeight="+str(parameters["cnnWeight"])
            params += ";lstmCells="+str(parameters["lstmCells"])
            params += ";LSTMDL1units="+str(parameters["LSTMDL1units"])
            params += ";LSTML2units="+str(parameters["LSTMDL2units"])
            params += ";LSTMDL3units="+str(parameters["LSTMDL3units"])
            params += ";CL1filters="+str(parameters["CL1filters"])
            params += ";CL1kernal_size="+str(parameters["CL1kernal_size"])
            params += ";CL1strides="+str(parameters["CL1strides"])
            params += ";PL1pool_size="+str(parameters["PL1pool_size"])
            params += ";CNNDL1units="+str(parameters["CNNDL1units"])
            params += ";CNNDL2units="+str(parameters["CNNDL2units"])
            params += ";CNNDL3units="+str(parameters["CNNDL3units"])
            params += ";epochs="+str(parameters["epochs"])
            params += ";batch_size="+str(parameters["batch_size"])
            params += ";training_ratio="+str(parameters["training_ratio"])
            params += ";sequance_length="+str(parameters["sequance_length"])
            params += ";no_of_prediction_points="+str(parameters["no_of_prediction_points"])

        elif(m=="multi_lstmcnn_kerascombinantion"): # LSTMCNN-keras model
            
            prediction = model.get_output()

            params = "lstmWeight="+str(parameters["lstmWeight"])
            params += ";cnnWeight="+str(parameters["cnnWeight"])
            params += ";lstmCells="+str(parameters["lstmCells"])
            params += ";LSTMDL1units="+str(parameters["LSTMDL1units"])
            params += ";LSTML2units="+str(parameters["LSTMDL2units"])
            params += ";LSTMDL3units="+str(parameters["LSTMDL3units"])
            params += ";CL1filters="+str(parameters["CL1filters"])
            params += ";CL1kernal_size="+str(parameters["CL1kernal_size"])
            params += ";CL1strides="+str(parameters["CL1strides"])
            params += ";PL1pool_size="+str(parameters["PL1pool_size"])
            params += ";CNNDL1units="+str(parameters["CNNDL1units"])
            params += ";CNNDL2units="+str(parameters["CNNDL2units"])
            params += ";CNNDL3units="+str(parameters["CNNDL3units"])
            params += ";epochs="+str(parameters["epochs"])
            params += ";batch_size="+str(parameters["batch_size"])
            params += ";training_ratio="+str(parameters["training_ratio"])
            params += ";sequance_length="+str(parameters["sequance_length"])
            params += ";no_of_prediction_points="+str(parameters["no_of_prediction_points"])

        else:
            print("Invalid Model!")

        # test model
        testing_start = int(parameters["training_ratio"]*len(value))
        training = np.ones(testing_start)
        testing = np.zeros(len(value)-testing_start)
        training_colomn = np.append(training,testing)
        
        # get anomaly label
        try :
            label = np.array(dataframe_expect['label'])
        except KeyError:
            print "Warnning :["+f+"] No `label` colomn found data set without labels !. Assumes there are no anomolies"
            label = np.zeros(len(value))


        testing_prediction = prediction[:]
        no_of_prediction_points = len(testing_prediction)
        
        data = { 'value':value, 'prediction_training':training_colomn, 'label':label }
        colomn_arrangement = ['value','prediction_training','label']
        for i in range(no_of_prediction_points):
            prediction[i] = np.append(training, prediction[i])
            data['prediction_'+str(i)] = prediction[i]
            colomn_arrangement.append('prediction_'+str(i))
   
        dataframe_out = pandas.DataFrame(data, index=timestamp)
        dataframe_out.index.name = "timestamp"
        dataframe_out = dataframe_out[colomn_arrangement]
        out_file = helpers.get_result_file_name(f, csv_output_directory,m)
        dataframe_out.to_csv(out_file)

        testing_value = value[testing_start:]
        mse = helpers.MSE_multipoint(testing_value, testing_prediction, no_of_prediction_points)
        output_files.append( helpers.get_result_dump_name(out_file) +","+ str(mse)+","+params )

        helpers.dump_results(output_files, csv_output_directory,m)
        print("##### ["+m+"]"+ str(count) + " CSV input File processed #####")
        count += 1

    print("##### " + m + " done ! #####")
print("All Done !")