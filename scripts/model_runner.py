import json
import pandas
import math
import sys
import os
import numpy as np
import re
import helpers
import models.arma as arma
import models.arima as arima
import models.lstm as lstm
import models.cnn as cnn
import models.lstmcnn as lstmcnn

# args
csv_input_directory = "../data"
csv_output_directory = "../results"
training_ratio = 0.1
sequance_length = 20
epochs = 10
batch_size = 1
models = ["arma","arima","lstm","cnn","lstmcnn"]
# models = ["arma","arima"]
# models = ["lstm"]
# models = ["cnn"]
# models = ["lstmcnn"]

# get all csv files in input directory
reg_x = re.compile(r'\.(csv)')
csv_input_files = []
for path, dnames, fnames in os.walk(csv_input_directory):
    csv_input_files.extend([os.path.join(path, f) for f in fnames if reg_x.search(f)])

csv_input_files.sort()

for m in models:
    output_files = []
    output_files.append("file,mse,parameters")
    nan_output_files = []
    nan_output_files.append("file")
    helpers.dump_results(output_files, csv_output_directory,m)
    helpers.dump_files_with_nan(nan_output_files, csv_output_directory,m)

    print("##### ["+m+"]"+ str(len(csv_input_files)) + " CSV input files to process #####")
    count = 1
    for f in csv_input_files:
        print("Processing ["+m+"]" + f)

        # fetching input file data
        dataframe_expect = pandas.read_csv(f)
        value = np.array(dataframe_expect['value'])
        timestamp = np.array(dataframe_expect['timestamp'])

        # running model
        if(m=="arma"): # ARMA model
            #parms
            ar_max = 1
            ma_max = 1

            arma_model = arma.arma(data=value, training_ratio=training_ratio, ar_max=ar_max, ma_max=ma_max)
            ar,ma = arma_model.train()
            params = "ar="+str(ar)+";ma="+str(ma)+";training_ratio="+str(training_ratio)
            prediction = arma_model.get_output()

        elif(m=="arima"): # ARIMA model
            #params
            ar_max = 1
            d_max = 1
            ma_max = 1

            arima_model = arima.arima(data=value, training_ratio=training_ratio, ar_max=ar_max, d_max=d_max, ma_max=ma_max)
            ar,d,ma = arima_model.train()
            params = "ar="+str(ar)+";d="+str(d)+";ma="+str(ma)+";training_ratio="+str(training_ratio)
            prediction = arima_model.get_output()

        elif(m=="lstm"):
            #params
            lstmCells = 10

            lstm_model = lstm.lstm(data=value,  epochs=epochs, batch_size=batch_size, training_ratio=training_ratio, sequance_length=sequance_length, lstmCells=lstmCells)
            lstm_model.train()
            params = "lstmCells="+str(lstmCells)+";epochs="+str(epochs)+";batch_size="+str(batch_size)+";training_ratio="+str(training_ratio)+";sequance_length="+str(sequance_length)
            prediction = lstm_model.get_output()

        elif(m=="cnn"):
            #params
            CL1filters = 1
            CL1kernal_size = 2
            CL1strides = 1
            PL1pool_size = 1
            DL1units = 20
            DL2units = 5
            DL3units = 1

            cnn_model = cnn.cnn(data=value, epochs=epochs, batch_size=batch_size, training_ratio=training_ratio, sequance_length=sequance_length,CL1filters=CL1filters, CL1kernal_size=CL1kernal_size, CL1strides=CL1strides, PL1pool_size=PL1pool_size, DL1units=DL1units, DL2units=DL2units, DL3units=DL3units)
            cnn_model.train()
            params = "CL1filters="+str(CL1filters)+";CL1kernal_size="+str(CL1kernal_size)+";CL1strides="+str(CL1strides)+";PL1pool_size="+str(PL1pool_size)+";DL1units="+str(DL1units)+";DL2units="+str(DL2units)+";DL3units="+str(DL3units)+";epochs="+str(epochs)+";batch_size="+str(batch_size)+";training_ratio="+str(training_ratio)+";sequance_length="+str(sequance_length)

            prediction = cnn_model.get_output()
        elif(m=="lstmcnn"):
            #params
            lstmWeight = 0.5
            cnnWeight = 0.5
            #lstm params
            lstmCells=10
            #cnn params
            CL1filters = 1
            CL1kernal_size = 2
            CL1strides = 1
            PL1pool_size = 1
            DL1units = 20
            DL2units = 5
            DL3units = 1

            lstmcnn_model = lstmcnn.lstmcnn(data=value, epochs=epochs, batch_size=batch_size, training_ratio=training_ratio, sequance_length=sequance_length, lstmCells=lstmCells, CL1filters=CL1filters, CL1kernal_size=CL1kernal_size, CL1strides=CL1strides, PL1pool_size=PL1pool_size, DL1units=DL1units, DL2units=DL2units, DL3units=DL3units, lstmWeight=lstmWeight, cnnWeight=cnnWeight)
            lstmcnn_model.train()
            params = "lstmWeight="+str(lstmWeight)+";cnnWeight="+str(cnnWeight)+";lstmCells="+str(lstmCells)+";CL1filters="+str(CL1filters)+";CL1kernal_size="+str(CL1kernal_size)+";CL1strides="+str(CL1strides)+";PL1pool_size="+str(PL1pool_size)+";DL1units="+str(DL1units)+";DL2units="+str(DL2units)+";DL3units="+str(DL3units)+";epochs="+str(epochs)+";batch_size="+str(batch_size)+";training_ratio="+str(training_ratio)+";sequance_length="+str(sequance_length)

            prediction = lstmcnn_model.get_output()
        else:
            print("Invalid Model!")

        # checking if prediction contains nan
        if helpers.check_nan(prediction):
            nan_output_files.append(f)
            helpers.dump_files_with_nan(nan_output_files, csv_output_directory,m)
        else :
            tesing_start = int(training_ratio*len(value))
            value = value[tesing_start:]
            timestamp = timestamp[tesing_start:]
            data = {'prediction':prediction, 'value':value } 
            dataframe_out = pandas.DataFrame(data, index=timestamp)
            dataframe_out.index.name = "timestamp"
            dataframe_out = dataframe_out[['value','prediction']]
            out_file = helpers.get_result_file_name(f, csv_output_directory,m)
            dataframe_out.to_csv(out_file)
            mse = helpers.MSE(value,prediction)
            output_files.append( helpers.get_result_dump_name(out_file) +","+ str(mse)+","+params )

            helpers.dump_results(output_files, csv_output_directory,m)
        print("##### ["+m+"]"+ str(count) + " CSV input File processed #####")
        count += 1

    print("##### " + m + " done ! #####")
print("All Done !")