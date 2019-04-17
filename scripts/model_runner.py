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

# args
csv_input_directory = "../data"
csv_output_directory = "../results"
# models = ["arma"]
# models = ["arma","arima"]
models = ["lstm"]

# get all csv files in input directory
reg_x = re.compile(r'\.(csv)')
csv_input_files = []
for path, dnames, fnames in os.walk(csv_input_directory):
    csv_input_files.extend([os.path.join(path, f) for f in fnames if reg_x.search(f)])

csv_input_files.sort()

for m in models:
    output_files = []
    output_files.append("file,mse")
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
            arma_model = arma.arma(1,1,value)
            ar,ma = arma_model.train()
            params = "ar="+str(ar)+";ma="+str(ma)
            prediction = arma_model.get_output()
        elif(m=="arima"): # ARIMA model
            arima_model = arima.arima(1,1,1,value)
            ar,d,ma = arima_model.train()
            params = "ar="+str(ar)+";d="+str(d)+";ma="+str(ma)
            prediction = arima_model.get_output()
        elif(m=="lstm"):
            lstm_model = lstm.lstm(lstmCells=10, lag=10, epochs=1, batch_size=1, training_ratio=0.1, data=value)
            lstm_model.train()
            params = "lstmCells="+str(10)
            prediction = lstm_model.get_output()
        elif(m=="cnn"):
            print("not impimented")
        elif(m=="lstmcnn"):
            print("not impimented")
        else:
            print("Invalid Model!")

        # checking if prediction contains nan
        if helpers.check_nan(prediction):
            nan_output_files.append(f)
            helpers.dump_files_with_nan(nan_output_files, csv_output_directory,m)
        else :
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