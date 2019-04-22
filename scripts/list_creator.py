import json
import pandas
import math
import sys
import os
import numpy as np
import re
import helpers

csv_input_directory = "../data"
results_folder = "../results/"
model = "sherlock-lstmcnn"
training_ends = 301

list_file = "../results/"
model_results_folder = results_folder + "data/" + model + "/"

csv_output_directory = results_folder

reg_x = re.compile(r'\.(csv)')
csv_input_files = []
for path, dnames, fnames in os.walk(csv_input_directory):
    csv_input_files.extend([os.path.join(path, f) for f in fnames if reg_x.search(f)])

csv_input_files.sort()

output_files = []
output_files.append("file,mse,parameters")
nan_output_files = []
nan_output_files.append("file")
helpers.dump_results(output_files, csv_output_directory,model)
helpers.dump_files_with_nan(nan_output_files, csv_output_directory,model)
for f in csv_input_files:

    try :
        # fetching input file data
        dataframe_expect = pandas.read_csv(f)
        value = np.array(dataframe_expect['value'])
        total_length = len(value)
        # timestamp = np.array(dataframe_expect['timestamp'])

        dataframe_predction = pandas.read_csv(helpers.get_result_file_name(f,results_folder,model))
        value = np.array(dataframe_predction['value'])
        timestamp = np.array(dataframe_predction['timestamp'])
        prediction = np.array(dataframe_predction['prediction'])
        params = "training_size=300"

        # checking if prediction contains nan
        # if helpers.check_nan(prediction):
        #     nan_output_files.append(f)
        #     helpers.dump_files_with_nan(nan_output_files, csv_output_directory,model)
        no_file = False
    except IOError:
        no_file = True
        print "Caught file "
        nan_output_files.append(f)
        helpers.dump_files_with_nan(nan_output_files, csv_output_directory,model)
        pass
    if no_file:
        print 1
    else :
        if(total_length - len(value) > 0):
            removed_length = total_length - len(value)
            training_ends = training_ends - removed_length
            tesing_start = training_ends
        else :
            print f
            print "Error file bigger than it should be"

        tesing_start = training_ends
        value = value[tesing_start:]
        timestamp = timestamp[tesing_start:]
        prediction = prediction[tesing_start:]
        data = {'prediction':prediction, 'value':value } 
        dataframe_out = pandas.DataFrame(data, index=timestamp)
        dataframe_out.index.name = "timestamp"
        dataframe_out = dataframe_out[['value','prediction']]
        out_file = helpers.get_result_file_name(f, csv_output_directory,model)
        dataframe_out.to_csv(out_file)
        mse = helpers.MSE(value,prediction)
        output_files.append( helpers.get_result_dump_name(out_file) +","+ str(mse)+","+params )

        helpers.dump_results(output_files, csv_output_directory,model)