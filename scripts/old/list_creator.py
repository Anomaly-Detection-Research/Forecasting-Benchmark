import json
import pandas
import math
import sys
import os
import numpy as np
import re
import helpers

# We take input data to created file index so sorted file colomn will be equal to other models
csv_input_directory = "../data"
results_folder = "../results/"
model = "sherlock-framework-lstmcnn"
tesing_start = 362

list_file = "../results/"
model_results_folder = results_folder + "data/" + model + "/"

reg_x = re.compile(r'\.(csv)')
csv_input_files = []
for path, dnames, fnames in os.walk(csv_input_directory):
    csv_input_files.extend([os.path.join(path, f) for f in fnames if reg_x.search(f)])

csv_input_files.sort()

output_files = []
output_files.append("file,mse,parameters")
helpers.dump_results(output_files, results_folder,model)
for f in csv_input_files:
    try :
        # fetching input file data
        dataframe_expect = pandas.read_csv(f)
        value = np.array(dataframe_expect['value'])
        timestamp = np.array(dataframe_expect['timestamp'])
        try :
            label = np.array(dataframe_expect['label'])
        except KeyError:
            print "Warnning :["+f+"] No `label` colomn found data set without labels !. Assumes there are no anomolies"
            label = np.zeros(len(value))

        dataframe_predction = pandas.read_csv(helpers.get_result_file_name(f,results_folder,model))
        prediction = np.array(dataframe_predction['prediction'])
        params = "training_size="+str(tesing_start)

        file_not_found = False
    except IOError:
        file_not_found = True
        print("Warnning :["+helpers.get_result_file_name(f,results_folder,model)+"] File not found !")
    if file_not_found:
        out_file = helpers.get_result_file_name(f, results_folder,model)
        output_files.append( helpers.get_result_dump_name(out_file) +",n/a,n/a" )
        helpers.dump_results(output_files, results_folder,model)
    else :
        
        training = np.ones(tesing_start)
        testing = np.zeros(len(value)-tesing_start)
        training_colomn = np.append(training, testing)

        testing_prediction = prediction[tesing_start:]
        prediction = np.append(training, testing_prediction)

        data = {'prediction':prediction, 'value':value, 'prediction_training':training_colomn, 'label':label } 
        dataframe_out = pandas.DataFrame(data, index=timestamp)
        dataframe_out.index.name = "timestamp"
        dataframe_out = dataframe_out[['value','prediction_training','prediction','label']]
        out_file = helpers.get_result_file_name(f, results_folder,model)
        dataframe_out.to_csv(out_file)

        testing_value = value[tesing_start:]
        mse = helpers.MSE(testing_value,testing_prediction)

        output_files.append( helpers.get_result_dump_name(out_file) +","+ str(mse)+","+params )
        helpers.dump_results(output_files, results_folder,model)