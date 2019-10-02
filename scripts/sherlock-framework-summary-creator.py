import json
import pandas
import math
import sys
import os
import numpy as np
import re
import helpers
import detector.confusion_metrics as confusion_metrics

# We take input data to created file index so sorted file colomn will be equal to other models
results_folder = "../results/"
model = "sherlock-framework-lstmcnn"

# summary_file = "../results/" + model + "_list.csv"
model_results_folder = results_folder + "data/" + model + "/"

reg_x = re.compile(r'\.(csv)')
csv_input_files = []
for path, dnames, fnames in os.walk(model_results_folder):
    csv_input_files.extend([os.path.join(path, f) for f in fnames if reg_x.search(f)])

csv_input_files.sort()

output_files = []
output_files.append("file,mse,TP,FP,FN,TN,parameters,threshold_parameters")

helpers.dump_results(output_files, results_folder, model)
for f in csv_input_files:
    # fetching input file data
    print "Processing " + f
    input_file_dataframe = pandas.read_csv(f)
    
    value = np.array(input_file_dataframe['value'])
    prediction_training = np.array(input_file_dataframe['prediction_training'])
    prediction = np.array(input_file_dataframe['prediction'])
    
    label = np.array(input_file_dataframe['label'])
    threshold_training = np.array(input_file_dataframe['threshold_training'])
    positive_detection = np.array(input_file_dataframe['positive_detection'])

    prediction_training_ends = 0

    for i in range(0,len(prediction_training)):
        prediction_training_ends += int(prediction_training[i])


    testing_value = value[prediction_training_ends:]
    testing_prediction = prediction[prediction_training_ends:]
    mse = helpers.MSE(testing_value,testing_prediction)

    # test_label = label[threshold_training_ends:]
    # test_positive_detection = positive_detection[threshold_training_ends:]

    # Calculating confusion metrics
    metrics = confusion_metrics.confusion_metrics(label=label, positive_detection=positive_detection, prediction_training=prediction_training, threshold_training=threshold_training)
    metrics.calculate_metrics()

    TP = metrics.get_TP()
    TN = metrics.get_TN()
    FP = metrics.get_FP()
    FN = metrics.get_FN()

    output_files.append(helpers.get_result_dump_name(f) +","+ 
    str(mse)+","+
    str(TP)+","+
    str(FP)+","+
    str(FN)+","+
    str(TN)+","+
    "n/a"+","+
    "n/a")
    helpers.dump_results(output_files, results_folder,model)
