import json
import pandas
import math
import sys
import os
import numpy as np
import re
import detector.confusion_metrics as confusion_metrics


# args
input_directory = "../results/data"
output_directory = "../results"

threshold_max_multipler = 2

models = ["arma","arima","lstm","cnn","lstmcnn","lstmcnn_kerascombinantion"]
# models = ["arma"]

for m in models:
    model_input_directory = input_directory + "/" + m
    # get all csv files in input directory
    reg_x = re.compile(r'\.(csv)')
    csv_input_files = []
    for path, dnames, fnames in os.walk(model_input_directory):
        csv_input_files.extend([os.path.join(path, f) for f in fnames if reg_x.search(f)])
    csv_input_files.sort()
    model_file_name = output_directory + "/" + m + "_list.csv"
    model_dataframe = pandas.read_csv(model_file_name, index_col="file")
    
    mse = np.array(model_dataframe['mse'])
    filler_values = np.zeros(len(mse))

    data = {'mse':model_dataframe['mse'], 
            'TP':filler_values,
            'FP':filler_values,
            'FN':filler_values,
            'TN':filler_values,
            'parameters':model_dataframe['parameters'],
            'threshold_parameters':model_dataframe['threshold_parameters'],
            'no_of_anomalies':model_dataframe['no_of_anomalies'],
            'first_label':model_dataframe['first_label'],
            'length':model_dataframe['length'],
            "first_label_ratio":model_dataframe['first_label_ratio'],}
    model_dataframe = pandas.DataFrame(data)
    model_dataframe.index.name = "file"
    model_dataframe = model_dataframe[['mse',
                                        'TP',
                                        'FP',
                                        'FN',
                                        'TN',
                                        'parameters',
                                        'threshold_parameters',
                                        'no_of_anomalies',
                                        'first_label',
                                        'length',
                                        "first_label_ratio"]]
    
    print("##### ["+m+"] "+ str(len(csv_input_files)) + " CSV input files to process #####")
    count = 1
    for input_file in csv_input_files:
        print("Processing ["+m+"]" + input_file)
        file_name = input_file.split("/")[-1]
        
        input_dataframe = pandas.read_csv(input_file)
        value = np.array(input_dataframe['value'])
        prediction_training = np.array(input_dataframe['prediction_training'])
        prediction = np.array(input_dataframe['prediction'])
        label = np.array(input_dataframe['label'])
        warp_distance = np.array(input_dataframe['warp_distance'])
        threshold_training = np.array(input_dataframe['threshold_training'])
        distance_threshold = np.ones(len(value)) * (-1)
        positive_detection = np.zeros(len(value))

        threshold_value = float(-1)
        for i in range(len(warp_distance)):
                if threshold_training[i] == 1:
                        if threshold_value < float(warp_distance[i]):
                                threshold_value = float(warp_distance[i])
        threshold_value = threshold_value * float(threshold_max_multipler)
        first_prediction_trained = False
        for i in range(len(distance_threshold)):
                if(prediction_training[i] == 0):
                        first_prediction_trained = True
                if(first_prediction_trained):
                        if threshold_training[i] == 0:
                                distance_threshold[i] = threshold_value
                                if warp_distance[i] >= threshold_value:
                                        positive_detection[i] = 1 

        data = {'value':value,
                'prediction_training':prediction_training,
                'prediction':prediction,
                'label':label,
                'warp_distance':warp_distance,
                'threshold_training':threshold_training,
                'distance_threshold':distance_threshold,
                'positive_detection':positive_detection}
        out_dataframe = pandas.DataFrame(data, index=np.array(input_dataframe['timestamp']))
        out_dataframe.index.name = "timestamp"
        out_dataframe = out_dataframe[['value',
                                'prediction_training',
                                'prediction',
                                'label',
                                'warp_distance',
                                'threshold_training',
                                'distance_threshold',
                                'positive_detection']]
        out_dataframe.to_csv(input_file)


        # Calculating confusion metrics
        metrics = confusion_metrics.confusion_metrics(label=label, positive_detection=positive_detection, prediction_training=prediction_training, threshold_training=threshold_training)
        metrics.calculate_metrics()

        model_dataframe.at[file_name, 'TP'] = metrics.get_TP()
        model_dataframe.at[file_name, 'TN'] = metrics.get_TN()
        model_dataframe.at[file_name, 'FP'] = metrics.get_FP()
        model_dataframe.at[file_name, 'FN'] = metrics.get_FN()
        # Setting new threshold_max_multipler in threshold_parameters
        threshold_parameter_list = model_dataframe.at[file_name, 'threshold_parameters'].split(";")
        threshold_parameters = ""
        for s in threshold_parameter_list:
                setting_max_multiplier = False
                if "threshold_max_multipler" in s:
                        if "threshold_max_multipler=" == s[:24]:
                                threshold_parameters += "threshold_max_multipler="+str(threshold_max_multipler)+";"
                                setting_max_multiplier = True
                if not setting_max_multiplier and not s == "":
                        threshold_parameters += s+";"
        
        model_dataframe.at[file_name, 'threshold_parameters'] = threshold_parameters

        print("##### ["+m+"] "+ str(count) + " CSV input File processed #####")
        count += 1

    model_dataframe.to_csv(model_file_name)
    print("##### " + m + " done ! #####")
print("All Done !")



