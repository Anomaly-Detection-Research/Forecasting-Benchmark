import json
import pandas
import math
import sys
import os
import numpy as np
import re
import helpers
import detector.detector as detector
import detector.confusion_metrics as confusion_metrics

# f ="./../results/data/arima/art_daily_no_noise.csv"
# dataframe = pandas.read_csv(f)
# value = np.array(dataframe['value'])
# prediction = np.array(dataframe['prediction'])

# args
csv_output_directory = "../results"
# models = ["arma","arima","lstm","cnn","lstmcnn", "sherlock-lstmcnn","lstmcnn_kerascombinantion"]
# models = ["arma"]
# models = ["sherlock-lstmcnn"]
models = ["sherlock-framework-lstmcnn"]

for m in models:
    csv_input_directory = csv_output_directory + "/data/" + m
    # get all csv files in input directory
    reg_x = re.compile(r'\.(csv)')
    csv_input_files = []
    for path, dnames, fnames in os.walk(csv_input_directory):
        csv_input_files.extend([os.path.join(path, f) for f in fnames if reg_x.search(f)])
    csv_input_files.sort()
    model_file_name = csv_output_directory + "/" + m + "_list.csv"
    model_dataframe = pandas.read_csv(model_file_name, index_col="file")
    mse = np.array(model_dataframe['mse'])
    confudion_metics = np.zeros(len(mse))
    threshold_parameters = []
    for i in range(0, len(mse)):
            if mse[i] == 'n/a':
                threshold_parameters.append('n/a')
            else:
                threshold_parameters.append("comparision_window_size="+str(dtw_window_size)+";threshold_max_multipler="+str(threshold_max_multipler)+";training_ratio="+str(training_ratio))
    data = {'mse':model_dataframe['mse'], 
            'TP':confudion_metics,
            'FP':confudion_metics,
            'FN':confudion_metics,
            'TN':confudion_metics,
            'parameters':model_dataframe['parameters'],
            'threshold_parameters':threshold_parameters}
    model_dataframe = pandas.DataFrame(data)
    model_dataframe.index.name = "file"
    model_dataframe = model_dataframe[['mse',
                                        'TP',
                                        'FP',
                                        'FN',
                                        'TN',
                                        'parameters',
                                        'threshold_parameters']]
    
    print("##### ["+m+"]"+ str(len(csv_input_files)) + " CSV input files to process #####")
    count = 1
    for input_file in csv_input_files:
        print("Processing ["+m+"]" + input_file)
        file_name = input_file.split("/")[-1]

        input_dataframe = pandas.read_csv(input_file)
        value = np.array(input_dataframe['value'])
        prediction = np.array(input_dataframe['prediction'])
        prediction = np.array(input_dataframe['prediction'])
        label = np.array(input_dataframe['label'])
        prediction_training = np.array(input_dataframe['prediction_training'])

        prediction_training_stops = 0
        for i in range(0, len(prediction_training)):
                if prediction_training[i] == 1:
                        prediction_training_stops += 1

        testing_value = value[prediction_training_stops:]
        testing_prediction = prediction[prediction_training_stops:]

        detector_instance = detector.detector(values=testing_value, predictions=testing_prediction)
        warp_distance = detector_instance.calculate_distances(comparision_window_size=dtw_window_size)
        threshold = detector_instance.set_threshold(training_ratio=training_ratio, max_multipler=threshold_max_multipler)
        positive_detection = detector_instance.get_anomalies()

        threshold_training_starts = prediction_training_stops
        threshold_training_size = int(len(testing_value)*training_ratio)
        
        threshold_ignore = np.zeros(threshold_training_starts)
        threshold_training = np.ones(threshold_training_size)
        threshold_testing = np.zeros(len(value)-threshold_training_starts - threshold_training_size)
        threshold_training_colomn = np.append(threshold_ignore, threshold_training)
        threshold_training_colomn = np.append(threshold_training_colomn, threshold_testing)


        threshold = np.ones(len(warp_distance))*threshold
        threshold = np.append(threshold_ignore, threshold)
        warp_distance = np.append(threshold_ignore, warp_distance)
        positive_detection = np.append(threshold_ignore, positive_detection)

        data = {'value':value,
                'prediction':prediction,
                'prediction_training':prediction_training,
                'label':label,
                'warp_distance':warp_distance,
                'threshold_training':threshold_training_colomn,
                'distance_threshold':threshold,
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
        metrics = confusion_metrics.confusion_metrics(label=label, positive_detection=positive_detection, prediction_training=prediction_training, threshold_training=threshold_training_colomn)
        metrics.calculate_metrics()

        model_dataframe.at[file_name, 'TP'] = metrics.get_TP()
        model_dataframe.at[file_name, 'TN'] = metrics.get_TN()
        model_dataframe.at[file_name, 'FP'] = metrics.get_FP()
        model_dataframe.at[file_name, 'FN'] = metrics.get_FN()
        
        

        print("##### ["+m+"]"+ str(count) + " CSV input File processed #####")
        count += 1

    model_dataframe.to_csv(model_file_name)
    print("##### " + m + " done ! #####")
print("All Done !")



