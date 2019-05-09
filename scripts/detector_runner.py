import json
import pandas
import math
import sys
import os
import numpy as np
import re
import helpers
import detector.detector as detector

f ="./../results/data/arima/art_daily_no_noise.csv"
dataframe = pandas.read_csv(f)
value = np.array(dataframe['value'])
prediction = np.array(dataframe['prediction'])

# args
dtw_window_size = 4
csv_output_directory = "../results"
training_ratio = 0.1
threshold_max_multipler = 1.1
models = ["arma","arima","lstm","cnn","lstmcnn","lstmcnn_kerascombinantion"]

for m in models:
    csv_input_directory = csv_output_directory + "/data/" + m
    # get all csv files in input directory
    reg_x = re.compile(r'\.(csv)')
    csv_input_files = []
    for path, dnames, fnames in os.walk(csv_input_directory):
        csv_input_files.extend([os.path.join(path, f) for f in fnames if reg_x.search(f)])
    csv_input_files.sort()

    model_list_file_name = csv_output_directory + "/" + m + "_list.csv"
    model_list_dataframe = pandas.read_csv(model_list_file_name, index_col="file")
    confudion_metics = np.zeros(len(csv_input_files))
    data = {'mse':model_list_dataframe['mse'], 
            'TP':confudion_metics,
            'FP':confudion_metics,
            'FN':confudion_metics,
            'TN':confudion_metics,
            'parameters':model_list_dataframe['parameters']}
    model_list_dataframe = pandas.DataFrame(data)
    for input_file in csv_input_files:
        file_name = input_file.split("/")[-1]
        # print m
        # print model_list_dataframe['mse'][file_name]
        input_dataframe = pandas.read_csv(input_file)
        value = np.array(input_dataframe['value'])
        prediction = np.array(input_dataframe['prediction'])
        detector_instance = detector.detector(values=value, predictions=prediction)
        warp_distance = detector_instance.calculate_distances(comparision_window_size=dtw_window_size)
        threshold = detector_instance.set_threshold(training_ratio=training_ratio, max_multipler=threshold_max_multipler)
        positive_detection = detector_instance.get_anomalies()
        data = {'value':value,
                'prediction':prediction,
                'label':np.array(input_dataframe['label']
                'warp_distance':warp_distance,
                'distance_threshold':threshold,
                'positive_detection':positive_detection}
        out_dataframe = pandas.DataFrame(data, index=np.array(input_dataframe['timestamp']))
        out_dataframe.index.name = "timestamp"
        out_dataframe = dataframe_out[['value','prediction','label','','','']]
        out_file = helpers.get_result_file_name(f, csv_output_directory,m)
        out_dataframe.to_csv(out_file)
        out_dataframe = pandas.DataFrame(data,index=input_dataframe['timestamp'])
        print(out_dataframe)
# dtw_val = helpers.get_dtw(value,prediction,3)
# print(dtw_val[:10])
# print(dtw_val)
# print(len(value))
# print(len(dtw_val))



