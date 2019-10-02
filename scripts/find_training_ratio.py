import json
import pandas
import math
import sys
import os
import numpy as np
import re

input_dir = "../data/nab_tuned"
summary_file = "../data/nab_tuned_summary.csv"

reg_x = re.compile(r'\.(csv)')
csv_input_files = []
for path, dnames, fnames in os.walk(input_dir):
    csv_input_files.extend([os.path.join(path, f) for f in fnames if reg_x.search(f)])

csv_input_files.sort()
summary = open(summary_file, "w")
summary.write("file,no_of_anomalies,first_label,length,first_label_ratio\n")

for f in csv_input_files:
    print("Processing " + f)
    dataframe = pandas.read_csv(f)
    label = np.array(dataframe['label'])

    first_label = 0
    for i in label:
        first_label += 1
        if i == 1:
            break

    no_of_anomalies = 0
    in_anomaly_window = False
    for i in label:
        if i == 1 and not in_anomaly_window:
            no_of_anomalies += 1
            in_anomaly_window = True
        
        if in_anomaly_window:
            if i == 0:
                in_anomaly_window = False


    fname = f.split("/")[-1]
    length = len(label)
    first_label_ratio = round(float(first_label)/float(length),3)

    line = fname + "," + str(no_of_anomalies) + "," + str(first_label) + "," + str(length) + "," + str(first_label_ratio) + "\n"
    summary.write(line)
    print(line)

summary.close()