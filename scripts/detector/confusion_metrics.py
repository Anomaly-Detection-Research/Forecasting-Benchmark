import pandas
import numpy as np

class confusion_metrics:
    def __init__(self, label, positive_detection, prediction_training, threshold_training):
        self.label = label
        self.positive_detection = positive_detection
        self.prediction_training = prediction_training
        self.threshold_training = threshold_training

    def calculate_metrics(self):
        self.TruePostive = 0
        self.TrueNegative = 0
        self.FalsePostive = 0
        self.FalseNegative = 0
        
        in_anomaly_window = False
        this_anomaly_detected = False
        for i in range(0, len(self.label)):
            #skip training data
            # if self.prediction_training[i] == 1 or self.threshold_training[i] == 1:
            #     pass
            
            #if label is true and detection is postive
            if self.label[i] == 1:
                #indicate we are in anomaly window
                in_anomaly_window = True

                #if detection was not postive in this anomaly window before and detection is postive now
                if (not this_anomaly_detected) and (self.positive_detection[i] == 1):
                    self.TruePostive += 1
                    #indicate this anomaly window is detected
                    this_anomaly_detected = True

            # if label is false
            elif self.label[i] == 0:
                # if detection is true 
                if self.positive_detection[i] == 1:
                    self.FalsePostive += 1
                
                # if detection is true
                else:
                    self.TrueNegative += 1
                
                #if prior data point was an anomaly (window)
                if in_anomaly_window:
                    # if prior anomaly window not detected
                    if not this_anomaly_detected:
                        self.FalseNegative += 1
                
                in_anomaly_window = False
                this_anomaly_detected = False
            
            # if in anomaly window at the end of data set
            if i == (len(self.label)-1) and in_anomaly_window:
                # if prior anomaly window not detected
                if not this_anomaly_detected:
                    self.FalseNegative += 1


    def get_TP(self):
        return self.TruePostive
    
    def get_TN(self):
        return self.TrueNegative
    
    def get_FP(self):
        return self.FalsePostive
    
    def get_FN(self):
        return self.FalseNegative