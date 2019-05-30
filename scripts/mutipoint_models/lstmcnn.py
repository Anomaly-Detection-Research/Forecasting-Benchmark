import numpy as np 
from lstm import lstm
from cnn import cnn

class lstmcnn:
    def __init__(self, parameters):
        lstm_parmeters = {
            # for all
            "data":parameters["data"],
            "training_ratio":parameters["training_ratio"],
            "no_of_prediction_points":parameters["no_of_prediction_points"],

            # for LSTM 
            "epochs":parameters["epochs"], 
            "batch_size":parameters["batch_size"],
            "sequance_length":parameters["sequance_length"],
            "DL1units":parameters["LSTMDL1units"], 
            "DL2units":parameters["LSTMDL2units"], 
            "DL3units":parameters["LSTMDL3units"],
            "lstmCells":parameters["lstmCells"],
        }
        self.lstm_model = lstm(lstm_parmeters)
        
        cnn_parmeters = {
            # for all
            "data":parameters["data"],
            "training_ratio":parameters["training_ratio"],
            "no_of_prediction_points":parameters["no_of_prediction_points"],

            # for LSTM 
            "epochs":parameters["epochs"], 
            "batch_size":parameters["batch_size"],
            "sequance_length":parameters["sequance_length"],
            "DL1units":parameters["CNNDL1units"], 
            "DL2units":parameters["CNNDL2units"], 
            "DL3units":parameters["CNNDL3units"],

            "CL1filters":parameters["CL1filters"], 
            "CL1kernal_size":parameters["CL1kernal_size"], 
            "CL1strides":parameters["CL1strides"], 
            "PL1pool_size":parameters["PL1pool_size"], 
        }
        self.cnn_model = cnn(cnn_parmeters)

        self.lstmWeight = parameters["lstmWeight"]
        self.cnnWeight = parameters["cnnWeight"]
        self.no_of_prediction_points = parameters["no_of_prediction_points"]
    
    def train(self):
        self.lstm_model.train()
        self.cnn_model.train()
    
    def get_output(self):
        lstm_predictions = self.lstm_model.get_output()
        cnn_predcitions = self.cnn_model.get_output()
        final_predictions = []
        for prediction_colomn_index in range(self.no_of_prediction_points):
            final_predictions.append(
                lstm_predictions[prediction_colomn_index]*self.lstmWeight + cnn_predcitions[prediction_colomn_index]*self.cnnWeight)
        # weighted_lstm_prediction = self.lstm_model.get_output()*self.lstmWeight
        # weighted_cnn_prediction = self.cnn_model.get_output()*self.cnnWeight

        # return weighted_lstm_prediction + weighted_cnn_prediction
        return final_predictions
