import numpy as np 
import lstm
import cnn

class lstmcnn:
    def __init__(self, data, epochs, batch_size, training_ratio, sequance_length, lstmCells=10, LSTMDL1units=20, LSTMDL2units=5, LSTMDL3units=1, CL1filters=1, CL1kernal_size=2, CL1strides=1, PL1pool_size=1, CNNDL1units=20, CNNDL2units=5, CNNDL3units=1, lstmWeight=0.5, cnnWeight=0.5, learningRate=0.001):
        
        self.lstm_model = lstm.lstm(data=data,  epochs=epochs, batch_size=batch_size, training_ratio=training_ratio, sequance_length=sequance_length, lstmCells=lstmCells, learningRate=learningRate)
        
        self.cnn_model = cnn.cnn(data=data, epochs=epochs, batch_size=batch_size, training_ratio=training_ratio, sequance_length=sequance_length,CL1filters=CL1filters, CL1kernal_size=CL1kernal_size, CL1strides=CL1strides, PL1pool_size=PL1pool_size, DL1units=CNNDL1units, DL2units=CNNDL2units, DL3units=CNNDL3units, learningRate=learningRate)

        self.lstmWeight = lstmWeight
        self.cnnWeight = cnnWeight
    
    def train(self):
        self.lstm_model.train()
        self.cnn_model.train()
    
    def get_output(self):
        weighted_lstm_prediction = self.lstm_model.get_output()*self.lstmWeight
        weighted_cnn_prediction = self.cnn_model.get_output()*self.cnnWeight

        return weighted_lstm_prediction + weighted_cnn_prediction
