import pandas
import numpy as np 
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Embedding
from keras.layers import LSTM

class lstm:
    def __init__(self,lstmCells, lag, epochs, batch_size, training_ratio, data):
        self.lstmCells = lstmCells
        self.lag = lag
        self.epochs = epochs
        self.batch_size = batch_size
        self.data = data
        
        
        training_data = data[:int(len(data)*training_ratio)]
        self.training_feature_set = []
        self.labels = []
        for i in range(self.lag,len(training_data)):
            self.training_feature_set.append(training_data[i-self.lag:i])
            self.labels.append(training_data[i])
        self.labels = np.array(self.labels)
        self.training_feature_set = np.array(self.training_feature_set)
        self.training_feature_set = np.reshape(self.training_feature_set, (self.training_feature_set.shape[0], self.training_feature_set.shape[1], 1))

        self.feature_set = []
        for i in range(self.lag, len(data)):  
            self.feature_set.append(data[i-self.lag:i])
        self.feature_set = np.array(self.feature_set)
        self.feature_set = np.reshape(self.feature_set, (self.feature_set.shape[0], self.feature_set.shape[1], 1))
    
    def train(self):
        self.model = Sequential()
        self.model.add(LSTM(units=self.lstmCells))  
        self.model.add(Dense(units = 1))

        self.model.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics=["mse"])  
        self.model.fit(self.training_feature_set, self.labels, epochs = self.epochs, batch_size = self.batch_size)  


    def get_output(self):
        predictions = self.model.predict(self.feature_set)
        predictions = np.concatenate((self.data[:self.lag], predictions),axis=None)
        return np.array(predictions) 