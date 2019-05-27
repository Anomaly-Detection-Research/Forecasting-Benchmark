import pandas
import numpy as np 
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalAveragePooling1D, MaxPooling1D, Flatten
from keras.utils import plot_model

class cnn:
    def __init__(self, parameters):
        self.CL1filters = parameters["CL1filters"]
        self.CL1kernal_size = parameters["CL1kernal_size"]
        self.CL1strides = parameters["CL1strides"]
        self.sequance_length = parameters["sequance_length"]
        self.PL1pool_size = parameters["PL1pool_size"]
        self.DL1units = parameters["DL1units"]
        self.DL2units = parameters["DL2units"]
        self.DL3units = parameters["DL3units"]
        self.epochs = parameters["epochs"]
        self.batch_size = parameters["batch_size"]
        self.no_of_prediction_points = parameters["no_of_prediction_points"]
        self.data = parameters["data"]
        training_data_end = int(parameters["training_ratio"]*len(self.data))
        testing_data_start = training_data_end - self.sequance_length
        training_data = self.data[:training_data_end]
        testing_data = self.data[testing_data_start:]

        self.training_feature_set = []
        self.labels = []
        for i in range(self.sequance_length, len(training_data)-self.no_of_prediction_points):
            self.training_feature_set.append(training_data[i-self.sequance_length:i])
            self.labels.append(training_data[i:i+self.no_of_prediction_points])
        self.labels = np.array(self.labels)
        self.training_feature_set = np.array(self.training_feature_set)

        self.training_feature_set = np.reshape(self.training_feature_set, (self.training_feature_set.shape[0], self.training_feature_set.shape[1], 1))
        
        self.testing_feature_set = []
        for i in range(self.sequance_length, len(testing_data)):
            self.testing_feature_set.append(testing_data[i-self.sequance_length:i])
        self.testing_feature_set = np.array(self.testing_feature_set)
        self.testing_feature_set = np.reshape(self.testing_feature_set, (self.testing_feature_set.shape[0], self.testing_feature_set.shape[1], 1))

    def train(self):
        self.model = Sequential()
        self.model.add(Conv1D(filters=self.CL1filters, kernel_size=self.CL1kernal_size,strides=self.CL1strides, input_shape=(self.sequance_length, 1)))
        self.model.add(MaxPooling1D(pool_size=self.PL1pool_size))
        self.model.add(Flatten())
        self.model.add(Dense(units = self.DL1units))
        self.model.add(Dense(units = self.DL2units))
        self.model.add(Dense(units = self.DL3units))

        self.model.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics=["mse"]) 
        self.model.fit(self.training_feature_set, self.labels, batch_size=self.batch_size, epochs=self.epochs)
    
    def get_output(self):
        predictions = self.model.predict(self.testing_feature_set)
        ret_prediction = []
        for i in range(predictions.shape[1]):
            ret_prediction_colomn = np.zeros(predictions.shape[0])
            for j in range(0, predictions.shape[0]):
                ret_prediction_colomn[j] = predictions[j][i]
            ret_prediction.append(ret_prediction_colomn)
        
        return ret_prediction


    def print_model(self,f):
        plot_model(self.model, to_file=f)


# f = "../../../data/realTweets/Twitter_volume_AAPL.csv"
# dataFrame = pandas.read_csv(f)
# values = dataFrame['value']
# parameters = {
#     "data":values,
#     "training_ratio":0.1,
#     "epochs":1, 
#     "batch_size":20, 
#     "training_ratio":0.1, 
#     "sequance_length":10, 
#     "no_of_prediction_points":5,
#     "CL1filters":1, 
#     "CL1kernal_size":2, 
#     "CL1strides":1, 
#     "PL1pool_size":1, 
#     "DL1units":20, 
#     "DL2units":15, 
#     "DL3units":5
# }
# cnn_model = cnn(parameters)
# print cnn_model.train()
# print cnn_model.get_output()