import pandas
import numpy as np 
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input, Add
from keras.layers import Embedding
from keras.layers import LSTM, Conv1D, GlobalAveragePooling1D, MaxPooling1D, Flatten
from keras.utils import plot_model


class lstmcnn_kerascombinantion:
    def __init__(self, parameters):
        self.lstmCells = parameters["lstmCells"]
        self.LSTMDL1units = parameters["LSTMDL1units"]
        self.LSTMDL2units = parameters["LSTMDL2units"]
        self.LSTMDL3units = parameters["LSTMDL3units"]


        self.CL1filters = parameters["CL1filters"]
        self.CL1kernal_size = parameters["CL1kernal_size"]
        self.CL1strides = parameters["CL1strides"]
        self.PL1pool_size = parameters["PL1pool_size"]
        self.CNNDL1units = parameters["CNNDL1units"]
        self.CNNDL2units = parameters["CNNDL2units"]
        self.CNNDL3units = parameters["CNNDL3units"]
        
        self.training_ratio = parameters["training_ratio"]
        self.sequance_length = parameters["sequance_length"]
        self.epochs = parameters["epochs"]
        self.batch_size = parameters["batch_size"]
        self.no_of_prediction_points = parameters["no_of_prediction_points"]
        self.data = parameters["data"]
        training_data_end = int(len(self.data)*self.training_ratio)
        testing_data_start = training_data_end - self.sequance_length
        training_data = self.data[:training_data_end]
        testing_data = self.data[testing_data_start:]

        self.training_feature_set = []
        self.labels = []
        for i in range(self.sequance_length,len(training_data)):
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

        input_shape = Input(shape=(self.sequance_length, 1))
        
        #lstm
        lstm = LSTM(units=self.lstmCells)(input_shape)
        lstmdense1 = Dense(units = self.LSTMDL1units)(lstm)
        lstmdense2 = Dense(units = self.LSTMDL2units)(lstmdense1)
        lstmdense3 = Dense(units = self.LSTMDL3units)(lstmdense2)
        
        #cnn
        cnn = Conv1D(filters=self.CL1filters, kernel_size=self.CL1kernal_size,strides=self.CL1strides, input_shape=(self.sequance_length, 1))(input_shape)
        cnnpooling = MaxPooling1D(pool_size=self.PL1pool_size)(cnn)
        cnnflaten = Flatten()(cnnpooling)
        cnndense1 = Dense(units = self.CNNDL1units)(cnnflaten)
        cnndense2 = Dense(units = self.CNNDL2units)(cnndense1)
        cnndense3 = Dense(units = self.CNNDL3units)(cnndense2)
        print("##########")
        print(self.CNNDL3units)
        print(self.LSTMDL3units)

        #combinantion layer
        out = Add()([lstmdense3, cnndense3])

        self.model = Model(input_shape, out)

        self.model.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics=["mse"])  
        self.model.fit(self.training_feature_set, self.labels, epochs = self.epochs, batch_size = self.batch_size)  

        


    def get_output(self):
        # predictions = self.model.predict(self.testing_feature_set)
        # ret_prediction = np.zeros(predictions.shape[0])
        # for i in range(0, predictions.shape[0]):
        #     ret_prediction[i] = predictions[i][0]

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