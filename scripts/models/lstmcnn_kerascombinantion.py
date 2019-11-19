import pandas
import numpy as np 
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input, Add, Layer
from keras.layers import Embedding
from keras.layers import LSTM, Conv1D, GlobalAveragePooling1D, MaxPooling1D, Flatten
from keras.utils import plot_model
from keras import optimizers

# Define custom layer for weighted sum
class WeightedSum(Layer):
    def __init__(self, weight1, weight2, **kwargs):
        self.weight1 = weight1
        self.weight2 = weight2
        super(WeightedSum, self).__init__(**kwargs)
    def call(self, model_outputs):
        return self.weight1 * model_outputs[0] + (self.weight2) * model_outputs[1]
    def compute_output_shape(self, input_shape):
        return input_shape[0]

class lstmcnn_kerascombinantion:
    def __init__(self, data,  epochs, batch_size, training_ratio,sequance_length, lstmCells=10, LSTMDL1units=20, LSTMDL2units=5, LSTMDL3units=1, CL1filters=1, CL1kernal_size=2, CL1strides=1, PL1pool_size=1, CNNDL1units=20, CNNDL2units=5, CNNDL3units=1,lstmWeight=0.5, cnnWeight=0.5, learningRate=0.001):
        self.lstmCells = lstmCells
        self.LSTMDL1units = LSTMDL1units
        self.LSTMDL2units = LSTMDL2units
        self.LSTMDL3units = LSTMDL3units

        self.CL1filters = CL1filters
        self.CL1kernal_size = CL1kernal_size
        self.CL1strides = CL1strides
        self.PL1pool_size = PL1pool_size
        self.CNNDL1units = CNNDL1units
        self.CNNDL2units = CNNDL2units
        self.CNNDL3units = CNNDL3units

        self.lstmWeight = lstmWeight
        self.cnnWeight = cnnWeight

        self.learningRate = learningRate

        self.sequance_length = sequance_length
        self.epochs = epochs
        self.batch_size = batch_size
        training_data_end = int(len(data)*training_ratio)
        testing_data_start = training_data_end - sequance_length
        training_data = data[:training_data_end]
        testing_data = data[testing_data_start:]
        self.training_feature_set = []
        self.labels = []
        for i in range(self.sequance_length,len(training_data)):
            self.training_feature_set.append(training_data[i-self.sequance_length:i])
            self.labels.append(training_data[i])
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

        #combinantion layer
        # out = Add()([lstmdense3, cnndense3])
        out = WeightedSum(self.lstmWeight, self.cnnWeight)([lstmdense3, cnndense3])
            

        self.model = Model(input_shape, out)

        adam = optimizers.Adam(lr=self.learningRate)

        self.model.compile(optimizer = adam, loss = 'mean_squared_error', metrics=["mse"])
        self.model.fit(self.training_feature_set, self.labels, epochs = self.epochs, batch_size = self.batch_size)  

        


    def get_output(self):
        predictions = self.model.predict(self.testing_feature_set)
        ret_prediction = np.zeros(predictions.shape[0])
        for i in range(0, predictions.shape[0]):
            ret_prediction[i] = predictions[i][0]
        return ret_prediction
    
    def print_model(self,f):
        plot_model(self.model, to_file=f)