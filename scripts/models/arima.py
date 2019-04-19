import model_helpers
import numpy as np
from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import warnings

class arima:
    def __init__(self, data, training_ratio, ar_max=3, d_max=2, ma_max=3):
        self.ar_max = ar_max # maximum AR parameter
        self.d_max = d_max # maximum Differance parameter
        self.ma_max = ma_max # maximum MA parameter
        training_data_end = int(len(data)*training_ratio)
        testing_data_start = training_data_end
        self.training_data = data[:training_data_end].astype('float64')
        self.testing_data = data[testing_data_start:].astype('float64')

        # self.y = data.astype('float64') # real values from time series
        # self.y_training = data[:int(len(data)*training_ratio)].astype('float64')
        # self.ar = None # best AR parameter set after training
        # self.d = None # best Differance parameter set after training
        # self.ma = None # best MA parameter set after training

    def train(self):
        ar_max = self.ar_max
        d_max = self.d_max
        ma_max = self.ma_max
        params = np.zeros((ar_max+1, d_max+1, ma_max+1))
        y = self.training_data

        for ar in range(0,ar_max+1):
            for d in range(0,d_max+1):
                for ma in range(0,ma_max+1):
                    try:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            arima_model = sm.tsa.ARIMA(y, order=(ar,d,ma))
                            arima_result = arima_model.fit(trend='c', disp=-1)
                        y_prediction = arima_result.predict()
                        mse = model_helpers.MSE(y,y_prediction)
                    except ValueError:
                        mse = float("inf") 
                    except np.linalg.LinAlgError:
                        mes = float("inf") 
                    params[ar][d][ma] = mse
                    print("("+str(ar)+","+str(d)+","+str(ma)+") : " +str(mse))
        
        result = np.where(params == np.amin(params))
        minAR = result[0][0]
        minD = result[1][0]
        minMA = result[2][0]
        print("Best model : ("+str(minAR)+","+str(minD)+","+str(minMA)+") | with MSE = "+str(params[minAR][minD][minMA]))
        self.ar = minAR
        self.d = minD
        self.ma = minMA
        return minAR,minD,minMA
    
    def get_output(self):
        ar = self.ar
        d = self.d
        ma = self.ma
        y = self.testing_data
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            arima_model = sm.tsa.ARIMA(y, order=(ar,d,ma))
            arima_result = arima_model.fit(trend='c', disp=-1)
        y_prediction = arima_result.predict()
        return np.array(y_prediction)