import model_helpers
import numpy as np
from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import warnings

class arma:
    def __init__(self,ar_max,ma_max,y):
        self.ar_max = ar_max # maximum AR parameter
        self.ma_max = ma_max # maximum MA parameter
        self.y = y.astype('float64') # real values from time series
        self.ma = None # best MA parameter set after training
        self.ar = None # best AR parameter set after training

    def train(self):
        ar_max = self.ar_max
        ma_max = self.ma_max
        params = np.zeros((ar_max+1, ma_max+1))
        y = self.y

        for ar in range(0,ar_max+1):
            for ma in range(0,ma_max+1):
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        arma_model = sm.tsa.ARMA(y, order=(ar,ma))
                        arma_result = arma_model.fit(trend='c', disp=-1)
                    y_prediction = arma_result.predict()
                    mse = model_helpers.MSE(y,y_prediction)
                except ValueError:
                    mse = float("inf") 
                except np.linalg.LinAlgError:
                    mes = float("inf") 
                params[ar][ma] = mse
                print("("+str(ar)+","+str(ma)+") : " +str(mse))
        
        result = np.where(params == np.amin(params))
        minAR = result[0][0]
        minMA = result[1][0]
        print("Best model : ("+str(minAR)+","+str(minMA)+") | with MSE = "+str(params[minAR][minMA]))
        self.ar = minAR
        self.ma = minMA
        return minAR,minMA
    
    def get_output(self):
        ar = self.ar
        ma = self.ma
        y = self.y
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            arma_model = sm.tsa.ARMA(y, order=(ar,ma))
            arma_result = arma_model.fit(trend='c', disp=-1)
        y_prediction = arma_result.predict()
        return np.array(y_prediction)