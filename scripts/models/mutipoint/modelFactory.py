from cnn import cnn
from lstm import lstm

class modelFactory:
    def __init__(self):
        pass
    
    def get_model(self,model_type, parameters):
        if model_type == "multi_arma":
            return arma(parameters=parameters)

        elif model_type == "multi_arima":
            return arima(parameters=parameters)

        elif model_type == "multi_lstm":
            return lstm(parameters=parameters)

        elif model_type == "multi_cnn":
            return cnn(parameters=parameters)

        elif model_type == "multi_lstmcnn":
            return lstmcnn(parameters=parameters)

        elif model_type == "multi_lstmcnn_kerascombinantion":
            return lstmcnn_kerascombinantion(parameters=parameters)