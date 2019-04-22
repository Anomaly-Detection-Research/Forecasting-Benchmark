import models.lstm as lstm
import models.cnn as cnn
import pandas
import numpy as np


f = "../data/artificialNoAnomaly/art_daily_no_noise.csv"
dataframe = pandas.read_csv(f)
value = np.array(dataframe['value'])

training_ratio = 0.1
sequance_length = 20
epochs = 1
batch_size = 100


lstmCells = 10

lstm_model = lstm.lstm(data=value,  epochs=epochs, batch_size=batch_size, training_ratio=training_ratio, sequance_length=sequance_length, lstmCells=lstmCells)
lstm_model.train()
lstm_model.print_model("../results/lstm_model.png")


CL1filters = 1
CL1kernal_size = 2
CL1strides = 1
PL1pool_size = 1
DL1units = 20
DL2units = 5
DL3units = 1

cnn_model = cnn.cnn(data=value, epochs=epochs, batch_size=batch_size, training_ratio=training_ratio, sequance_length=sequance_length,CL1filters=CL1filters, CL1kernal_size=CL1kernal_size, CL1strides=CL1strides, PL1pool_size=PL1pool_size, DL1units=DL1units, DL2units=DL2units, DL3units=DL3units)
cnn_model.train()
cnn_model.print_model("../results/cnn_model.png")