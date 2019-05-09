import dtw 
import pandas
import numpy as np

class detector:
    def __init__(self, values, predictions):
        self.values = values
        self.predictions = predictions

    def calculate_distances(self, comparision_window_size):
        self.comparision_window_size = comparision_window_size
        self.distances = dtw.get_dtw(self.values, self.predictions, self.comparision_window_size)
        return self.distances

    def set_threshold(self, training_ratio, max_multipler):
        training_end = int(len(self.distances)*training_ratio)
        training_distances = self.distances[0:training_end]
        self.threshold = np.amax(training_distances)*max_multipler
        # self.threshold = 1
        return self.threshold

    def get_anomalies(self):
        self.anomalies = np.zeros(len(self.distances))
        for i in range(0,len(self.distances)):
            if self.distances[i] > self.threshold:
                self.anomalies[i] = 1
                print i
        return self.anomalies


# f ="./../../results/data/arima/ec2_cpu_utilization_24ae8d.csv"
# dataframe = pandas.read_csv(f)
# value = np.array(dataframe['value'])
# prediction = np.array(dataframe['prediction'])

# mydetector = detector(value, prediction, 3)
# print mydetector.calculate_distances()
# print mydetector.set_threshold(training_ratio=0.1,max_multipler=1.1)
# print mydetector.get_anomalies()

















def get_dtw(series1, series2, sequamce_length):
    euclidean_norm = lambda x, y: np.abs(x - y)
    dtw_series = np.zeros(len(series1))
    print dtw_series
    for i in range(sequamce_length, len(series1)):
        sub_series1 = series1[i-sequamce_length:i]
        sub_series2 = series2[i-sequamce_length:i]
        distance, cost_matrix, acc_cost_matrix, path = dtw(sub_series1, sub_series2, dist=euclidean_norm)
        dtw_series[i] = distance
        print(distance)
    
    # # You can also visualise the accumulated cost and the shortest path
    # import matplotlib.pyplot as plt

    # plt.imshow(acc_cost_matrix.T, origin='lower', cmap='gray', interpolation='nearest')
    # plt.plot(path[0], path[1], 'w')
    # plt.show()
    return dtw_series