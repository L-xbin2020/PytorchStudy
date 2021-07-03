import torch
import pandas as pd

rain_data = pd.read_csv('datasets/dataset_rain.csv')

feature = ['MinPressure','MaxPressure', 'MinVapour', 'MaxVapour', 'MinRelativeHumidity', 'MaxRelativeHumidity', 'MinTemperature','MaxTemperature']

rain_list = rain_data[feature]
print(rain_list)