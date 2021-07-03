import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class MyDataset(Dataset):
    def __init__(self, filepath):
        features = ["Pclass", "Sex", "SibSp", "Parch", "Fare"]
        data = pd.read_csv(filepath)
        self.len = data.shape[0]
        self.x_data = torch.from_numpy(np.array(pd.get_dummies(data[features])))
        self.y_data = torch.from_numpy(np.array(data['Survived']))

    def __getitem__(self, item):
        return self.x_data[item], self.y_data[item]

    def __len__(self):
        return self.len




#
# data = pd.read_csv('datasets/train.csv')
# x_data = data.shape[0]
# x_data = data[["Pclass", "Sex", "SibSp", "Parch", "Fare"]]
#
# x_data = pd.get_dummies(x_data)
# x_data = torch.from_numpy(np.array(x_data))
# x_data = np.array(x_data)
# print(x_data[:12])
# print(x_data.shape)
# print(type(x_data))
