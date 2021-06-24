import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from Day7 import Model


class MyDataset(Dataset):
    def __init__(self, filepath):
        # xy = np.loadtxt(filepath, delimiter=',', dtype=np.float32)
        # self.len = xy.shape[0]
        # self.x_data = torch.from_numpy(xy[:, :-1])
        features= ["Pclass", "Sex", "SibSp", "Parch", "Fare"]
        data = pd.read_csv(filepath)
        self.len = data.shape[0]
        self.x_data = torch.from_numpy(np.array(pd.get_dummies(data[features])))
        self.y_data = torch.from_numpy(np.array(data['Survived']))

    def __getitem__(self, item):
        return self.x_data[item], self.y_data[item]
        pass

    def __len__(self):
        return self.len


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = torch.nn.Linear(6,4)
        self.linear2 = torch.nn.Linear(4, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.sigmoid(self.linear1(x))
        x = self.sigmoid(self.linear2(x))

        return x


dataset2 = MyDataset('datasets/train.csv')
print(len(dataset2.x_data))
total_data = DataLoader(dataset=dataset2, batch_size=1, shuffle=True, num_workers=0)

# print(dataset2.y_data)
model2 = Model()

criterion = torch.nn.BCELoss(size_average=True)
optimizer = torch.optim.SGD(model2.parameters(), lr=0.02)




# if __name__ == '__main__':
#     for epoch in range(100):
#         for i, data in enumerate(total_data, 0):
#             inputs, label = data
#             inputs = inputs.float()
#             label = label.float()
#
#             y_pred = model2(inputs)
#             y_pred = y_pred.squeeze(-1)
#             loss = criterion(y_pred, label)
#             print(epoch, i, loss.item())
#
#             optimizer.zero_grad()
#             loss.backward()
#
#             optimizer.step()



#optimizer

# data = pd.read_csv('train.csv')
# features = ["Pclass", "Sex", "SibSp", "Parch", "Fare"]
# # print(data)
# # print(data.info)
# # print(data.head())
# # print(data[features])
# a = data[features]
# print(a.head())
# data2 = pd.get_dummies(data[features])
#
# b = np.array(data2)
# data3 = torch.from_numpy(b)
# print(b)
# print(data3)


for i, data in enumerate(total_data, 0):
    inputs, labels = data
    inputs = inputs.float()
    labels = labels.float()
    print("*"*50)
    print(inputs)
    print(labels)
    print("*"*50)

    # y_pred = model2(inputs)
    # print(y_pred.shape)
    # y_pred = y_pred.squeeze(-1)

     # print(y_pred.shape)