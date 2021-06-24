# import torch
# import numpy as np
# from torch.utils.data import Dataset
# from torch.utils.data import DataLoader
# from Day7 import Model
#
#
# class DiabetesDataset(Dataset):
#     def __init__(self, filepath):
#         xy = np.loadtxt(filepath, delimiter=',', dtype=np.float32)
#         self.x_data = torch.from_numpy(xy[:, :-1])
#         self.y_data = torch.from_numpy(xy[:, [-1]])
#         self.len = xy.shape[0]
#
#     def __getitem__(self, item):
#         return self.x_data[item], self.y_data[item]
#
#     def __len__(self):
#         return self.len
#
#
# dataset = DiabetesDataset('diabetes.csv.gz')
# train_loader = DataLoader(dataset=dataset, batch_size=32, shuffle=True, num_workers=0)
#
# model = Model()
#
#
# criterion = torch.nn.BCELoss(size_average=True)
# optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
#
# if __name__ == '__main__':
#     for epoch in range(100):
#         for i, data in enumerate(train_loader, 0):
#
#             inputs, labels = data
#
#             y_pred = model(inputs)
#             loss = criterion(y_pred, labels)
#             print(epoch, i, loss.item())
#
#             optimizer.zero_grad()
#             loss.backward()
#
#             optimizer.step()
#
#
#
#

import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import torch


class MyDataset(Dataset):
    # ’‘’定义数据集类‘’‘

    def __init__(self, filepath):
        # 从原始数据集中取五个特征
        features = ["Pclass", "Sex", "SibSp", "Parch", "Fare"]
        data = pd.read_csv(filepath)
        self.len = data.shape[0]  # shape(多少行，多少列)

        # data[features]的类型是DataFrame,先进行独热表示，然后转成array,最后转成tensor用于进行矩阵计算。
        self.x_data = torch.from_numpy(np.array(pd.get_dummies(data[features])))
        self.y_data = torch.from_numpy(np.array(data["Survived"]))

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


# 建立数据集
dataset = MyDataset('datasets/train.csv')

# 建立数据集加载器
from torch.utils.data import DataLoader
train_loader = DataLoader(dataset=dataset, batch_size=1, shuffle=True, num_workers=0)


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        # 选取的五个特征经过独热表示后变为6维。
        self.linear1 = torch.nn.Linear(6, 3)
        self.linear2 = torch.nn.Linear(3, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.sigmoid(self.linear1(x))
        x = self.sigmoid(self.linear2(x))
        return x

    # 定义的预测函数。
    def predict(self, x):
        with torch.no_grad():
            x = self.sigmoid(self.linear1(x))
            x = self.sigmoid(self.linear2(x))
            y = []
            for i in x:
                if i > 0.5:
                    y.append(1)
                else:
                    y.append(0)
            return y


model = Model()
criterion = torch.nn.BCELoss(reduction='mean')
optimizer = torch.optim.SGD(model.parameters(), lr=0.005)

if __name__ == '__main__':
    for epoch in range(100):
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            # 这里先转换了一下数据类型。
            inputs = inputs.float()
            labels = labels.float()

            y_pred = model(inputs)
            # 将维度压缩至1维。
            y_pred = y_pred.squeeze(-1)
            loss = criterion(y_pred, labels)
            print(epoch, i, loss.item())

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()

#
# # 读取test文件
# test_data = pd.read_csv("./titanic/test.csv")
# features = ["Pclass", "Sex", "SibSp", "Parch", "Fare"]
# test = torch.from_numpy(np.array(pd.get_dummies(test_data[features])))
#
# # 进行预测
# y = model.predict(test.float())
#
# # 输出预测结果到文件
# output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': y})
# output.to_csv('my_predict.csv', index=False)

