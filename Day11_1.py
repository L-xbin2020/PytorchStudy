from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import torch
import os  # 一定要有这个设定，要不然报错，也生成不了图
import matplotlib.pyplot as plt

import numpy as np

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
batch_size = 64
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST(root='./',
                               train=True,
                               download=True,
                               transform=transform)

train_loader = DataLoader(train_dataset,
                          shuffle=True,
                          batch_size=batch_size)

test_dataset = datasets.MNIST(root='./',
                              train=False,
                              download=True,
                              transform=transform)

test_loader = DataLoader(test_dataset,
                         shuffle=False,
                         batch_size=batch_size)


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.channels = channels
        self.conv1 = nn.Conv2d(channels, channels,
                               kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels,
                               kernel_size=3, padding=1)

    def forward(self, x):
        y = F.relu(self.conv1(x))
        y = self.conv2(y)

        return F.relu(x + y)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5)
        self.mp = nn.MaxPool2d(2)

        self.rblock1 = ResidualBlock(16)
        self.rblock2 = ResidualBlock(32)

        self.fc = nn.Linear(512, 10)

    def forward(self, x):
        in_size = x.size(0)
        x = self.mp(F.relu(self.conv1(x)))
        x = self.rblock1(x)
        x = self.mp(F.relu((self.conv2(x))))
        x = self.rblock2(x)
        x = x.view(in_size, -1)
        x = self.fc(x)

        return x


model = Net()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)


def train(epoch):
    running_loss = 0
    for batch_idx, data in enumerate(train_loader, 0):
        inputs, target = data
        inputs, target = inputs.to(device), target.to(device)

        y_pred = model(inputs)

        optimizer.zero_grad()

        loss = criterion(y_pred, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if batch_idx % 300 == 299:
            print('[%d, %5d], loss: %.3f' % (epoch + 1, batch_idx + 1, running_loss / 300))

            running_loss = 0.


def test():

    correct = 0
    total = 0

    with torch.no_grad():
        for data in test_loader:
            inputs, target = data
            inputs, target = inputs.to(device), target.to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, dim=1)
            total += target.size(0)

            correct += (predicted == target).sum().item()

        print('Accuracy on test set : %d %%  [%d, %d]' % (100 * correct / total, correct, total))
        y = correct * 100 / total

    return y


running_list = []

if __name__ == "__main__":
    """
    for epoch in range(10):
        
        train(epoch)
        running_list.append(test())"""

        # print(type(running_list))

# x = np.arange(0, 10)
# # y = running_list
# print(x)
#
# plt.plot(x, running_list)
# plt.show()
# num = 0
# for data in test_loader:
#     inputs, target = data
#     num += 1
#     print(num)
#     print(target)
