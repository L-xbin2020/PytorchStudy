from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import torch

import os  # 一定要有这个设定，要不然报错，也生成不了图


from matplotlib import pyplot as plt
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


# for data in train_dataset:
#     inputs, targets = data
#     image = inputs[0]
#
#     print(image)

x = np.arange(1, 11)
y = np.random.randn(10)
print(x)
print(y)
plt.plot(x, y)
plt.show()

