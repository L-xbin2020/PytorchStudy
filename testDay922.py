import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader

batch_size = 64
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST(root='./',
                               train=True,
                               transform=transform,
                               download=True)

train_loader = DataLoader(dataset=train_dataset,
                          batch_size=batch_size,
                          shuffle=True)


test_dataset = datasets.MNIST(root='./',
                              train=False,
                              download=True,
                              transform=transform)

test_loader = DataLoader(test_dataset,
                         shuffle=False,
                         batch_size=batch_size)


class MyModel(torch.nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.lin1 = torch.nn.Linear(784, 512)
        self.lin2 = torch.nn.Linear(512, 256)
        self.lin3 = torch.nn.Linear(256, 128)
        self.lin4 = torch.nn.Linear(128, 64)
        self.lin5 = torch.nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        x = F.relu(self.lin3(x))
        x = F.relu(self.lin4(x))

        return self.lin5(x)


model = MyModel()

criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)


def train(epoch):
    running_loss = 0

    for indexs, data in enumerate(train_loader, 0):

        inputs, labels = data

        y_pred = model(inputs)

        optimizer.zero_grad()
        loss = criterion(y_pred, labels)
        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        if indexs % 300 == 299:
            print('[%d, %5d], loss: %.3f' % (epoch + 1, indexs + 1, running_loss / 300))
            running_loss = 0.0

def test():

    correct = 0
    total = 0

    with torch.no_grad():
        for data in test_loader:

            inputs, labels = data
            output = model(inputs)
            # print(output.data)

            value1, predicted = torch.max(output.data, dim=1)
            print(value1)
            print('*' * 50)
            print(predicted)
            print(labels.data)
            print(len(labels))
            total += labels.size(0)

            correct += (predicted == labels).sum().item()

    print('Accuracy on test set: %d %%' % (100 * correct / total))

#
# def train(epoch):
#     running_loss = 0.0
#     for batch_idx, data in enumerate(train_loader, 0):
#         inputs, lables = data
#
#         y_pred = model(inputs)
#
#         optimizer.zero_grad()
#         loss = criterion(y_pred, lables)
#
#         loss.backward()
#
#         optimizer.step()
#         running_loss += loss.item()
#         if batch_idx % 300 == 299:
#             print('[%d, %5d], loss: %.3f' % (epoch + 1, batch_idx, running_loss / 300 ))
#             running_loss =  0.0


# def test():
#     correct = 0
#
#     total = 0
#     with torch.no_grad():
#         for data in test_loader:
#
#             inputs, lables = data
#
#             outputs = model(inputs)
#
#             total += lables.size(0)
#
#             _, predicted = torch.max(outputs.data, dim=1)
#             correct += (predicted == lables).sum().item()
#
#     print('Accuracy on test set %d %% ' % (100 * correct / total))



if __name__ == '__main__':

    for epoch in range(10):
        train(epoch)
        test()