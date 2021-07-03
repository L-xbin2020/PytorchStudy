from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
import torch


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


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.linear11 = torch.nn.Linear(784, 512)
        self.linear12 = torch.nn.Linear(512, 256)
        self.linear13 = torch.nn.Linear(256, 128)
        self.linear14 = torch.nn.Linear(128, 64)
        self.linear15 = torch.nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = F.relu(self.linear11(x))
        x = F.relu(self.linear12(x))
        x = F.relu(self.linear13(x))
        x = F.relu(self.linear14(x))

        return self.linear15(x)


model = Net()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)


def train(epoch):
    running_loss = 0.0
    for batch_idx, data in enumerate(train_loader, 0):
        inputs, target = data
        # 传入gpu
        inputs, target = inputs.to(device), target.to(device)

        optimizer.zero_grad()

        output = model(inputs)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if batch_idx % 300 == 299:
            print('[%d, %5d] loss: %.3f' % (epoch + 1, batch_idx + 1, running_loss / 300))
            running_loss = 0.0
    

def test():
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data

            images, labels = images.to(device), labels.to(device)

            output = model(images)
            _, predicted = torch.max(output.data, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy on test set: %d %%' % (100 * correct / total))


if __name__ == '__main__':

    for epoch in range(10):
        train(epoch)
        test()