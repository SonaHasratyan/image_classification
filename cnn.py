"""
    Image classification task with pytorch, including dataloader, train, inference and much more...
"""


# dataloader, transformation
# multilayer NN, activation function
# loss and optimizer
# training loop (batch training)
# model evaluation
# GPU support

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size = 32
learning_rate = 0.001
n_epochs = 10
# todo: is the nn.Sequential more preferable and why
# todo: why do we normalize this way
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)

train_dataset = torchvision.datasets.CIFAR10(
    train=True, root="./CIFAR10", download=True, transform=transform
)

test_dataset = torchvision.datasets.CIFAR10(
    train=False, root="./CIFAR10", transform=transform
)

train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)

test_loader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size)

train_loader_example = iter(train_loader)
images, labels = next(train_loader_example)
# image size: 64 x 3 x 32 x 32
# 64 - batch size
# 3 - num of channels (3 colors)
# 32 x 32 - width & height of the img
print(images.shape, labels.shape)

for i in range(6):
    plt.subplot(2, 3, i + 1)
    plt.imshow(images[i][0], cmap="gray")
plt.show()

classes = (
    "plane",
    "car",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)


class ConvNN(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        in_channels = 3
        out_channels = 6
        kernel_size = 5
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size)
        in_channels = out_channels

        out_channels = 16
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size)

        hidden_size = 120
        self. flatten_size = out_channels * kernel_size * kernel_size
        self.fc1 = nn.Linear(self.flatten_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 2 * hidden_size // 3)
        self.fc3 = nn.Linear(2 * hidden_size // 3, len(classes))
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        out = self.pool(F.relu(self.conv1(x)))
        out = self.pool(F.relu(self.conv2(out)))

        out = out.view(-1, self.flatten_size)

        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = F.relu(self.fc3(out))
        # softmax is included in the CrossEntropy

        return out


model = ConvNN()

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


n_steps = len(train_loader)  # equal to batch_size
for epoch in range(n_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        outputs = model.forward(images)

        # todo: how is the loss calculated if outputs has 100x10 dimension and labels has 100 dimension
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss .backward()
        optimizer.step()

        if i % 100 == 0:
            print(f"epoch {epoch + 1}/{n_epochs}, step {i}/{n_steps}, loss: {loss.item():4f}")


with torch.no_grad():
    n_correct = 0
    n_samples = 0

    for (images, labels) in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)  # 100 x 10, whereas labels -> 100 x 1

        # we only need the index which is the predictions here
        _, predictions = torch.max(outputs, 1)

        n_correct += (predictions == labels).sum().item()
        n_samples += labels.shape[0]

acc = 100.0 * n_correct / n_samples

print(f"Accuracy: {acc}")
