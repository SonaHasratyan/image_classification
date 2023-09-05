import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from torch.optim import lr_scheduler


writer = SummaryWriter("runs/mnist2")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

input_size = 28 * 28
hidden_size = 500
batch_size = 64
num_epochs = 2
learning_rate = 0.01
num_classes = 10

train_dataset = torchvision.datasets.MNIST(
    train=True, root="./data", transform=transforms.ToTensor(), download=True
)

test_dataset = torchvision.datasets.MNIST(
    train=False, root="./data", transform=transforms.ToTensor()
)

train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset, shuffle=True, batch_size=batch_size
)

test_loader = torch.utils.data.DataLoader(
    dataset=test_dataset, shuffle=False, batch_size=batch_size
)


examples = iter(train_loader)
samples, labels = next(examples)  # .next() didn't work


for i in range(12):
    plt.subplot(3, 4, i + 1)
    plt.imshow(samples[i][0], cmap="gray")
# plt.show()
img_grid = torchvision.utils.make_grid(samples)
writer.add_image("mnist images", img_grid)
writer.close()


class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # super(NeuralNet, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.relu(out)
        out = self.layer2(out)

        return out


model = NeuralNet(input_size, hidden_size, num_classes)

# criterion already includes softmax, that's why we didn't include it in the NN
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

writer.add_graph(model, samples.reshape(-1, 28*28))
writer.close()

running_loss = 0.0
running_correct = 0
n_steps = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):


        # an  image has a size of 100 x 1 x 28 x 28
        # we want it to have a size of 100 x 784, let's reshape it!
        images = images.reshape(-1, 28 * 28).to(device)
        labels = labels.to(device)

        # forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(outputs, 1)
        running_correct += (predicted == labels).sum().item()

        if (i + 1) % 100 == 0:
            print(f"epoch {(epoch + 1)} / {num_epochs}, step {i + 1}/{n_steps}, loss = {running_loss:.4f}")

            # we sum this up for 100 steps = > we divide the running_loss by 100
            writer.add_scalar("training loss", running_loss / 100, n_steps * epoch + i)
            writer.add_scalar("accuracy", running_correct / 100, n_steps * epoch + i)

            running_loss = 0.0
            running_correct = 0


labels_list = []
predictions_list = []
with torch.no_grad():
    n_samples = 0
    n_correct = 0

    for images, labels in test_loader:
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)

        outputs = model(images)

        # value, index - in MNIST the index corresponds to the class
        _, predictions = torch.max(outputs, 1)
        n_samples += labels.shape[0]  # actually this is the batch size
        n_correct += (predictions == labels).sum().item()

        class_predictions = [F.softmax(output, dim=0) for output in outputs]
        predictions_list.append(class_predictions)
        labels_list.append(predictions)

predictions_list = torch.cat([torch.stack(batch) for batch in predictions_list])
labels_list = torch.cat(labels_list)
acc = 100.0 * n_correct / n_samples

for i in range(10):  # 0,...,9 are our classes' labels
    labels_i = labels_list == i
    preds_i = predictions_list[:, i]
    writer.add_pr_curve(str(i), labels_i, preds_i, global_step=0)  # precision for each class
    writer.close()

print(f"Accuracy = {acc}")
