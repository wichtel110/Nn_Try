import torch
from torch import nn , optim
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

#Settings
inputSize = 784
hiddenLayerSize = 100
outputSize = 5

epochs = 100
#LearningRate AdamOptimizer
learning_rate = 0.01
l2_reg = 0.001

class Net(nn.Module):
    gainFactor = 1

    def __init__(self):
        super(Net, self).__init__()
        #InputLayer
        self.fc1 = nn.Linear(inputSize, hiddenLayerSize)
        nn.init.xavier_normal_(self.fc1.weight, self.gainFactor)
        self.bn1 = nn.BatchNorm1d(hiddenLayerSize)
        #HiddenLayer 1
        self.fc2 = nn.Linear(hiddenLayerSize, hiddenLayerSize)
        nn.init.xavier_normal_(self.fc2.weight, self.gainFactor)
        self.bn2 = nn.BatchNorm1d(hiddenLayerSize)

        #HiddenLayer 2
        self.fc3 = nn.Linear(hiddenLayerSize, hiddenLayerSize)
        nn.init.xavier_normal_(self.fc3.weight, self.gainFactor)
        self.bn3 = nn.BatchNorm1d(hiddenLayerSize)

        #HiddenLayer 3
        self.fc4 = nn.Linear(hiddenLayerSize, hiddenLayerSize)
        nn.init.xavier_normal_(self.fc4.weight, self.gainFactor)
        self.bn4 = nn.BatchNorm1d(hiddenLayerSize)

        #HiddenLayer 4
        self.fc5 = nn.Linear(hiddenLayerSize, hiddenLayerSize)
        nn.init.xavier_normal_(self.fc5.weight, self.gainFactor)
        self.bn5 = nn.BatchNorm1d(hiddenLayerSize)

        #HiddenLayer 5
        self.fc6 = nn.Linear(hiddenLayerSize, hiddenLayerSize)
        nn.init.xavier_normal_(self.fc6.weight, self.gainFactor)
        self.bn6 = nn.BatchNorm1d(hiddenLayerSize)

        #Output
        self.fc7 = nn.Linear(hiddenLayerSize, outputSize)
        nn.init.xavier_normal_(self.fc7.weight, self.gainFactor)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(self.bn1(x))
        x = F.dropout(x)
        x = self.fc2(x)
        x = F.relu(self.bn2(x))
        x = F.dropout(x)
        x = self.fc3(x)
        x = F.relu(self.bn3(x))
        x = F.dropout(x)
        x = self.fc4(x)
        x = F.relu(self.bn4(x))
        x = F.dropout(x)
        x = self.fc5(x)
        x = F.relu(self.bn5(x))
        x = F.dropout(x)
        x = self.fc6(x)
        x = F.relu(self.bn6(x))
        x = F.dropout(x)
        x = self.fc7(x)
        x = F.relu(x)
        x = F.dropout(x)

        output = F.softmax(x, dim=1)

        return output

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def getAccuracy(model, data, val=False):
    correct = 0
    total = 0
    losses = []
    model.eval()
    for imgs, labels in data:
        imgs = imgs.view(imgs.size(0), -1)

        output = model(imgs)
        _, predicted = torch.max(output.data, 1)
        if val:
            loss = loss_func(output, labels)
            losses.append(loss.data.cpu().numpy())

        correct += (predicted == labels).sum().item()
        total += imgs.shape[0]


    if val:
        losses_val.append(np.mean(losses).item())
    print('Accurarcy: {}'.format(correct / total))
    return correct / total

# Define a transform to normalize the data
transform = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.5,), (0.5,)),
                              ])

# Download and load the training data
trainset = datasets.MNIST('drive/My Drive/mnist/MNIST_data/', download=True, train=True, transform=transform)
valset = datasets.MNIST('drive/My Drive/mnist/MNIST_data/', download=True, train=False, transform=transform)

idx = (trainset.targets < 5)

trainset.targets = trainset.targets[idx]
trainset.data = trainset.data[idx]

trainloader = torch.utils.data.DataLoader(trainset, shuffle=True, batch_size=512)

idx = (valset.targets < 5)

valset.targets = valset.targets[idx]
valset.data = valset.data[idx]

valloader = torch.utils.data.DataLoader(valset, shuffle=False, batch_size=512)

## Creating Network Module

net = Net()

#Adam optimizer with l2 regularization
adamOptimizer = optim.Adam(net.parameters(), lr=learning_rate,  weight_decay=l2_reg)
loss_func = nn.CrossEntropyLoss()

iters, losses_all, losses_val, train_acc, val_acc =[], [], [], [], []

for epoch in range(epochs):
    losses = []

    running_loss = 0.0
    for index, (image, label) in enumerate(trainloader):
        target, x = label, image
        #Flatten the Image because of a fully connected layer
        x = x.view(x.size(0), -1)

        adamOptimizer.zero_grad()

        out = net(x)
        loss = loss_func(out, target)

        loss.backward()
        adamOptimizer.step()
        running_loss += loss.item()

        losses.append(loss.data.cpu().numpy())

        if (index+1) % 100 == 0 or (index+1) == len(trainloader):
            print('==>>> epoch: {}, index: {}, train loss: {:.6f}'.format(
                epoch, index+1, np.mean(losses).item()))

            running_loss = 0.0

    if epoch % 20 == 0:
        PATH = './trained_network_epoch_{}.pth'.format(epoch)
        torch.save(net.state_dict(), PATH)


    iters.append(epoch)
    losses_all.append(np.mean(losses).item())  # compute *average* loss
    train_acc.append(getAccuracy(net, trainloader))  # compute training accuracy
    val_acc.append(getAccuracy(net, valloader, val=True))  # compute validation accuracy



PATH = './trained_network_final.pth'
torch.save(net.state_dict(), PATH)

# plotting
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.title("Training Curve")
plt.plot(iters, losses_all, label="Train")
plt.plot(iters, losses_val, label="Val")
plt.legend(loc='best')
plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.subplot(1, 2, 2)
plt.title("Training Curve")
plt.plot(iters, train_acc, label="Train")
plt.plot(iters, val_acc, label="Validation")
plt.xlabel("Epoch")
plt.ylabel("Training Accuracy")
plt.legend(loc='best')
plt.show()

print("Final Training Accuracy: {}".format(train_acc[-1]))
print("Final Validation Accuracy: {}".format(val_acc[-1]))

# Aufgaben
# AUfgabe 1.1
# (x) 5 Hidden Layer with 100 neurons
# (x) Xavier Init
# (x) Relu activation
# (x) Adam Optimization
# (x) L2 regularization
# (x) softmax output with 5 neurons
# (x) save CheckPoints
# (x) save final Model
#
#Aufgabe 1.2
#
# (x) Tune hyperparameter cross-validation
#
# Accuracy around 97%

#Aufgabe 1.3
#
# (x) add Batch Normalization
# (x) Compare learning curves
#
# Model ist nicht besser als vorher. Genauigkeit nimmt ab.
# Konvergiert mit einem höheren Loss.
#
#
#AUfgabe 1.4
#
# (x) Is Model Overfitting?
# (x) add Dropout to every layer
#
#

