import torch
from torch import nn , optim
import numpy
import matplotlib.pyplot as plt
import PIL
from time import time

from PIL import Image
from torchvision import datasets, transforms

# Define a transform to normalize the data
transform = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.5,), (0.5,)),
                              ])


# Download and load the training data
trainset = datasets.MNIST('drive/My Drive/mnist/MNIST_data/', download=True, train=True, transform=transform)
valset = datasets.MNIST('drive/My Drive/mnist/MNIST_data/', download=True, train=False, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
valloader = torch.utils.data.DataLoader(valset, batch_size=64, shuffle=True)



dataiter = iter(trainloader)
images, labels = dataiter.next()
print(type(images))
print(images.shape)
print(labels.shape)
print(labels[0])


result = transforms.ToPILImage()(images[0])
result.show()


## 28 * 28 Bildgröße -> somit hat jedes Pixel ein input

## Xaviar init
## Relu activation
inputSize = 784
hiddenLayerSize = 100
outputSize = 5


## Network bauen

model = nn.Sequential(
    #Input
    nn.Linear(inputSize, hiddenLayerSize),
    nn.ReLU(),
    #Hidden Layer 1
    nn.Linear(hiddenLayerSize, hiddenLayerSize),
    nn.ReLU(),
    # Hidden Layer 2
    nn.Linear(hiddenLayerSize, hiddenLayerSize),
    nn.ReLU(),
    # Hidden Layer 3
    nn.Linear(hiddenLayerSize, hiddenLayerSize),
    nn.ReLU(),
    # Hidden Layer 4
    nn.Linear(hiddenLayerSize, hiddenLayerSize),
    nn.ReLU(),
    # Hidden Layer 5
    nn.Linear(hiddenLayerSize, hiddenLayerSize),
    nn.ReLU(),
    #Output
    nn.Linear(hiddenLayerSize, outputSize),
    nn.Softmax(dim=1),
)


images, labels = next(iter(trainloader))
images = images.view(images.shape[0], -1)

logps = model(images)

## Adam optimazation
## L2 regular
optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
optimizer.zero_grad()

### Training
time0 = time()

epochs = 15
for e in range(epochs):
    running_loss = 0
    for images, labels in trainloader:
        print("ia am the label",labels)
        # Flatten MNIST images into a 784 long vector
        images = images.view(images.shape[0], -1)

        # Training pass
        optimizer.zero_grad()

        output = model(images)

        # This is where the model learns by backpropagating

        # And optimizes its weights here
        optimizer.step()

    else:
        print("Epoch {} - Training loss: {}".format(e, running_loss / len(trainloader)))
print("\nTraining Time (in minutes) =", (time() - time0) / 60)

