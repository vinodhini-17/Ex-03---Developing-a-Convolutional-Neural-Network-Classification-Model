# DL-Convolutional Deep Neural Network for Image Classification
## AIM
To develop a convolutional neural network (CNN) classification model for the given dataset.

## THEORY
The Fashin-MNIST dataset consists of 70,000 grayscale images of clothes images, each of size 28×28 pixels. The task is to classify these images into their respective categories. CNNs are particularly well-suited for image classification tasks as they can automatically learn spatial hierarchies of features through convolutional layers, pooling layers, and fully connected layers.

Neural Network Model
Include the neural network model diagram.

## DESIGN STEPS
### STEP 1:
Import all the required libraries (PyTorch, TorchVision, NumPy, Matplotlib, etc.)

### STEP 2:
Download and preprocess the MNIST dataset using transforms.

### STEP 3:
Create a CNN model with convolution, pooling, and fully connected layers.

### STEP 4:
Set the loss function and optimizer. Move the model to GPU if available.

### STEP 5:
Train the model using the training dataset for multiple epochs.

### STEP 6:
Evaluate the model using the test dataset and visualize the results (accuracy, confusion matrix, classification report, sample prediction).

## PROGRAM

### Name:  VINODHINI K
### Reg No: 212223230245
```
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import make_grid

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
%matplotlib inline

transform = transforms.ToTensor()

train_data = datasets.FashionMNIST(root='../Data', train=True, download=True, transform=transform)
test_data = datasets.FashionMNIST(root='../Data', train=False, download=True, transform=transform)

class_names = ['T-shirt','Trouser','Sweater','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Boot']

train_loader = DataLoader(train_data, batch_size=10, shuffle=True)
test_loader = DataLoader(test_data, batch_size=10, shuffle=False)


for images, labels in train_loader:
    break

print("Labels:", labels)

im = make_grid(images, nrow=10)

plt.figure(figsize=(12,4))
plt.imshow(np.transpose(im.numpy(), (1,2,0)))
plt.axis('off')
plt.show()

for i in labels:
    print(class_names[i], end=' ')

class ConvolutionalNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=6,
            kernel_size=3,
            stride=1,
            padding=1
        )

        self.conv2 = nn.Conv2d(
            in_channels=6,
            out_channels=16,
            kernel_size=3,
            stride=1,
            padding=1
        )

        self.fc1 = nn.Linear(16*7*7, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, X):

        X = F.relu(self.conv1(X))
        X = F.max_pool2d(X, 2, 2)

        X = F.relu(self.conv2(X))
        X = F.max_pool2d(X, 2, 2)

        X = X.view(-1, 16*7*7)

        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))

        X = self.fc3(X)

        return X


torch.manual_seed(101)

model = ConvolutionalNetwork()

total_params = 0

for param in model.parameters():
    total_params += param.numel()

print("Total Parameters:", total_params)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.01
)


epochs = 5

for i in range(epochs):

    for X_train, y_train in train_loader:

        y_pred = model(X_train)

        loss = criterion(y_pred, y_train)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

    print(f"Epoch {i+1}/{epochs} completed")

correct = 0
total = 0

model.eval()

with torch.no_grad():

    for X_test, y_test in test_loader:

        y_val = model(X_test)

        predicted = torch.max(y_val.data, 1)[1]

        total += y_test.size(0)

        correct += (predicted == y_test).sum()

accuracy = correct.item() * 100 / total

print(f'Accuracy: {accuracy:.2f}%')
```
## OUTPUT
## Epochs
<img width="227" height="101" alt="image" src="https://github.com/user-attachments/assets/270b7afd-3dfe-40d9-9fa8-215399bdebdb" />

## A batch of images
<img width="1182" height="197" alt="image" src="https://github.com/user-attachments/assets/c1183571-4248-4625-83ed-39cc9ecd17cf" />

## Accuracy Score
<img width="242" height="42" alt="image" src="https://github.com/user-attachments/assets/b4e8bfde-e968-4dd9-8bdb-c589cfa14178" />

## RESULT
Thus the CNN model was trained and tested successfully on the MNIST dataset.
