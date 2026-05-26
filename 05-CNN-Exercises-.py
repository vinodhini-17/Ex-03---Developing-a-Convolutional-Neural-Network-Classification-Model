#!/usr/bin/env python
# coding: utf-8

# # CNN Exercises
# For these exercises we'll work with the <a href='https://www.kaggle.com/zalando-research/fashionmnist'>Fashion-MNIST</a> dataset, also available through <a href='https://pytorch.org/docs/stable/torchvision/index.html'><tt><strong>torchvision</strong></tt></a>. Like MNIST, this dataset consists of a training set of 60,000 examples and a test set of 10,000 examples. Each example is a 28x28 grayscale image, associated with a label from 10 classes:
# 0. T-shirt/top
# 1. Trouser
# 2. Pullover
# 3. Dress
# 4. Coat
# 5. Sandal
# 6. Shirt
# 7. Sneaker
# 8. Bag
# 9. Ankle boot
# 
# <div class="alert alert-danger" style="margin: 10px"><strong>IMPORTANT NOTE!</strong> Make sure you don't run the cells directly above the example output shown, <br>otherwise you will end up writing over the example output!</div>

# ## Perform standard imports, load the Fashion-MNIST dataset
# Run the cell below to load the libraries needed for this exercise and the Fashion-MNIST dataset.<br>
# PyTorch makes the Fashion-MNIST dataset available through <a href='https://pytorch.org/docs/stable/torchvision/datasets.html#fashion-mnist'><tt><strong>torchvision</strong></tt></a>. The first time it's called, the dataset will be downloaded onto your computer to the path specified. From that point, torchvision will always look for a local copy before attempting another download.

# In[1]:


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
get_ipython().run_line_magic('matplotlib', 'inline')

transform = transforms.ToTensor()

train_data = datasets.FashionMNIST(root='../Data', train=True, download=True, transform=transform)
test_data = datasets.FashionMNIST(root='../Data', train=False, download=True, transform=transform)

class_names = ['T-shirt','Trouser','Sweater','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Boot']


# ## 1. Create data loaders
# Use DataLoader to create a <tt>train_loader</tt> and a <tt>test_loader</tt>. Batch sizes should be 10 for both.

# In[2]:


# CODE HERE

train_loader = DataLoader(train_data, batch_size=10, shuffle=True)
test_loader = DataLoader(test_data, batch_size=10, shuffle=False)


# In[2]:


# DON'T WRITE HERE


# ## 2. Examine a batch of images
# Use DataLoader, <tt>make_grid</tt> and matplotlib to display the first batch of 10 images.<br>
# OPTIONAL: display the labels as well

# In[3]:


# CODE HERE


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




# In[3]:


# DON'T WRITE HERE
# IMAGES ONLY


# In[4]:


# DON'T WRITE HERE
# IMAGES AND LABELS


# ## Downsampling
# <h3>3. If a 28x28 image is passed through a Convolutional layer using a 5x5 filter, a step size of 1, and no padding, what is the resulting matrix size?</h3>

# <div style='border:1px black solid; padding:5px'>
# <br><br>
# </div>

# In[4]:


##################################################
###### ONLY RUN THIS TO CHECK YOUR ANSWER! ######
################################################

# Run the code below to check your answer:
conv = nn.Conv2d(1, 1, 5, 1)
for x,labels in train_loader:
    print('Orig size:',x.shape)
    break
x = conv(x)
print('Down size:',x.shape)


# ### 4. If the sample from question 3 is then passed through a 2x2 MaxPooling layer, what is the resulting matrix size?

# <div style='border:1px black solid; padding:5px'>
# <br><br>
# </div>

# In[5]:


##################################################
###### ONLY RUN THIS TO CHECK YOUR ANSWER! ######
################################################

# Run the code below to check your answer:
x = F.max_pool2d(x, 2, 2)
print('Down size:',x.shape)


# ## CNN definition
# ### 5. Define a convolutional neural network
# Define a CNN model that can be trained on the Fashion-MNIST dataset. The model should contain two convolutional layers, two pooling layers, and two fully connected layers. You can use any number of neurons per layer so long as the model takes in a 28x28 image and returns an output of 10. Portions of the definition have been filled in for convenience.

# In[6]:


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


# ## Trainable parameters

# ### 6. What is the total number of trainable parameters (weights & biases) in the model above?
# Answers will vary depending on your model definition.

# <div style='border:1px black solid; padding:5px'>
# <br><br>
# </div>

# In[7]:


# CODE HERE
total_params = 0

for param in model.parameters():
    total_params += param.numel()

print("Total Parameters:", total_params)


# ### 7. Define loss function & optimizer
# Define a loss function called "criterion" and an optimizer called "optimizer".<br>
# You can use any functions you want, although we used Cross Entropy Loss and Adam (learning rate of 0.001) respectively.

# In[8]:


# CODE HERE
criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.01
)



# In[6]:


# DON'T WRITE HERE


# ### 8. Train the model
# Don't worry about tracking loss values, displaying results, or validating the test set. Just train the model through 5 epochs. We'll evaluate the trained model in the next step.<br>
# OPTIONAL: print something after each epoch to indicate training progress.

# In[9]:


# CODE HERE

epochs = 5

for i in range(epochs):

    for X_train, y_train in train_loader:

        y_pred = model(X_train)

        loss = criterion(y_pred, y_train)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

    print(f"Epoch {i+1}/{epochs} completed")




# In[7]:





# ### 9. Evaluate the model
# Set <tt>model.eval()</tt> and determine the percentage correct out of 10,000 total test images.

# In[10]:


# CODE HERE

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




# In[8]:





# ## Great job!
