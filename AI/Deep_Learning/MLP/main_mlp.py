# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 14:42:19 2024

@author: Kian Sartipzadeh
"""


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(MLP, self).__init__()
        
        # Layers
        self.FC1  = nn.Linear(in_dim, hidden_dim)
        self.FC2  = nn.Linear(hidden_dim, out_dim)
        
        # Activation
        self.sig = nn.Sigmoid()
        
    def forward(self, x):
        x = self.FC1(x)
        x = self.sig(x)
        x = self.FC2(x)
        x = self.sig(x)
        return x
        
        


# class to store data
class BlobDataset(Dataset):
    
    def __init__(self, data, targets):
        
        # Converting to tensors
        self.data = torch.tensor(data, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.long)
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]



def plot_decision_boundary(model, X, y, device):
    # Set the model to evaluation mode
    model.eval()

    # Define the grid over which to evaluate the model
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))

    # Flatten the grid to pass it through the model
    grid = np.c_[xx.ravel(), yy.ravel()]
    grid_tensor = torch.tensor(grid, dtype=torch.float32).to(device)

    # Get model predictions for the grid
    with torch.no_grad():
        Z = model(grid_tensor)
        Z = torch.argmax(Z, dim=1)
        Z = Z.cpu().numpy()

    # Reshape the predictions back into the grid shape
    Z = Z.reshape(xx.shape)

    # Plot the decision boundary and the data points
    plt.contourf(xx, yy, Z, alpha=0.5, cmap=plt.cm.RdYlBu)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, edgecolor='k', cmap=plt.cm.RdYlBu)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title("Decision Boundary")
    plt.show()






# data
X, y = make_blobs(n_samples=1000, centers=2, n_features=2)


# Standardize
scaler = StandardScaler()
X = scaler.fit_transform(X)


# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=1)


# plot
plt.figure()
plt.title("Blobs dataset")
plt.scatter(X_train[:,0][y_train==0], X_train[:,1][y_train==0], color="C0", label="Class 1")
plt.scatter(X_train[:,0][y_train==1], X_train[:,1][y_train==1], color="C1", label="Class 2")
plt.xlabel("X1")
plt.ylabel("X2")
plt.legend()
plt.show()




# Storing datasets
train_dataset = BlobDataset(X_train, y_train)
test_dataset = BlobDataset(X_test, y_test)


# Create DataLoader
train_loader = DataLoader(train_dataset, batch_size=100, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)




# Initialize network
model = MLP(2, 10, 2).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)



epochs = 100

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    
    print(f'Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}')



# Test the model
model.eval()
correct = 0
total = 0




with torch.no_grad():
    for inputs, labels in test_loader:
        # Move inputs and labels to the device
        inputs, labels = inputs.to(device), labels.to(device)
        
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the model on the test set: {100 * correct / total:.2f}%')






