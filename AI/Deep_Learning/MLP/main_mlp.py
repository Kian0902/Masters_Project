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



class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(MLP, self).__init__()
        
        self.FC1  = nn.Linear(in_dim, hidden_dim)
        self.FC2  = nn.Linear(hidden_dim, out_dim)
        
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
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)











        






















