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
from sklearn.model_selection import train_test_split

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)




class StoreDataset(Dataset):
    def __init__(self, data, targets):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]




class MLP(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(MLP, self).__init__()
        
        # MLP layers
        self.layers = nn.Sequential(nn.Linear(in_dim, 124),
                                    nn.ReLU(),
                                    nn.Linear(124, 64),
                                    nn.ReLU(),
                                    nn.Linear(64, out_dim)
                                    )

 
        # Activation
        # self.relu = nn.ReLU()
        self.softplus_H = nn.Softplus(beta=0.1, threshold=10)
        # self.softplus_Z = nn.Softplus(beta=0.0074, threshold=10)
        self.softplus_N = nn.Softplus(beta=0.15, threshold=10)
        
        
    
    def double_chapman(self, x, z):
        HE_below, HF_below, HE_above, HF_above, nE_peak, nF_peak, zE_peak, zF_peak = x.split(1, dim=1)
        
        # print(f'HE_b {HE_below[1]}')
        # print(f'HE_a {HE_above[1]}')
        # print(f'HF_b {HF_below[1]}')
        # print(f'HF_a {HF_above[1]}')
    
        # print(f'nE {nE_peak[1]}')
        # print(f'nF {nF_peak[1]}')
        # print(f'zE {zE_peak[1]}')
        # print(f'zF {zF_peak[1]}')
        # print("--------")

        
        
        # Adding epsilon to avoid division by zero
        HE = torch.where(z < zE_peak, HE_below, HE_above)
        HF = torch.where(z < zF_peak, HF_below, HF_above)
    
        # Clamping to avoid overflow in torch.exp
        neE = nE_peak * torch.exp(1 - ((z - zE_peak) / HE) - torch.exp(-((z - zE_peak) / HE)))
        neF = nF_peak * torch.exp(1 - ((z - zF_peak) / HF) - torch.exp(-((z - zF_peak) / HF)))
        
        ne = neE + neF
        
        
        # print(ne)
        
        return ne
        
    
    
    def forward(self, x, z):


        x = self.layers(x)
        
        x_H = x[:,:4]
        x_N = x[:,4:]
        
        
        x_H = self.softplus_H(x_H)
        x_N = self.softplus_N(x_N)
        
        
        x_final =  torch.cat([x_H, x_N], dim=1)
        
        batch_size = x_final.size(0)
        z = z.unsqueeze(0).expand(batch_size, -1).to(device)
        
        chapman_output = self.double_chapman(x_final, z)
        
        
        return chapman_output
    



# Importing data
data_sp19 = np.load('sp19_data.npy')
data_eiscat = np.load('eiscat_data.npy')




X_train, X_test, y_train, y_test = train_test_split(data_sp19, data_eiscat, train_size=0.8, shuffle=True)



y_train[y_train < 10**5] = 10**5
y_test[y_test < 10**5] = 10**5


# Apply log10 to all the datasets
y_train = np.log10(y_train)
y_test = np.log10(y_test)


y_train = np.round(y_train, decimals=3)
y_test = np.round(y_test, decimals=3)




# Split training data further into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, train_size=0.8, shuffle=True)

# Creating datasets and data loaders for training, validation, and test sets
train_dataset = StoreDataset(X_train, y_train)
val_dataset = StoreDataset(X_val, y_val)
test_dataset = StoreDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=100, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=100, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=True)


