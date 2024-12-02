# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 14:42:19 2024

@author: Kian Sartipzadeh
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import numpy as np
from sklearn.model_selection import train_test_split

from mlp_models import MLP19, MLP19ION
from storing_dataset import StoreDataset
from training_mlp import train_model, plot_losses
from testing_mlp import test_model, plot_results


# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)




# Importing data
data_sp19_ion = np.load('SP19_Ionogram_data.npy')
data_eiscat = np.load('eiscat_data_SI.npy')




X_train, X_test, y_train, y_test = train_test_split(data_sp19_ion, data_eiscat, train_size=0.8, shuffle=True, random_state=42)



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
val_loader = DataLoader(val_dataset, batch_size=100, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=True)

# Initialize model, loss function, optimizer, and scheduler
in_dim = X_train.shape[1]

model = MLP19ION().to(device)
loss_function = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)
# scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.1)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 1500, 3000], gamma=0.1)
# scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=200)



num_epochs = 4000

model, train_loss, val_loss = train_model(model, train_loader, val_loader, loss_function, optimizer, scheduler, device, num_epochs)


plot_losses(train_loss, val_loss)


best_model_path = 'best_model.pth'
model.load_state_dict(torch.load(best_model_path, weights_only=True))


avg_test_loss, r2, accuracy, predicted_outputs, true_outputs = test_model(model, test_loader, loss_function)

print(f'Test Loss: {avg_test_loss:.4f}')
print(f'R² Score: {r2:.4f}')
print(f'Accuracy (within 2% tolerance): {accuracy:.2f}%')



plot_results(predicted_outputs, true_outputs, num_plots=100)




















