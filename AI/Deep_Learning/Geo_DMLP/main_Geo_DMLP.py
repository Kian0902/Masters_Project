# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 15:59:50 2025

@author: Kian Sartipzadeh
"""

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torchvision import transforms
import torch.optim as optim
from torch.utils.data import DataLoader

from storing_dataset import StoreDataset, MatchingPairs
from Geo_DMLP_model import GeoDMLP, he_initialization
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)



def plot_losses(train_losses, val_losses):
    """
    Plots the training and validation loss for each epoch.

    Args:
        train_losses (list or array-like): Training loss values per epoch.
        val_losses (list or array-like): Validation loss values per epoch.
    """
    epochs = range(56, len(train_losses) + 1)
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, train_losses[55:], 'bo-', label='Training Loss')
    plt.plot(epochs, val_losses[55:], 'ro-', label='Validation Loss')
    plt.title('Training and Validation Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()




radar_folder = "training_data/train_eiscat_folder"
geophys_folder = "training_data/train_geophys_folder"



# Creating Training pairs
print("Matching Pairs...")
Pairs = MatchingPairs(geophys_folder, radar_folder)
rad, geo = Pairs.find_pairs()
print("Done!")


# Mitigatin for unusual low values
rad = np.abs(rad)
rad[rad < 1e5] = 1e6


print("Storing Data...")
# Storing training pairs as object
A = StoreDataset(geo, np.log10(rad))
print("Done!")




# Split the dataset into train and validation sets
train_size = int(0.8 * len(A))
val_size = len(A) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(A, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=val_size, shuffle=True)


num_epochs = 2500
model = GeoDMLP().to(device)
model.apply(he_initialization)


# criterion = nn.L1Loss()
criterion = nn.MSELoss()
criterion = nn.HuberLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# scheduler1 = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)
scheduler2 = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2100, gamma=0.1)



# Initialize variables to track the best model
best_val_loss = float('inf')
best_model_weights = model.state_dict()



train_losses = []
val_losses = []
for epoch in range(num_epochs):
    model.train()  # Set model to training mode
    running_loss = 0.0
    for i, (geodata, targets) in enumerate(train_loader):
        geodata, targets = geodata.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(geodata)
        
        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    
    avg_train_loss = running_loss / len(train_loader)
    train_losses.append(avg_train_loss)
    # Step the learning rate scheduler
    # scheduler1.step()
    scheduler2.step()
    
    # Validation loop
    model.eval()  # Set model to evaluation mode
    val_loss = 0.0
    val_outputs = []
    val_targets = []
    with torch.no_grad():
        for geodata, targets in val_loader:
            geodata, targets = geodata.to(device), targets.to(device)

            
            outputs = model(geodata)
            
            # Store outputs and targets for plotting
            val_outputs.append(outputs.cpu().numpy())
            val_targets.append(targets.cpu().numpy())
            
            # Compute loss
            loss = criterion(outputs, targets)
            val_loss += loss.item()
            
    # Calculate average validation loss
    avg_val_loss = val_loss / len(val_loader)
    val_losses.append(avg_val_loss)
    
    # Save the best model weights
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        best_model_weights = model.state_dict()
        best_val_outputs = val_outputs
        best_val_targets = val_targets
    
    # Print epoch summary
    # if epoch % 10 == 0:
    print(f"Epoch [{epoch + 1}/{num_epochs}], Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}")



# Save the best model to a file
torch.save(best_model_weights, 'Geo_DMLPv1_very_long_dropout02.pth')
plot_losses(train_losses, val_losses)









