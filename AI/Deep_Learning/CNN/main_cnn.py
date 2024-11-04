# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 12:58:31 2024

@author: Kian Sartipzadeh
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
import torch.optim as optim
from torch.utils.data import DataLoader

from sklearn.model_selection import train_test_split


from utils import plot_ionogram
from storing_dataset import StoreDataset, MatchingPairs
from cnn_models import CNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)




radar_folder = "EISCAT_samples"
ionogram_folder = "Ionogram_images"



Pairs = MatchingPairs(ionogram_folder, radar_folder)

rad, ion = Pairs.find_pairs()





A = StoreDataset(ion, np.log10(rad), transforms.Compose([transforms.ToTensor()]))


# data_loader = DataLoader(A, batch_size=100, shuffle=True)


model = CNN()
criterion = nn.MSELoss()  
optimizer = optim.Adam(model.parameters(), lr=0.1)  # Use Adam optimizer
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=75, gamma=0.1)



# Function to count the number of parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Print the number of parameters
print(f"Total number of trainable parameters: {count_parameters(model)}")





# Define the number of epochs and split the dataset into train and validation sets
num_epochs = 100
train_size = int(0.7 * len(A))
val_size = len(A) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(A, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=50, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=50, shuffle=False)

# Move the model to the appropriate device
model = model.to(device)


# Initialize variables to track the best model
best_val_loss = float('inf')
best_model_weights = model.state_dict()


# Training loop
for epoch in range(num_epochs):
    model.train()  # Set model to training mode
    running_loss = 0.0
    for i, (images, targets) in enumerate(train_loader):
        images, targets = images.to(device), targets.to(device)

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)

        # Compute loss
        loss = criterion(outputs, targets)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Update running loss
        running_loss += loss.item()


    # Step the learning rate scheduler
    scheduler.step()
    
    
    # Validation loop
    model.eval()  # Set model to evaluation mode
    val_loss = 0.0
    val_outputs = []
    val_targets = []
    with torch.no_grad():
        for images, targets in val_loader:
            images, targets = images.to(device), targets.to(device)

            # Forward pass
            outputs = model(images)

            # Store outputs and targets for plotting
            val_outputs.append(outputs.cpu().numpy())
            val_targets.append(targets.cpu().numpy())

            # Compute loss
            loss = criterion(outputs, targets)
            val_loss += loss.item()

    # Calculate average validation loss
    avg_val_loss = val_loss / len(val_loader)

    # Save the best model weights
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        best_model_weights = model.state_dict()
        best_val_outputs = val_outputs
        best_val_targets = val_targets

    # Print epoch summary
    print(f"Epoch [{epoch + 1}/{num_epochs}], Training Loss: {running_loss / len(train_loader):.4f}, Validation Loss: {avg_val_loss:.4f}")


# Load the best model weights
model.load_state_dict(best_model_weights)


















