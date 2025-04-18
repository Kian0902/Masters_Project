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
ionogram_folder = "Good_Ionograms_Images"



Pairs = MatchingPairs(ionogram_folder, radar_folder)

rad, ion = Pairs.find_pairs()

rad = np.abs(rad)

# for n in rad:
#     if np.isnan(n).any():
#         print(f"NaN detected {key}: {array.shape}")
#     elif (n < 0).any():
#         print(f"Negative value detected in array: {n}")
    
    
A = StoreDataset(ion, np.log10(rad), transforms.Compose([transforms.ToTensor()]))


print("Data stored")



# Define the number of epochs and split the dataset into train and validation sets
num_epochs = 100
train_size = int(0.8 * len(A))
val_size = len(A) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(A, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=100, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=100, shuffle=True)



model = CNN()
criterion = nn.HuberLoss()
optimizer = optim.Adam(model.parameters(), lr=0.1)  # Use Adam optimizer
scheduler1 = torch.optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.1)
# scheduler2 = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)


# Function to count the number of parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Print the number of parameters
print(f"Total number of trainable parameters: {count_parameters(model)}")



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
    scheduler1.step()
    # scheduler2.step()
    
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


# Plot model outputs vs target values from validation data
import matplotlib.pyplot as plt
best_val_outputs = np.concatenate(best_val_outputs, axis=0)
best_val_targets = np.concatenate(best_val_targets, axis=0)



r_h = np.array([[ 91.5687711 ],[ 94.57444598],[ 97.57964223],[100.57010953],
        [103.57141624],[106.57728701],[110.08393175],[114.60422289],
        [120.1185208 ],[126.61221111],[134.1346149 ],[142.53945817],
        [152.05174717],[162.57986185],[174.09833378],[186.65837945],
        [200.15192581],[214.62769852],[230.12198695],[246.64398082],
        [264.11728204],[282.62750673],[302.15668686],[322.70723831],
        [344.19596481],[366.64409299],[390.113117  ]])

for i in range(0, len(best_val_outputs)):
    plt.figure(figsize=(10, 6))
    plt.plot(best_val_targets[i, :], r_h.flatten(), label='True')
    plt.plot(best_val_outputs[i, :], r_h.flatten(), label='Pred')
    plt.xlabel('log10(ne)')
    plt.ylabel('Alt (km)')
    plt.title('Model Outputs vs Target Values (Validation Data)')
    plt.legend()
    plt.show()

print("Training complete.")



















