# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 17:40:06 2024

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
from storing_dataset import Store3Dataset, Matching3Pairs
from cnn_models import CombinedNetwork


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)




radar_folder = "EISCAT_samples"
ionogram_folder = "Good_Ionograms_Images"
sp19_folder = "SP19_samples"


Pairs = Matching3Pairs(ionogram_folder, radar_folder, sp19_folder)

rad, ion, sp = Pairs.find_pairs()


rad = np.abs(rad)
rad[rad < 1e5] = 1e6







A = Store3Dataset(ion, sp, np.log10(rad), transforms.Compose([transforms.ToTensor()]))

print("Data stored")



# Split the dataset into train and validation sets

train_size = int(0.8 * len(A))
val_size = len(A) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(A, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=500, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=val_size, shuffle=True)


num_epochs = 200
model = CombinedNetwork().to(device)
# criterion = CustomLoss()
criterion = nn.HuberLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)  # Use Adam optimizer

scheduler1 = torch.optim.lr_scheduler.StepLR(optimizer, step_size=150, gamma=0.1)
#scheduler2 = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)


# Function to count the number of parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Print the number of parameters
print(f"Total number of trainable parameters: {count_parameters(model)}")



# Initialize variables to track the best model
best_val_loss = float('inf')
best_model_weights = model.state_dict()
last_model_weights = model.state_dict()

# Training loop
for epoch in range(num_epochs):
    model.train()  # Set model to training mode
    running_loss = 0.0
    for i, (images, geophys, targets) in enumerate(train_loader):
        images, geophys, targets = images.to(device), geophys.to(device), targets.to(device)

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images, geophys)

        # Compute loss
        loss = criterion(outputs, targets)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Update running loss
        running_loss += loss.item()

    
    # Save the last epoch model weights
    last_model_weights = model.state_dict()
    
    # Step the learning rate scheduler
    scheduler1.step()
    #scheduler2.step()
    
    # Validation loop
    model.eval()  # Set model to evaluation mode
    val_loss = 0.0
    val_outputs = []
    val_targets = []
    with torch.no_grad():
        for images, geophys, targets in val_loader:
            images, geophys, targets = images.to(device), geophys.to(device), targets.to(device)

            # Forward pass
            outputs = model(images, geophys)

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

# Save the best model to a file
torch.save(best_model_weights, '/kaggle/working/best_model_weights.pth')
torch.save(last_model_weights, '/kaggle/working/last_model_weights.pth')
print("Weights from best model saved to 'best_model_weights.pth'.")
print("Weights from last epoch saved to 'last_model_weights.pth'.")

print("Training complete.")

























