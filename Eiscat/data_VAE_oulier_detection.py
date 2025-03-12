# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 13:12:37 2025

@author: Kian Sartipzadeh
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.dates import DateFormatter
from Processing.data_utils import load_dict, inspect_dict, from_array_to_datetime, conv_folder_to_list
from matplotlib.widgets import Cursor

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

r_h = np.array([[ 91.5687711 ],[ 94.57444598],[ 97.57964223],[100.57010953],
       [103.57141624],[106.57728701],[110.08393175],[114.60422289],
       [120.1185208 ],[126.61221111],[134.1346149 ],[142.53945817],
       [152.05174717],[162.57986185],[174.09833378],[186.65837945],
       [200.15192581],[214.62769852],[230.12198695],[246.64398082],
       [264.11728204],[282.62750673],[302.15668686],[322.70723831],
       [344.19596481],[366.64409299],[390.113117  ]])

def plot_losses(train_losses, val_losses):
    """
    Plots the training and validation loss for each epoch.

    Args:
        train_losses (list or array-like): Training loss values per epoch.
        val_losses (list or array-like): Validation loss values per epoch.
    """
    epochs = range(31, len(train_losses) + 1)
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, train_losses[30:], 'bo-', label='Training Loss')
    plt.plot(epochs, val_losses[30:], 'ro-', label='Validation Loss')
    plt.title('Training and Validation Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()


def interactive_latent_space(model, dataset, batch_size=256):
    """
    Creates an interactive 1x2 subplot where the left plot shows the latent space and 
    the right plot shows the corresponding original data sample when a point is clicked.
    
    Args:
        model: Trained VAE model.
        dataset: Dataset to encode (e.g., your full StoreDataset).
        batch_size: Batch size for DataLoader (default: 256).
    """
    # Create a DataLoader to get latent codes and keep original data
    # data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    
    data_loader = dataset
    
    latent_codes = []
    original_data = []

    model.eval()
    with torch.no_grad():
        for data in data_loader:
            data = data.to(device)
            # Get the latent mean (mu) as the latent code
            mu, _ = model.encode(data)
            latent_codes.append(mu.cpu().numpy())
            original_data.append(data.cpu().numpy())
    
    latent_codes = np.concatenate(latent_codes, axis=0)
    original_data = np.concatenate(original_data, axis=0)

    # Create a figure with a 1x2 grid of subplots
    fig, (ax_latent, ax_sample) = plt.subplots(1, 2, figsize=(12, 6))
    
    # Plot the latent space scatter plot on the left subplot
    scatter = ax_latent.scatter(latent_codes[:, 0], latent_codes[:, 1], alpha=0.6, edgecolors='k')
    ax_latent.set_xlabel('Latent Dimension 1')
    ax_latent.set_ylabel('Latent Dimension 2')
    ax_latent.set_title('Latent Space')
    # ax_latent.grid(True)

    # Initialize the right subplot to show the original data sample
    ax_sample.set_title('Original Data Sample')
    ax_sample.set_xlabel('Feature Index')
    ax_sample.set_ylabel('Value')
    ax_sample.grid(True)
    
    # Create a cursor on the latent space plot for better interactivity
    cursor = Cursor(ax_latent, useblit=True, color='red', linewidth=1)

    def on_click(event):
        # Only consider clicks on the latent space subplot
        if event.inaxes != ax_latent:
            return

        click_x, click_y = event.xdata, event.ydata

        # Compute the distance between the click and all latent codes
        distances = np.sqrt((latent_codes[:, 0] - click_x)**2 + (latent_codes[:, 1] - click_y)**2)
        nearest_index = np.argmin(distances)
        
        # Retrieve the corresponding original data sample
        sample = original_data[nearest_index]
        
        # Update the right subplot to display the original data sample
        ax_sample.cla()  # Clear the previous plot
        ax_sample.plot(sample, r_h, marker='o')
        ax_sample.set_title(f'Original Data Sample (Index {nearest_index})')
        ax_sample.set_xlabel('Feature Index')
        ax_sample.set_ylabel('Value')
        ax_sample.set_xscale("log")
        # ax_sample.grid(True)
        fig.canvas.draw_idle()

    # Connect the click event on the figure to our handler
    fig.canvas.mpl_connect('button_press_event', on_click)
    plt.tight_layout()
    plt.show()


def visualize_latent_space(model, dataset, batch_size=256):
    """
    Visualizes the latent space by encoding the dataset and plotting the 2D latent codes.

    Args:
        model: Trained VAE model.
        dataset: Dataset to encode (e.g. your full StoreDataset).
        batch_size: Batch size for DataLoader (default: 256).
    """
    
    # data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    latent_codes = []
    
    data_loader = dataset
    
    model.eval()
    with torch.no_grad():
        for data in data_loader:
            data = data.to(device)
            
            mu, _ = model.encode(data)
            latent_codes.append(mu.cpu().numpy())
    
    # Concatenate all batches into one array
    latent_codes = np.concatenate(latent_codes, axis=0)

    # Create a scatter plot of the latent space
    plt.figure(figsize=(8, 6))
    plt.scatter(latent_codes[:, 0], latent_codes[:, 1], alpha=0.6, edgecolors='k')
    plt.xlabel('Latent Dimension 1')
    plt.ylabel('Latent Dimension 2')
    plt.title('Latent Space Visualization')
    plt.grid(True)
    plt.show()




# Store training pairs with unsqueezed data to add channel dimension.
class StoreDataset(Dataset):
    def __init__(self, target):
        self.target = torch.tensor(target, dtype=torch.float32)
        
    def __len__(self):
        return len(self.target)

    def __getitem__(self, idx):
        return self.target[idx]




class VAE(nn.Module):
    def __init__(self, input_dim=27, latent_dim=2):
        super(VAE, self).__init__()
        
        
        
        # Encoder
        self.enc = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            
            nn.Linear(16, 8),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            
            nn.Linear(8, 4),
            nn.BatchNorm1d(4),
            nn.ReLU()
            )
        
        
        # Latent space
        self.fc_mean = nn.Linear(4, latent_dim)
        self.fc_logvar = nn.Linear(4, latent_dim)
        
        
        # Decoder
        self.dec = nn.Sequential(
            nn.Linear(latent_dim, 4),
            nn.BatchNorm1d(4),
            nn.ReLU(),
            
            nn.Linear(4, 8),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            
            nn.Linear(8, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            
            nn.Linear(16, input_dim),
            )
    
    def encode(self, x):
        h = self.enc(x)
        mu = self.fc_mean(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        return self.dec(z)
    
    def forward(self, x):
        
        mu, logvar = self.encode(x)
        
        z = self.reparameterize(mu, logvar)
        
        recon_x = self.decode(z)
        return recon_x, mu, logvar

# Define the VAE loss function
def loss_function(recon_x, x, mu, logvar):
    MSE = F.mse_loss(recon_x, x, reduction='mean')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return MSE + KLD

def preprocess_data(X):
    # X = np.log10(X)
    X = (X - np.min(X)) / (np.max(X) - np.min(X))  # Normalize to [0,1]
    return X





radar_folder = "Processing/EISCAT_Healthy"
rad = conv_folder_to_list(radar_folder)


rad = preprocess_data(rad)



A = StoreDataset(rad)



# Split the dataset into train and validation sets
train_size = int(0.8 * len(A))
val_size = len(A) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(A, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=val_size, shuffle=True)


num_epochs = 400
model = VAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)


# Initialize variables to track the best model
best_val_loss = float('inf')
best_model_weights = model.state_dict()



train_losses = []
val_losses = []
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for i, samples in enumerate(train_loader):
        x = samples.to(device)  # Move data to GPU
        optimizer.zero_grad()
        recon_x, mu, logvar = model(x)  # Use GPU tensor as input
        loss = loss_function(recon_x, x, mu, logvar)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    avg_train_loss = running_loss / len(train_loader)
    train_losses.append(avg_train_loss)
    
    # Validation loop (already correct)
    model.eval()
    val_loss = 0.0
    val_outputs = []
    val_targets = []
    with torch.no_grad():
        for samples in val_loader:
            x = samples.to(device)
            recon_x, mu, logvar = model(x)
            val_outputs.append(recon_x.cpu().numpy())
            val_targets.append(recon_x.cpu().numpy())
            loss = loss_function(recon_x, x, mu, logvar)
            val_loss += loss.item()
    
    avg_val_loss = val_loss / len(val_loader)
    val_losses.append(avg_val_loss)
    
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        best_model_weights = model.state_dict()
        best_val_outputs = val_outputs
        best_val_targets = val_targets
    
    print(f"Epoch [{epoch + 1}/{num_epochs}], Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}")



# Save the best model to a file
torch.save(best_model_weights, 'weights/VAE_l16l8l4l2_epoch400_lr0001_trBS256.pth')
plot_losses(train_losses, val_losses)





# Example usage:
# Visualize the latent space using the entire dataset (A is your StoreDataset)
# visualize_latent_space(model, val_loader)
interactive_latent_space(model, val_loader)









