# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 13:12:37 2025

@author: Kian Sartipzadeh
"""


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.dates import DateFormatter
from data_utils import load_dict, inspect_dict, from_array_to_datetime
from matplotlib.widgets import Cursor


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


# Store training pairs
class StoreDataset(Dataset):
    def __init__(self, data, target):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.target = torch.tensor(target, dtype=torch.float32)
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        
        return self.data[idx], self.target[idx]





# VariationalAutoEncoder
class VariationalAutoEncoder(nn.Module):
    def __init__(self):
        super(VariationalAutoEncoder, self).__init__()
        
        # Encoder
        self.Encoder = nn.Sequential(
            nn.Linear(27, 64),
            nn.Dropout(0.2),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2),
            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(0.2)
        )
        
        self.mean = nn.Linear(16, 3)
        self.logvar = nn.Linear(16, 3)
        
        # Decoder
        self.Decoder = nn.Sequential(
            nn.Linear(3, 16),
            # nn.Dropout(0.2),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(0.2),
            nn.Linear(16, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2),
            nn.Linear(32, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 27),
            nn.Sigmoid()  # Constrain output to [0,1]
        )
    
    def Encode(self, x):
        x = self.Encoder(x)
        
        x_mean = self.mean(x)
        x_logvar = self.logvar(x)
        return x_mean, x_logvar
        
    
    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        
        par = mean + std*eps
        return par
    
    
    def Decode(self, x):        
        x_par = self.Decoder(x) 
        return x_par
    
    def forward(self, x):
        mean, logvar = self.Encode(x)
        z = self.reparameterize(mean, logvar)
        z_recon = self.Decode(z)
        return z_recon, z, mean, logvar



def loss_function(recon_x, x, mean, logvar, beta=0.0001):
    
    # MSE or BCE Reconstruction Loss
    recon_loss = nn.functional.mse_loss(recon_x, x, reduction="sum")
    
    # KL Divergence Loss
    KLD = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
    return recon_loss + beta*KLD






def ravel_dict(data_dict):
    return np.concatenate([data_dict[date]['r_param'] for date in data_dict], axis=1)

def preprocess_data(X):
    X = np.where(X > 0, X, np.nan)  # Replace negatives/zeros with NaN
    X = np.log10(X)
    X = np.nan_to_num(X, nan=0.0)  # Replace NaNs with 0
    X[X < 6] == 8
    X = (X - np.min(X)) / (np.max(X) - np.min(X))  # Normalize to [0,1]
    return X


x = load_dict("X_avg_all_VAE")
x = ravel_dict(x).T
X = preprocess_data(x)

A = StoreDataset(X, X)




# Split the dataset into train and validation sets
train_size = int(0.8 * len(A))
print(train_size)
val_size = len(A) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(A, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=256*4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=val_size, shuffle=True)


num_epochs = 200
model = VariationalAutoEncoder().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.01)  # Use Adam optimizer

scheduler1 = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)


# Function to count the number of parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Print the number of parameters
print(f"Total number of trainable parameters: {count_parameters(model)}")



# Initialize variables to track the best model
best_val_loss = float('inf')
best_model_weights = model.state_dict()



# Training loop
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0

    for data, _ in train_loader:  # We ignore targets since it's an autoencoder
        data = data.to(device)

        # Forward pass
        recon_x, z, mu, logvar = model(data)

        # Compute loss
        loss = loss_function(recon_x, data, mu, logvar)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
    
    
    # Step the learning rate scheduler
    scheduler1.step()
    
    avg_train_loss = train_loss / len(train_dataset)

    # Validation loop
    model.eval()
    val_loss = 0.0
    bottleneck_representations = []

    with torch.no_grad():
        for data, _ in val_loader:
            data = data.to(device)

            # Forward pass
            recon_x, z, mu, logvar = model(data)

            # Compute loss
            loss = loss_function(recon_x, data, mu, logvar)
            val_loss += loss.item()

            # Store bottleneck representations for visualization
            bottleneck_representations.append(z.cpu().numpy())

    avg_val_loss = val_loss / len(val_dataset)

    # Save the best model
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        best_model_weights = model.state_dict()
        best_bottleneck = np.concatenate(bottleneck_representations, axis=0)

    print(f"Epoch [{epoch+1}/{num_epochs}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")


print("Training complete. Best model saved.")


with torch.no_grad():
    _, _, mu, logvar = model(torch.tensor(X[:100], dtype=torch.float32).to(device))

print("Mean (mu):", mu.cpu().numpy().mean(axis=0))
print("Variance (exp(logvar)):", logvar.cpu().exp().numpy().mean(axis=0))



# Visualizing the latent space
fig1, ax1 = plt.subplots(figsize=(10, 8))
sc1 = ax1.scatter(best_bottleneck[:, 0], best_bottleneck[:, 1], c='blue', alpha=0.6)
ax1.set_title('Latent Space Visualization')
ax1.set_xlabel('Latent Dimension 1')
ax1.set_ylabel('Latent Dimension 2')
# ax.grid(True)

# Visualizing the latent space
fig2, ax2 = plt.subplots(figsize=(10, 8))
sc2 = ax2.scatter(best_bottleneck[:, 0], best_bottleneck[:, 2], c='blue', alpha=0.6)
ax2.set_title('Latent Space Visualization')
ax2.set_xlabel('Latent Dimension 1')
ax2.set_ylabel('Latent Dimension 2')
# ax.grid(True)

# Visualizing the latent space
fig3, ax3 = plt.subplots(figsize=(10, 8))
sc3 = ax3.scatter(best_bottleneck[:, 1], best_bottleneck[:, 2], c='blue', alpha=0.6)
ax3.set_title('Latent Space Visualization')
ax3.set_xlabel('Latent Dimension 1')
ax3.set_ylabel('Latent Dimension 2')
# ax.grid(True)


# Create a new figure for displaying the original samples
fig_original, ax_original = plt.subplots(figsize=(10, 8))
ax_original.set_title('Original Sample')
ax_original.set_xlabel('Feature Index')
ax_original.set_ylabel('Value')


r_h = np.array([[ 91.5687711 ],[ 94.57444598],[ 97.57964223],[100.57010953],
       [103.57141624],[106.57728701],[110.08393175],[114.60422289],
       [120.1185208 ],[126.61221111],[134.1346149 ],[142.53945817],
       [152.05174717],[162.57986185],[174.09833378],[186.65837945],
       [200.15192581],[214.62769852],[230.12198695],[246.64398082],
       [264.11728204],[282.62750673],[302.15668686],[322.70723831],
       [344.19596481],[366.64409299],[390.113117  ]])



# Function to update the original sample plot
def update_original_sample(event):
    if event.inaxes in [ax1, ax2, ax3]:
        cont, ind = sc1.contains(event)
        cont, ind = sc2.contains(event)
        cont, ind = sc3.contains(event)
        if cont:
            idx = ind['ind'][0]
            original_sample = X[idx]
            ax_original.clear()
            ax_original.plot(original_sample, r_h.flatten(), label=f'Sample {idx}')
            ax_original.legend()
            ax_original.set_title(f'Original Sample {idx}')
            ax_original.set_xlabel('Feature Index')
            ax_original.set_ylabel('Value')
            fig_original.canvas.draw_idle()

# Connect the cursor and the click event handler
Cursor(ax1, useblit=True, color='red', linewidth=2)
Cursor(ax2, useblit=True, color='red', linewidth=2)
Cursor(ax3, useblit=True, color='red', linewidth=2)
fig1.canvas.mpl_connect('button_press_event', update_original_sample)
fig2.canvas.mpl_connect('button_press_event', update_original_sample)
fig3.canvas.mpl_connect('button_press_event', update_original_sample)

plt.show()


# # Training loop
# for epoch in range(num_epochs):
#     model.train()  # Set model to training mode
#     running_loss = 0.0
#     for i, (data, target) in enumerate(train_loader):
#         data, target = data.to(device), target.to(device)

#         # Zero the gradients
#         optimizer.zero_grad()
        
#         output, _ = model(data)

#         loss = criterion(output, target)
#         loss.backward()
#         running_loss += loss.item()
        
#         optimizer.step()
        
#     # print(f"Epoch [{epoch + 1}/{num_epochs}], Training Loss: {running_loss / len(train_loader):.4f}")
#     # Validation loop
#     model.eval()  # Set model to evaluation mode
#     val_loss = 0.0
#     val_outputs = []
#     val_targets = []
#     with torch.no_grad():
#         for data, target in val_loader:
#             data, target = data.to(device), target.to(device)
            
#             # Forward pass
#             output, _  = model(data)
            
#             # Store outputs and targets for plotting
#             val_outputs.append(output.cpu().numpy())
#             val_targets.append(target.cpu().numpy())
            
#             # Compute loss
#             loss = criterion(output, target)
#             val_loss += loss.item()
            
#     # Calculate average validation loss
#     avg_val_loss = val_loss / len(val_loader)

#     # Save the best model weights
#     if avg_val_loss < best_val_loss:
#         best_val_loss = avg_val_loss
#         best_model_weights = model.state_dict()
#         best_val_outputs = val_outputs
#         best_val_targets = val_targets
    
#     # Print epoch summary
#     print(f"Epoch [{epoch + 1}/{num_epochs}], Training Loss: {running_loss / len(train_loader):.4f}, Validation Loss: {avg_val_loss:.4f}")


# import torch
# import matplotlib.pyplot as plt








# # Load the best model weights
# model.load_state_dict(best_model_weights)
# model.eval()

# # Collect bottleneck representations
# bottleneck_representations = []
# with torch.no_grad():
#     for data, target in val_loader:
#         data = data.to(device)
#         _, z_bottleneck = model(data)  # Get the bottleneck representation
#         bottleneck_representations.append(z_bottleneck.cpu().numpy())




# # Convert to numpy array
# bottleneck_representations = np.concatenate(bottleneck_representations, axis=0)

# # Scatter plot of the 2D bottleneck space
# # plt.figure(figsize=(8, 6))
# plt.scatter(bottleneck_representations[:, 0], bottleneck_representations[:, 1], alpha=0.5)
# # plt.xlabel("Bottleneck Dimension 1")
# # plt.ylabel("Bottleneck Dimension 2")
# # plt.title("2D Bottleneck Space Visualization")
# # plt.grid(True)
# plt.show()


# # Training loop
# for epoch in range(num_epochs):
#     model.train()  # Set model to training mode
#     running_loss = 0.0
#     for i, (data, target) in enumerate(train_loader):
#         data, target = data.to(device), target.to(device)

#         # Zero the gradients
#         optimizer.zero_grad()
        
#         recon_batch, mean, logvar, _ = model(data)

#         loss = loss_function(recon_batch, target, mean, logvar)
#         loss.backward()
#         running_loss += loss.item()
        
#         optimizer.step()
        
#     print(f"Epoch [{epoch + 1}/{num_epochs}], Training Loss: {running_loss / len(train_loader):.4f}")
    # # Validation loop
    # model.eval()  # Set model to evaluation mode
    # val_loss = 0.0
    # val_outputs = []
    # val_targets = []
    # with torch.no_grad():
    #     for data, target in val_loader:
    #         data, target = data.to(device), target.to(device)
            
    #         # Forward pass
    #         val_batch, _  = model(data)
            
    #         # Store outputs and targets for plotting
    #         val_outputs.append(val_batch.cpu().numpy())
    #         val_targets.append(target.cpu().numpy())
            
    #         # Compute loss
    #         loss = loss_function(val_batch, target, mean, logvar)
    #         val_loss += loss.item()
            
    # # Calculate average validation loss
    # avg_val_loss = val_loss / len(val_loader)

    # # Save the best model weights
    # if avg_val_loss < best_val_loss:
    #     best_val_loss = avg_val_loss
    #     best_model_weights = model.state_dict()
    #     best_val_outputs = val_outputs
    #     best_val_targets = val_targets
    
    # Print epoch summary
    # print(f"Epoch [{epoch + 1}/{num_epochs}], Training Loss: {running_loss / len(train_loader):.4f}, Validation Loss: {avg_val_loss:.4f}")


























