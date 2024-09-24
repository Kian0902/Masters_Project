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
                                    nn.BatchNorm1d(124),
                                    nn.ReLU(),
                                    nn.Linear(124, 64),
                                    nn.BatchNorm1d(64),
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

# Initialize model, loss function, optimizer, and scheduler
in_dim = X_train.shape[1]

model = MLP(in_dim, out_dim=8).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.1)


# # Print the initialized weights
# print("Initialized model weights:")
# for name, param in model.named_parameters():
#     if param.requires_grad:
#         print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]}")  # Printing only the first two weights for brevity



# To track the best validation loss
best_val_loss = float('inf')
best_model_path = 'best_model.pth'

num_epochs = 3000

for epoch in range(num_epochs):
    model.train()
    total_train_loss = 0.0
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        
        z = torch.linspace(90, 400, 27).to(device)
        
        outputs = model(inputs, z)
        loss = criterion(outputs, targets)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_train_loss += loss.item()
        
    avg_train_loss = total_train_loss / len(train_loader)
    
    # Validation loop
    model.eval()
    total_val_loss = 0.0
    with torch.no_grad():
        for val_inputs, val_targets in val_loader:
            val_inputs, val_targets = val_inputs.to(device), val_targets.to(device)
            
            z = torch.linspace(90, 400, 27).to(device)
            val_outputs = model(val_inputs, z)
            val_loss = criterion(val_outputs, val_targets)
            total_val_loss += val_loss.item()
    
    avg_val_loss = total_val_loss / len(val_loader)
    
    # Step the learning rate scheduler
    scheduler.step()
    
    # Check if current validation loss is the best we've seen so far
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), best_model_path)  # Save the model with best validation loss
        
    # Print training and validation loss for the epoch
    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}')
    
# After training, load the best model weights before testing
model.load_state_dict(torch.load(best_model_path))


# Now use the model on the test set
model.eval()
total_test_loss = 0.0
with torch.no_grad():
    for test_inputs, test_targets in test_loader:
        test_inputs, test_targets = test_inputs.to(device), test_targets.to(device)
        
        z = torch.linspace(90, 400, 27).to(device)
        test_outputs = model(test_inputs, z)
        
        outputs_np = outputs.cpu().numpy()
        targets_np = targets.cpu().numpy()
        
        test_loss = criterion(test_outputs, test_targets)
        total_test_loss += test_loss.item()

avg_test_loss = total_test_loss / len(test_loader)




plt.figure(figsize=(10, 6))
for i in range(len(outputs_np)):
    plt.plot(outputs_np[i], z.cpu().numpy(), label='Predicted', color="C1")
    plt.plot(targets_np[i], z.cpu().numpy(), label='True', color="C0")

    plt.xlabel('Altitude (km)')
    plt.ylabel('Log Electron Density')
    # plt.title('Predicted vs True Electron Density for 10 Samples')
    plt.legend()
    # plt.grid(True)
    plt.show()



print(f'Test Loss: {avg_test_loss:.4f}')





