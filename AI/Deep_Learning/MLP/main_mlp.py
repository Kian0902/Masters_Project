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
        self.FC1  = nn.Linear(in_dim, 128)
        self.FC2  = nn.Linear(128, 128)
        self.FC3  = nn.Linear(128, 64)
        self.FC4  = nn.Linear(64, out_dim)
        
        
        # Activation
        self.relu = nn.ReLU()
        self.softplus = nn.Softplus(beta=0.2, threshold=1)
        
    
    def double_chapman(self, x, z):
        zE_peak, zF_peak, nE_peak, nF_peak, HE_below, HF_below, HE_above, HF_above = x.split(1, dim=1)
        
        # print(f"zE_peak:  min={torch.min(zE_peak)}, max={torch.max(zE_peak)}")
        # print(f"zF_peak:  min={torch.min(zF_peak)}, max={torch.max(zF_peak)}")
        # print(f"nE_peak:  min={torch.min(nE_peak)}, max={torch.max(nE_peak)}")
        # print(f"nF_peak:  min={torch.min(nF_peak)}, max={torch.max(nF_peak)}")
        # print(f"HE_below:  min={torch.min(HE_below)}, max={torch.max(HE_below)}")
        # print(f"HF_below:  min={torch.min(HF_below)}, max={torch.max(HF_below)}")
        # print(f"nE_peak:  min={torch.min(HE_above)}, max={torch.max(HE_above)}")
        # print(f"nF_peak:  min={torch.min(HF_above)}, max={torch.max(HF_above)}")
        # print("\n")
        # print(zE_peak.shape)
        # print(torch.min(zE_peak))
        
        
        # Adding epsilon to avoid division by zero
        HE = torch.where(z < zE_peak, HE_below, HE_above)
        HF = torch.where(z < zF_peak, HF_below, HF_above)
    
        # Clamping to avoid overflow in torch.exp
        neE = nE_peak * torch.exp(1 - ((z - zE_peak) / HE) - torch.exp(-((z - zE_peak) / HE)))
        neF = nF_peak * torch.exp(1 - ((z - zF_peak) / HF) - torch.exp(-((z - zF_peak) / HF)))
        # Calculate min and max for neE and neF
        neE_min, neE_max = torch.min(neE), torch.max(neE)
        neF_min, neF_max = torch.min(neF), torch.max(neF)
        
        # print(neE_min, neE_max)
        # print(neF_min, neF_max)
        return neE + neF
        
    
    
    def forward(self, x, z):

        x = self.FC1(x)
        x = self.relu(x)
        x = self.FC2(x)
        x = self.relu(x)
        x = self.FC3(x)
        x = self.relu(x)
        x = self.FC4(x)
        x = self.softplus(x)
        
        # Clamp parameters within specified ranges
        # min_values = torch.tensor([80, 200, 3, 3, 3, 3, 3, 3], device=x.device)  # Set appropriate lower bounds for each parameter
        # max_values = torch.tensor([200, 400, 15, 15, 100, 100, 100, 100], device=x.device)  # Set appropriate upper bounds for each parameter
    
        # x = torch.clamp(x, min=min_values, max=max_values)
            
        batch_siz1e = x.size(0)
        # z = z.unsqueeze(0).expand(batch_size, -1).to(device)
        
        # chapman_output = self.double_chapman(x, z)
        # print(torch.min(chapman_output, 0))
        
        return x
    




data_sp19 = np.load('sp19_data.npy')
data_eiscat = np.load('eiscat_data.npy')




X_train, X_test, y_train, y_test = train_test_split(data_sp19, data_eiscat, train_size=0.8, shuffle=True)

y_train[y_train < 10**5] = 10**5
y_test[y_test < 10**5] = 10**5


# Apply log10 to all the datasets
y_train = np.log10(y_train)
y_test = np.log10(y_test)








train_dataset = StoreDataset(X_train, y_train)
test_dataset = StoreDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=100, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=30, shuffle=True)

# Initialize model, loss function, and optimizer
in_dim = X_train.shape[1]
hidden_dim = 124


model = MLP(in_dim, out_dim=27).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)






# Example training loop
num_epochs = 2000  # You can adjust the number of epochs

for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    # i=0
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        
        z = torch.linspace(90, 400, 27).to(device)
        
        # print(i)
        outputs = model(inputs, z)
        # i+=1
        # outputs = torch.log10(outputs)
        # print(outputs)
        loss = criterion(outputs, targets)
        
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        
        optimizer.step()
        
        total_loss += loss.item()
        
    # print(outputs)
    avg_train_loss = total_loss / len(train_loader)
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_train_loss:.4f}')




# Set the model to evaluation mode
model.eval()

# Disable gradient computation for evaluation
with torch.no_grad():
    for inputs, targets in test_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Define the range of z values
        z = torch.linspace(90, 400, 27).to(device)
        
        # Get model predictions
        outputs = model(inputs, z)
        
        # Convert the outputs and targets to NumPy arrays for plotting
        outputs_np = outputs.cpu().numpy()
        targets_np = targets.cpu().numpy()
        
        print(f'Predictions for 10 test samples (numpy):\n{outputs_np}')
        print(f'True targets for 10 test samples (numpy):\n{targets_np}')
        break  # Exit after processing the first 10 samples

# Now you can use matplotlib to plot the outputs and targets
# Example plot:
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
for i in range(len(outputs_np)):
    plt.plot(outputs_np[i], z.cpu().numpy(), label='Predicted', color="C0")
    plt.plot(targets_np[i], z.cpu().numpy(), label='True', color="C1")

    plt.xlabel('Altitude (km)')
    plt.ylabel('Log Electron Density')
    # plt.title('Predicted vs True Electron Density for 10 Samples')
    plt.legend()
    # plt.grid(True)
    plt.show()


















