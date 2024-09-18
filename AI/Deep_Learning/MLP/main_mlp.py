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

# Dataset class to store data
class StoreDataset(Dataset):
    def __init__(self, data, targets):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

# Combined MLP and Chapman model class
class ElectronDensityModel(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim=8):
        super(ElectronDensityModel, self).__init__()
        
        # MLP layers
        self.FC1  = nn.Linear(in_dim, hidden_dim)
        self.FC2  = nn.Linear(hidden_dim, hidden_dim)
        self.FC3  = nn.Linear(hidden_dim, out_dim)
        
        # Activation
        self.relu = nn.ReLU()
    
    def double_chapman(self, z, mlp_output):
        # Slice the mlp_output tensor to get the 8 separate values
        zE_peak = mlp_output[:, 0]
        zF_peak = mlp_output[:, 1]
        nE_peak = mlp_output[:, 2]
        nF_peak = mlp_output[:, 3]
        HE_below = mlp_output[:, 4]
        HE_above = mlp_output[:, 5]
        HF_below = mlp_output[:, 6]
        HF_above = mlp_output[:, 7]
        
        print(HF_below)
        
        
        # Appending scale heights corresponding to altitude
        HE = torch.where(z < zE_peak.unsqueeze(1), HE_below.unsqueeze(1), HE_above.unsqueeze(1))
        HF = torch.where(z < zF_peak.unsqueeze(1), HF_below.unsqueeze(1), HF_above.unsqueeze(1))
        
        # Defining Chapman electron density profile for both regions
        neE = nE_peak.unsqueeze(1) * torch.exp(1 - ((z - zE_peak.unsqueeze(1)) / HE) - torch.exp(-((z - zE_peak.unsqueeze(1)) / HE)))
        neF = nF_peak.unsqueeze(1) * torch.exp(1 - ((z - zF_peak.unsqueeze(1)) / HF )- torch.exp(-((z - zF_peak.unsqueeze(1)) / HF)))
        
        return neE + neF
    
    def forward(self, x, z):
        # MLP forward pass
        x = self.FC1(x)
        x = self.relu(x)
        x = self.FC2(x)
        x = self.relu(x)
        mlp_output = self.FC3(x)
        
        
        # Compute the electron density profile using the MLP outputs
        electron_density = self.double_chapman(z, mlp_output)
        
        return electron_density




data_sp19 = np.load('sp19_data.npy')
data_eiscat = np.load('eiscat_data.npy')

X_train, X_test, y_train, y_test = train_test_split(data_sp19, data_eiscat, train_size=0.8, shuffle=False)

y_train[y_train < 1] = 10**5
y_test[y_test < 1] = 10**5

# Apply log10 to all the datasets
y_train = np.log10(y_train)
y_test = np.log10(y_test)




train_dataset = StoreDataset(X_train, y_train)
test_dataset = StoreDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=100, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)

# Initialize model, loss function, and optimizer
in_dim = X_train.shape[1]
hidden_dim = 64


model = ElectronDensityModel(in_dim, hidden_dim).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Example training loop
num_epochs = 1  # You can adjust the number of epochs

for epoch in range(num_epochs):
    model.train()
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Forward pass (assuming z is provided in the data or externally)
        z = torch.linspace(90, 400, 27).to(device)  # Example altitude values for electron density
        
    
        outputs = model(inputs, z)
        # outputs = torch.clamp(outputs, min=1e-8)
        # Compute the loss
        loss = criterion(torch.log10(outputs + 1e-8), targets)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# You can similarly implement evaluation or test loop
















# You can similarly implement evaluation or test loop


# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import Dataset, DataLoader

# import numpy as np
# import matplotlib.pyplot as plt

# # from sklearn.datasets import make_blobs
# from sklearn.model_selection import train_test_split
# # from sklearn.preprocessing import StandardScaler

# # Check if GPU is available
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(device)





# # class to store data
# class StoreDataset(Dataset):
    
#     def __init__(self, data, targets):
#         self.data = torch.tensor(data, dtype=torch.float32)
#         self.targets = torch.tensor(targets, dtype=torch.float32)
        
#     def __len__(self):
#         return len(self.data)
        
#     def __getitem__(self, idx):
#         return self.data[idx], self.targets[idx]



# class MLP(nn.Module):
#     def __init__(self, in_dim, hidden_dim, out_dim=8):
#         super(MLP, self).__init__()
        
#         # Layers
#         self.FC1  = nn.Linear(in_dim, hidden_dim)
#         self.FC2  = nn.Linear(hidden_dim, out_dim)
        
#         # Activation
#         self.relu = nn.ReLU()
        
#     def forward(self, x):
#         x = self.FC1(x)
#         x = self.relu(x)
#         x = self.FC2(x)
#         x = self.relu(x)
#         return x
    





# data_sp19 = np.load('sp19_data.npy')
# data_eiscat = np.load('eiscat_data.npy')


# X_train, X_test, y_train, y_test = train_test_split(data_sp19, data_eiscat, train_size=0.8, shuffle=False)


# train_dataset = StoreDataset(X_train, y_train)
# test_dataset = StoreDataset(X_test, y_test)




# # Create DataLoader
# train_loader = DataLoader(train_dataset, batch_size=100, shuffle=True)
# test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)
















# # class to store data
# class BlobDataset(Dataset):
    
#     def __init__(self, data, targets):
        
#         # Converting to tensors
#         self.data = torch.tensor(data, dtype=torch.float32)
#         self.targets = torch.tensor(targets, dtype=torch.long)
        
#     def __len__(self):
#         return len(self.data)
        
#     def __getitem__(self, idx):
#         return self.data[idx], self.targets[idx]



# def plot_decision_boundary(model, X, y, device):
#     # Set the model to evaluation mode
#     model.eval()

#     # Define the grid over which to evaluate the model
#     x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
#     y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
#     xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
#                          np.arange(y_min, y_max, 0.01))

#     # Flatten the grid to pass it through the model
#     grid = np.c_[xx.ravel(), yy.ravel()]
#     grid_tensor = torch.tensor(grid, dtype=torch.float32).to(device)

#     # Get model predictions for the grid
#     with torch.no_grad():
#         Z = model(grid_tensor)
#         Z = torch.argmax(Z, dim=1)
#         Z = Z.cpu().numpy()

#     # Reshape the predictions back into the grid shape
#     Z = Z.reshape(xx.shape)

#     # Plot the decision boundary and the data points
#     plt.contourf(xx, yy, Z, alpha=0.5, cmap=plt.cm.RdYlBu)
#     plt.scatter(X[:, 0], X[:, 1], c=y, s=40, edgecolor='k', cmap=plt.cm.RdYlBu)
#     plt.xlim(xx.min(), xx.max())
#     plt.ylim(yy.min(), yy.max())
#     plt.title("Decision Boundary")
#     plt.show()






# # data
# X, y = make_blobs(n_samples=1000, centers=2, n_features=2)


# # Standardize
# scaler = StandardScaler()
# X = scaler.fit_transform(X)


# # Split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=1)


# # plot
# plt.figure()
# plt.title("Blobs dataset")
# plt.scatter(X_train[:,0][y_train==0], X_train[:,1][y_train==0], color="C0", label="Class 1")
# plt.scatter(X_train[:,0][y_train==1], X_train[:,1][y_train==1], color="C1", label="Class 2")
# plt.xlabel("X1")
# plt.ylabel("X2")
# plt.legend()
# plt.show()




# # Storing datasets
# train_dataset = BlobDataset(X_train, y_train)
# test_dataset = BlobDataset(X_test, y_test)


# # Create DataLoader
# train_loader = DataLoader(train_dataset, batch_size=100, shuffle=True)
# test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)




# # Initialize network
# model = MLP(2, 10, 2).to(device)
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.01)



# epochs = 100

# for epoch in range(epochs):
#     model.train()
#     running_loss = 0.0
#     for inputs, labels in train_loader:
        
#         inputs, labels = inputs.to(device), labels.to(device)
        
#         optimizer.zero_grad()
#         outputs = model(inputs)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()
#         running_loss += loss.item()
    
    
#     print(f'Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}')



# # Test the model
# model.eval()
# correct = 0
# total = 0




# with torch.no_grad():
#     for inputs, labels in test_loader:
#         # Move inputs and labels to the device
#         inputs, labels = inputs.to(device), labels.to(device)
        
#         outputs = model(inputs)
#         _, predicted = torch.max(outputs.data, 1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()

# print(f'Accuracy of the model on the test set: {100 * correct / total:.2f}%')






# plot_decision_boundary(model, X, y, device)


