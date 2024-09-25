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

from mlp_models import MLP19
from storing_dataset import StoreDataset
from training_mlp import train_model

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)




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

model = MLP19().to(device)
loss_function = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1400, gamma=0.1)



num_epochs = 100

model = train_model(model, train_loader, val_loader, loss_function, optimizer, scheduler, device, num_epochs)


best_model_path = 'best_model.pth'
model.load_state_dict(torch.load(best_model_path))


from sklearn.metrics import r2_score

# Now use the model on the test set
model.eval()
total_test_loss = 0.0
predicted_outputs = []
true_outputs = []
with torch.no_grad():
    for test_inputs, test_targets in test_loader:
        test_inputs, test_targets = test_inputs.to(device), test_targets.to(device)
        
        z = torch.linspace(90, 400, 27).to(device)
        test_outputs = model(test_inputs, z)
        
        # Convert the predictions and targets to numpy for further calculations
        outputs_np = test_outputs.cpu().numpy()
        targets_np = test_targets.cpu().numpy()
        
        predicted_outputs.append(outputs_np)
        true_outputs.append(targets_np)
        
        test_loss = loss_function(test_outputs, test_targets)
        total_test_loss += test_loss.item()

avg_test_loss = total_test_loss / len(test_loader)

# Concatenate the list of predictions and targets
predicted_outputs = np.concatenate(predicted_outputs, axis=0)
true_outputs = np.concatenate(true_outputs, axis=0)

# Flatten the arrays for R2 score calculation
predicted_outputs_flat = predicted_outputs.flatten()
true_outputs_flat = true_outputs.flatten()

# Calculate R² score using sklearn's r2_score function
r2 = r2_score(true_outputs_flat, predicted_outputs_flat)

# Calculate accuracy within a certain percentage tolerance (example: 10% tolerance)
tolerance = 0.02  # 2% tolerance
relative_error = np.abs(predicted_outputs_flat - true_outputs_flat) / np.abs(true_outputs_flat)
accuracy = np.mean(relative_error < tolerance) * 100  # Accuracy in percentage




plt.figure(figsize=(10, 6))
for i in range(len(outputs_np)):
    plt.plot(outputs_np[i], z.cpu().numpy(), label='Predicted', color="C1")
    plt.plot(targets_np[i], z.cpu().numpy(), label='True', color="C0")

    plt.ylabel('Altitude (km)')
    plt.xlabel('Log Electron Density')
    # plt.title('Predicted vs True Electron Density for 10 Samples')
    plt.legend()
    # plt.grid(True)
    plt.show()



# Print out test metrics
print(f'Test Loss: {avg_test_loss:.4f}')
print(f'R² Score: {r2:.4f}')
print(f'Accuracy (within {tolerance * 100}% tolerance): {accuracy:.2f}%')



