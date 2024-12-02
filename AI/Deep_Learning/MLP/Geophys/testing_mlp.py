# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 15:46:29 2024

@author: Kian Sartipzadeh
"""




import pickle
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader

from storing_dataset import MatchingPairs, StoreDataset

from mlp_utils import from_csv_to_numpy, from_strings_to_datetime, plot_compare
from mlp_model import FFN_Geophys
from sklearn.metrics import r2_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)





# Test data folder names
test_eiscat_folder = "test_eiscat_folder"        # These are the true data
test_geophys_folder = "test_geophys_folder"





# Initializing class for matching pairs
Pairs = MatchingPairs(test_eiscat_folder, test_geophys_folder)


# Returning matching sample pairs
eis, ge = Pairs.find_pairs()


eis = np.abs(eis)
eis[eis < 1e5] = 1e6


# Storing the sample pairs
A = StoreDataset(ge, np.log10(eis))

# Creating DataLoader
batch_size = len(A)
test_loader = DataLoader(A, batch_size=batch_size, shuffle=False)



model = FFN_Geophys()

# criterion = nn.MSELoss()
criterion = nn.HuberLoss()

# Loading the trained network weights
weights_path = 'geophys_FFN_weights.pth'
model.load_state_dict(torch.load(weights_path, weights_only=True))
model.to(device)



model.eval()



predictions = []
true_targets = []


with torch.no_grad():
    for data, targets in test_loader:
        
        data = data.to(device)
        targets = targets.to(device)
        
        
        outputs = model(data)
        
        loss = criterion(outputs, targets)
        print(loss)
        predictions.extend(outputs.cpu().numpy())
        true_targets.extend(targets.cpu().numpy())


model_ne = np.array(predictions)
eiscat_ne = np.array(true_targets)



# Calculate R² score for each feature
r2_scores = [r2_score(10**eiscat_ne[:, i], 10**model_ne[:, i]) for i in range(model_ne.shape[1])]

# Print R² scores for all features
for i, r2 in enumerate(r2_scores):
    print(f"R² score for feature {i + 1}: {r2:.4f}")


# Processing Artist sample files
eis_times = from_csv_to_numpy(test_eiscat_folder)[-1]
eis_time = from_strings_to_datetime(eis_times)  # convert filenames to datetimes




plot_compare(eiscat_ne, model_ne, eis_time)












