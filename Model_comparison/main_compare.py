# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 16:50:34 2024

@author: Kian Sartipzadeh
"""



import pickle
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
from storing_dataset import Matching3Pairs, Store3Dataset


from eval_utils import from_csv_to_numpy, from_strings_to_datetime, plot_compare
from hnn_model import CombinedNetwork


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)







# Test data folder names
test_ionogram_folder = "testing_data/test_ionogram_folder - Copy"
test_artist_folder = "testing_data/test_artist_folder - Copy"
test_radar_folder = "testing_data/test_radar_folder - Copy"        # These are the true data
test_sp19_folder = "testing_data/test_sp19_folder - Copy"





# Initializing class for matching pairs
Pairs = Matching3Pairs(test_ionogram_folder, test_radar_folder, test_sp19_folder)


# Returning matching sample pairs
rad, ion, sp, radar_times = Pairs.find_pairs(return_date=True)
r_times = from_strings_to_datetime(radar_times)


# Processing Artist sample files
art, artist_times = from_csv_to_numpy(test_artist_folder)
art_time = from_strings_to_datetime(artist_times)  # convert filenames to datetimes


# art_h = np.arange(80, 485, 5)
# plt.pcolormesh(art_time, art_h, art.T)
# plt.show()




rad = np.abs(rad)
rad[rad < 1e5] = 1e6


# Storing the sample pairs
A = Store3Dataset(ion, sp, np.log10(rad), transforms.Compose([transforms.ToTensor()]))


# Creating DataLoader
batch_size = len(A)
test_loader = DataLoader(A, batch_size=batch_size, shuffle=False)



model = CombinedNetwork()
criterion = nn.HuberLoss()

# Loading the trained network weights
weights_path = 'best_model_weights.pth'
model.load_state_dict(torch.load(weights_path, weights_only=True))
model.to(device)



model.eval()



predictions = []
true_targets = []


with torch.no_grad():
    for data1, data2, targets in test_loader:
        
        data1 = data1.to(device)
        data2 = data2.to(device)
        targets = targets.to(device)
        
        
        outputs = model(data1, data2)
        
        loss = criterion(outputs, targets)
        print(loss)
        predictions.extend(outputs.cpu().numpy())
        true_targets.extend(targets.cpu().numpy())


model_ne = np.array(predictions)
eiscat_ne = np.array(true_targets)




plot_compare(eiscat_ne, model_ne, np.log10(art), r_times, art_time)











