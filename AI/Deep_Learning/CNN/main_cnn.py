# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 12:58:31 2024

@author: Kian Sartipzadeh
"""

import os
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
ionogram_folder = "Ionogram_sampled_images"



Pairs = MatchingPairs(ionogram_folder, radar_folder)

rad, ion = Pairs.find_pairs()


A = StoreDataset(ion, rad, transforms.Compose([transforms.ToTensor()]))


data_loader = DataLoader(A, batch_size=100, shuffle=True)


model = CNN()
criterion = nn.MSELoss()  # Assuming this is a classification problem
optimizer = optim.Adam(model.parameters(), lr=0.1)  # Use Adam optimizer



# Function to count the number of parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Print the number of parameters
print(f"Total number of trainable parameters: {count_parameters(model)}")























