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
from storing_dataset import IonoEisDataset
from cnn_models import CNN




def list_csv_files(folder_path):
    return [f for f in os.listdir(folder_path) if f.endswith('.csv')]


def list_png_files(folder_path):
    return [f for f in os.listdir(folder_path) if f.endswith('.png')]


def get_filename_without_extension(filename):
    return os.path.splitext(filename)[0]




radar_folder = "EISCAT_samples"
ionogram_folder = "Ionogram_sampled_images"

radar_names = list_csv_files(radar_folder)
ionogram_names = list_png_files(ionogram_folder)

# Extract base filenames without ".csv"
radar_filenames = set(get_filename_without_extension(f) for f in radar_names)
ionogram_filenames = set(get_filename_without_extension(f) for f in ionogram_names)

# Find the matching filenames
matching_filenames = sorted(list(ionogram_filenames.intersection(radar_filenames)))
transform = transforms.Compose([transforms.ToTensor()])
A = IonoEisDataset(ionogram_folder, radar_folder, matching_filenames, transform=transform)



sample = DataLoader(A, batch_size=2, shuffle=False)


model = CNN()

# Iterate through the DataLoader to get images
for image, _ in sample:
    print(image.shape)  # Ensure this prints [1, 3, 81, 81]
    y = model(image)
    
    for i in range(5):
        print(y[i].shape)
        
    break












































