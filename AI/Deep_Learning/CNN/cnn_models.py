# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 11:16:01 2024

@author: Kian Sartipzadeh
"""



import os
import torch
import torch.nn as nn
from torchvision import transforms

from utils import plot_ionogram
from storing_dataset import IonoEisDataset




class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        
        # Conv Layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        
        
        # Pooling Layers
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
    
    def forward(self, x):
        print(x.shape)
        
        x1 = self.pool(x)
        
        print(x1.shape)
        
        x2 = self.pool(x1)
        
        print(x2.shape)
        
        x3 = self.pool(x2)
        
        print(x3.shape)
        
        x4 = self.pool(x3)
        
        print(x4.shape)
        return x1, x2, x3, x4

# model = CNN()
# dummy_input = torch.randn(1, 3, 81, 81)
# output = model(dummy_input)


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

# A.plot_sample_pair(93)


sample = A[0][0]

plot_ionogram(sample)


model = CNN()
y1, y2, y3, y4 = model(sample)


plot_ionogram(y1)
plot_ionogram(y2)
plot_ionogram(y3)
plot_ionogram(y4)
























