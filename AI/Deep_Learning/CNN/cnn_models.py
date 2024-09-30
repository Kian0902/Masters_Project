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
        
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
            )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
            )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
            )
        
        
        self.fc1 = nn.Sequential(
            nn.Linear(64*10*10, 800),
            nn.BatchNorm1d(800),
            nn.ReLU(),
            )
        
        self.fc2 = nn.Sequential(
            nn.Linear(800, 27),
            nn.BatchNorm1d(27),
            nn.ReLU(),
            )
    
    def forward(self, x):
        
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        
        x_flat = x3.view(x3.size(0), -1)
        
        x4 = self.fc1(x_flat)
        x5 = self.fc2(x4)
        
        return x1, x2, x3, x4, x5


























