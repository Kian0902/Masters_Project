# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 16:06:57 2025

@author: kian0
"""

import os
import torch
import torch.nn as nn
from torchvision import transforms
import torch.nn.functional as F
# from utils import plot_ionogram
from torchsummary import summary




class BranchCNN(nn.Module):
    def __init__(self):
        super(BranchCNN, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),   # 81 --> 40

            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),   # 40 --> 20

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),   # 20 --> 10
            
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        
        
    def forward(self, x):
        c = self.conv(x)
        x_flat = c.view(c.size(0), -1)
        return x_flat




# Function for He initialization
def he_initialization(module):
    
    # For Conv and fc layers
    if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
            
    # For Batchnorm Layers
    elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)




if __name__ == "__main__":
    # Move models to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create instances of the networks
    cnn = BranchCNN().to(device)
   
    # Apply He initialization
    cnn.apply(he_initialization)
    
    
    print("Iono-CNN Summary:")
    summary(cnn, input_size=(3, 81, 81))
    
