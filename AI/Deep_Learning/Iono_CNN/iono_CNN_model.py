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




class IonoCNN(nn.Module):
    def __init__(self):
        super(IonoCNN, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),   # 81 --> 40

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),   # 40 --> 20

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),   # 20 --> 10
            
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),   # 10 --> 5

            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            
        )
        
        
        self.mlp = nn.Sequential(
            nn.Linear(12800, 6400),
            nn.BatchNorm1d(6400),
            nn.ReLU(),
            
            nn.Linear(6400, 3200),
            nn.BatchNorm1d(3200),
            nn.ReLU(),
            
            nn.Linear(3200, 1600),
            nn.BatchNorm1d(1600),
            nn.ReLU(),
            
            nn.Linear(1600, 800),
            nn.BatchNorm1d(800),
            nn.ReLU(),
            
            nn.Linear(800, 200),
            nn.BatchNorm1d(200),
            nn.ReLU(),
            
            nn.Linear(200, 27)
        )
        
        
    def forward(self, x):
        c = self.conv(x)
        x_flat = c.view(c.size(0), -1)
        
        m = self.mlp(x_flat)
        
        return m




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
    iono_cnn = IonoCNN().to(device)
   
    # Apply He initialization
    iono_cnn.apply(he_initialization)
    
    
    print("Iono-CNN Summary:")
    summary(iono_cnn, input_size=(3, 81, 81))
    
