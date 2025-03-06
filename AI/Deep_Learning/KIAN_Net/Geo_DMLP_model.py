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


class GeoDMLP(nn.Module):
    def __init__(self):
        super(GeoDMLP, self).__init__()
        
        self.fc1 = nn.Sequential(
            nn.Linear(19, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(1024, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        
        
        self.fc2 = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(64, 27),
        )
        
        
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x






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
    geo_dmlp = GeoDMLP().to(device)
   
    # Apply He initialization
    geo_dmlp.apply(he_initialization)
    
    
    print("Iono-CNN Summary:")
    summary(geo_dmlp, input_size=(19,))
    
