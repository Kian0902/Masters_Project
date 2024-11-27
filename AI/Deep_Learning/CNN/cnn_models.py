# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 11:16:01 2024

@author: Kian Sartipzadeh
"""


import torch
import torch.nn as nn



class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)      # (81,81) --> (40, 40)
        )

        
    
        self.fc1 = nn.Sequential(
            
            nn.Linear(16*40*40, 4800),
            nn.BatchNorm1d(4800),
            nn.ReLU(),
            
            nn.Linear(4800, 600),
            nn.BatchNorm1d(600),
            nn.ReLU(),
            
            nn.Linear(600, 27)
            
        )
        

    def forward(self, x):
        c1 = self.conv1(x)

        x_flat = c1.view(c1.size(0), -1)
        
        x1 = self.fc1(x_flat)
        
        return x1





if __name__ == "__main__":
    print("...")


















