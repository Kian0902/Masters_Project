import os
import torch
import torch.nn as nn
from torchvision import transforms

# Assuming utils.py and plot_ionogram are in the same folder
# from utils import plot_ionogram


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
            
            nn.Linear(64 * 10 * 10, 3200),
            nn.BatchNorm1d(3200),
            nn.ReLU(),
            
            nn.Linear(3200, 1600),
            nn.BatchNorm1d(1600),
            nn.ReLU(),
            
            nn.Linear(1600, 400),
            nn.BatchNorm1d(400),
            nn.ReLU(),
            
            nn.Linear(400, 27)
        )

        # self.fc2 = nn.Sequential(
        #     nn.Linear(800, 27)
        # )

    def forward(self, x):
        c1 = self.conv1(x)
        # print(f'After conv1: {c1.shape}')
        c2 = self.conv2(c1)
        # print(f'After conv2: {c2.shape}')
        c3 = self.conv3(c2)
        # print(f'After conv3: {c3.shape}')

        x_flat = c3.view(c3.size(0), -1)
        # print(f'After flattening: {x_flat.shape}')

        x4 = self.fc1(x_flat)
        # print(f'After fc1: {x4.shape}')
        # x5 = self.fc2(x4)
        # print(f'After fc2: {x5.shape}')

        return x4


# # Instantiate the model
# model = CNN()
# model.eval()
# # Create a dummy input tensor with shape (1, 3, 81, 81)
# dummy_input = torch.randn(1, 3, 81, 81)

# # Pass the dummy input through the model
# output = model(dummy_input)



# # -*- coding: utf-8 -*-
# """
# Created on Mon Sep 30 11:16:01 2024

# @author: Kian Sartipzadeh
# """



# import os
# import torch
# import torch.nn as nn
# from torchvision import transforms

# from utils import plot_ionogram





# class CNN(nn.Module):
#     def __init__(self):
#         super(CNN, self).__init__()
        
        
#         self.conv1 = nn.Sequential(
#             nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(16),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
#             )
        
#         self.conv2 = nn.Sequential(
#             nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(32),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
#             )
        
#         self.conv3 = nn.Sequential(
#             nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(64),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
#             )
        
        
#         self.fc1 = nn.Sequential(
#             nn.Linear(64*10*10, 800),
#             nn.BatchNorm1d(800),
#             nn.ReLU(),
#             )
        
#         self.fc2 = nn.Sequential(
#             nn.Linear(800, 27)
#             )
    
#     def forward(self, x):
        
#         c1 = self.conv1(x)
#         c2 = self.conv2(c1)
#         c3 = self.conv3(c2)
        
#         x_flat = c3.view(c3.size(0), -1)
        
#         x4 = self.fc1(x_flat)
#         x5 = self.fc2(x4)
        
#         return x5


























