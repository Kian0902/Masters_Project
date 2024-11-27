import os
import torch
import torch.nn as nn
from torchvision import transforms
import torch.nn.functional as F
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
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
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
        c2 = self.conv2(c1)
        c3 = self.conv3(c2)
        c4 = self.conv4(c3)
        
        x_flat = c4.view(c4.size(0), -1)

        x4 = self.fc1(x_flat)
        return x4






class BranchFNN(nn.Module):
    def __init__(self):
        super(BranchFNN, self).__init__()
        
        self.f1 = nn.Sequential(
            nn.Linear(19, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )
    
        self.f2 = nn.Sequential(
            nn.Linear(64, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
        )
        
        self.f3 = nn.Sequential(
            nn.Linear(256, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
        )
    
    def forward(self, x):
        x1 = self.f1(x)
        x2 = self.f2(x1)
        x3 = self.f3(x2)
        return x3






class BranchCNN(nn.Module):
    def __init__(self):
        super(BranchCNN, self).__init__()

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
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            #nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        )
        
        #self.conv4 = nn.Sequential(
        #    nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
        #    nn.BatchNorm2d(64),
        #    nn.ReLU(),
        #)
        

    def forward(self, x):
        c1 = self.conv1(x)
        c2 = self.conv2(c1)
        c3 = self.conv3(c2)
        #c4 = self.conv4(c3)
        
        x_flat = c3.view(c3.size(0), -1)

        return x_flat






class CombinedNetwork(nn.Module):
    def __init__(self):
        super(CombinedNetwork, self).__init__()
        
        # Instantiate the two branches
        self.cnn_branch = BranchCNN()
        self.ffnn_branch = BranchFNN()
        

        
        self.fc1 = nn.Sequential(
            nn.Linear(32 * 20 * 20 + 1024, 6912),
            nn.BatchNorm1d(6912),
            nn.ReLU(),
            )
        
        self.fc2 = nn.Sequential(
            nn.Linear(6912, 3456),
            nn.BatchNorm1d(3456),
            nn.ReLU(),
            )
        
        self.fc3 = nn.Sequential(
            nn.Linear(3456, 1728),
            nn.BatchNorm1d(1728),
            nn.ReLU(),
            )
        
        self.fc4 = nn.Sequential(
            nn.Linear(1728, 864),
            nn.BatchNorm1d(864),
            nn.ReLU(),
            )
        
        self.fc5 = nn.Sequential(
            nn.Linear(864, 27),
            )
        
    def forward(self, img, geo):
        # Forward pass through each branch
        img_out = self.cnn_branch(img)
        geo_out = self.ffnn_branch(geo)
        
        # Concatenate outputs
        combined = torch.cat((img_out, geo_out), dim=1)
        
        x = self.fc1(combined)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        x = self.fc5(x)
        return x








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







if __name__ == "__main__":
    print("...")


















