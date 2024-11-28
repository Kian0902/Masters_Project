import os
import torch
import torch.nn as nn
from torchvision import transforms
import torch.nn.functional as F
# Assuming utils.py and plot_ionogram are in the same folder
# from utils import plot_ionogram




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
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)   # 81 --> 40
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)   # 40 --> 20
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)   # 20 --> 10
        )
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        

    def forward(self, x):
        c1 = self.conv1(x)
        c2 = self.conv2(c1)
        c3 = self.conv3(c2)
        c4 = self.conv4(c3)
        
        x_flat = c4.view(c4.size(0), -1)

        return x_flat





class CombinedNetwork(nn.Module):
    def __init__(self):
        super(CombinedNetwork, self).__init__()
        
        # Instantiate the two branches
        self.cnn_branch = BranchCNN()
        self.ffnn_branch = BranchFNN()
        

        
        self.fc1 = nn.Sequential(
            nn.Linear(128 * 10 * 10 + 1024, 6912),
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
    print("...")


















