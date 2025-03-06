import os
import torch
import torch.nn as nn
from torchvision import transforms
import torch.nn.functional as F
# Assuming utils.py and plot_ionogram are in the same folder
# from utils import plot_ionogram
from torchsummary import summary



class FuDMLP(nn.Module):
    def __init__(self, input_size):
        super(FuDMLP, self).__init__()
        
        self.fuse = nn.Sequential(
            nn.Linear(input_size, 7424),
            nn.BatchNorm1d(7424),
            nn.ReLU(),
            
            nn.Linear(7424, 3712),
            nn.BatchNorm1d(3712),
            nn.ReLU(),
            
            nn.Linear(3712, 1856),
            nn.BatchNorm1d(1856),
            nn.ReLU(),
            
            nn.Linear(1856, 928),
            nn.BatchNorm1d(928),
            nn.ReLU(),
            
            nn.Linear(928, 464),
            nn.BatchNorm1d(464),
            nn.ReLU(),
            
            nn.Linear(464, 232),
            nn.BatchNorm1d(232),
            nn.ReLU(),
            
            nn.Linear(232, 116),
            nn.BatchNorm1d(116),
            nn.ReLU(),
            
            nn.Linear(116, 58),
            nn.BatchNorm1d(58),
            nn.ReLU(),
            
            
            nn.Linear(58, 27)
            
        )
    
    
    def forward(self, x):
        x = self.fuse(x)
        return x




class KIANNet(nn.Module):
    def __init__(self, iono_conv, geo_fc1, fu_dmlp):
        super(KIANNet, self).__init__()
        
        self.iono_conv = iono_conv
        self.geo_fc1 = geo_fc1
        self.fu_dmlp = fu_dmlp
        
        
        # Freeze the pre-trained features
        for param in self.iono_conv.parameters():
            param.requires_grad = False
        for param in self.geo_fc1.parameters():
            param.requires_grad = False
    
    
    def forward(self, image, tabular):
        iono_feat = self.iono_conv(image).view(image.size(0), -1)
        geo_feat = self.geo_fc1(tabular)
        combined = torch.cat((iono_feat, geo_feat), dim=1)
        output = self.fu_dmlp(combined)
        return output





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


















