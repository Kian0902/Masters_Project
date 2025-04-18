import os
import torch
import torch.nn as nn
from torchvision import transforms
import torch.nn.functional as F
# Assuming utils.py and plot_ionogram are in the same folder
# from utils import plot_ionogram
from torchsummary import summary


# # Feed-Forward Neural Network (FFNN)
# class GeoDMLP(nn.Module):
#     def __init__(self):
#         super(GeoDMLP, self).__init__()
        
#         self.fc = nn.Sequential(
#             nn.Linear(25, 64),
#             nn.BatchNorm1d(64),
#             nn.ReLU(),
#             # nn.Dropout(0.2),
            
#             nn.Linear(64, 128),
#             nn.BatchNorm1d(128),
#             nn.ReLU(),
#             # nn.Dropout(0.2),
            
#             nn.Linear(128, 256),
#             nn.BatchNorm1d(256),
#             nn.ReLU(),
#             # nn.Dropout(0.2),
            
#             nn.Linear(256, 512),
#             nn.BatchNorm1d(512),
#             nn.ReLU(),
#             # nn.Dropout(0.2),
            
#             nn.Linear(512, 1024),
#             nn.BatchNorm1d(1024),
#             nn.ReLU(),
#             # nn.Dropout(0.2),
#         )
        
#     def forward(self, x):
#         x = self.fc(x)
#         return x






# class IonoCNN(nn.Module):
#     def __init__(self):
#         super(IonoCNN, self).__init__()

#         self.conv = nn.Sequential(
#             nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(16),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2, stride=2, padding=0),   # 81 --> 40

#             nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(32),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2, stride=2, padding=0),   # 40 --> 20

#             nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(64),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2, stride=2, padding=0),   # 20 --> 10
            
#             nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(128),
#             nn.ReLU(),
#         )
        
        
#     def forward(self, x):
#         c = self.conv(x)
#         x_flat = c.view(c.size(0), -1)
#         return x_flat






# class FuDMLP(nn.Module):
#     def __init__(self):
#         super(FuDMLP, self).__init__()
        
#         # Instantiate the two branches
#         self.iono_cnn = IonoCNN()
#         self.geo_dmlp = GeoDMLP()

        
#         self.fc = nn.Sequential(
#             nn.Linear(128*10*10 + 1024, 6912),
#             nn.BatchNorm1d(6912),
#             nn.ReLU(),

#             nn.Linear(6912, 3456),
#             nn.BatchNorm1d(3456),
#             nn.ReLU(),

#             nn.Linear(3456, 1728),
#             nn.BatchNorm1d(1728),
#             nn.ReLU(),
            
#             nn.Linear(1728, 864),
#             nn.BatchNorm1d(864),
#             nn.ReLU(),

#             nn.Linear(864, 432),
#             nn.BatchNorm1d(432),
#             nn.ReLU(),


#             nn.Linear(432, 216),
#             nn.BatchNorm1d(216),
#             nn.ReLU(),

#             nn.Linear(216, 108),
#             nn.BatchNorm1d(108),
#             nn.ReLU(),
            

#             nn.Linear(108, 27)
#             )
        
#     def forward(self, img, geo):
#         # Forward pass through each branch
#         img_out = self.iono_cnn(img)
#         geo_out = self.geo_dmlp(geo)
        
#         # Concatenate outputs
#         fusion = torch.cat((img_out, geo_out), dim=1)
        
#         x = self.fc(fusion)
#         return x




class GeoDMLP(nn.Module):
    def __init__(self):
        super(GeoDMLP, self).__init__()
        
        self.fc = nn.Sequential(
            nn.Linear(25, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
        )
        
    def forward(self, x):
        x = self.fc(x)
        return x



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
        
        
    def forward(self, x):
        c = self.conv(x)
        x_flat = c.view(c.size(0), -1)
        return x_flat






class FuDMLP(nn.Module):
    def __init__(self):
        super(FuDMLP, self).__init__()
        
        # Instantiate the two branches
        self.iono_cnn = IonoCNN()
        self.geo_dmlp = GeoDMLP()

        
        self.fc = nn.Sequential(
            nn.Linear(128*10*10 + 1024, 6912),
            nn.BatchNorm1d(6912),
            nn.ReLU(),

            nn.Linear(6912, 3456),
            nn.BatchNorm1d(3456),
            nn.ReLU(),

            nn.Linear(3456, 1728),
            nn.BatchNorm1d(1728),
            nn.ReLU(),
            
            nn.Linear(1728, 864),
            nn.BatchNorm1d(864),
            nn.ReLU(),

            nn.Linear(864, 432),
            nn.BatchNorm1d(432),
            nn.ReLU(),

            nn.Linear(432, 216),
            nn.BatchNorm1d(216),
            nn.ReLU(),

            nn.Linear(216, 108),
            nn.BatchNorm1d(108),
            nn.ReLU(),

            #nn.Linear(108, 54),
            #nn.BatchNorm1d(54),
            #nn.ReLU(),

            nn.Linear(108, 27)
            )
        
    def forward(self, img, geo):
        # Forward pass through each branch
        img_out = self.iono_cnn(img)
        geo_out = self.geo_dmlp(geo)
        
        # Concatenate outputs
        fusion = torch.cat((img_out, geo_out), dim=1)
        
        x = self.fc(fusion)
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


















