# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 17:35:25 2024

@author: Kian Sartipzadeh
"""


import torch
import torch.nn as nn

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")






class FFN_Geophys(nn.Module):
    def __init__(self):
        super(FFN_Geophys, self).__init__()
        
        
        # Layers
        self.FC = nn.Sequential(
            nn.Linear(19, 128),
            # nn.BatchNorm1d(128),
            nn.ReLU(),
            
            nn.Linear(128, 256),
            # nn.BatchNorm1d(256),
            nn.ReLU(),
            
            
            nn.Linear(256, 128),
            # nn.BatchNorm1d(128),
            nn.ReLU(),
            
            nn.Linear(128, 64),
            # nn.BatchNorm1d(64),
            nn.ReLU(),
            
            nn.Linear(64, 27)
        )

    
    def forward(self, x):
        x = self.FC(x)
        return x






class MLP19_Chap(nn.Module):
    
    def __init__(self):
        super(MLP19_Chap, self).__init__()
        
        # Layers
        self.layers = nn.Sequential(nn.Linear(19, 124),
                                    nn.BatchNorm1d(124),
                                    nn.ReLU(),
                                    nn.Linear(124, 64),
                                    nn.BatchNorm1d(64),
                                    nn.ReLU(),
                                    nn.Linear(64, 8)
                                    )
        
        # Activation
        self.softplus_H = nn.Softplus(beta=0.1, threshold=10)   # For scale-heights
        self.softplus_N = nn.Softplus(beta=0.15, threshold=10)  # For peaks
    
    
    
    def double_chapman(self, x, z):
        HE_below, HF_below, HE_above, HF_above, nE_peak, nF_peak, zE_peak, zF_peak = x.split(1, dim=1)
        
        HE = torch.where(z < zE_peak, HE_below, HE_above)
        HF = torch.where(z < zF_peak, HF_below, HF_above)
    
        neE = nE_peak * torch.exp(1 - ((z - zE_peak) / HE) - torch.exp(-((z - zE_peak) / HE)))
        neF = nF_peak * torch.exp(1 - ((z - zF_peak) / HF) - torch.exp(-((z - zF_peak) / HF)))
        
        ne = neE + neF
        return ne
    
    
    
    def forward(self, x, z):
        
        x = self.layers(x)
        
        x_H = x[:,:4]
        x_N = x[:,4:]
        
        x_H = self.softplus_H(x_H)
        x_N = self.softplus_N(x_N)
        
        
        x_final =  torch.cat([x_H, x_N], dim=1)
        
        batch_size = x_final.size(0)
        z = z.unsqueeze(0).expand(batch_size, -1).to(device)
        
        DChap_pred = self.double_chapman(x_final, z)
        return DChap_pred
        




# Function for He initialization
def he_initialization(module):
    
    # For Conv and fc layers
    if isinstance(module, nn.Linear):
        nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
            
    # For Batchnorm Layers
    elif isinstance(module, (nn.BatchNorm1d)):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)
























