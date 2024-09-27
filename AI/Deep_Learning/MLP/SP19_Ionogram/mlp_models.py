# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 17:35:25 2024

@author: Kian Sartipzadeh
"""


import torch
import torch.nn as nn

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



class MLP19(nn.Module):
    
    def __init__(self):
        super(MLP19, self).__init__()
        
        # Layers
        self.layers = nn.Sequential(nn.Linear(19, 124),
                                    nn.BatchNorm1d(124),
                                    nn.ReLU(),
                                    nn.Linear(124, 64),
                                    nn.BatchNorm1d(64),
                                    nn.ReLU(),
                                    nn.Linear(64, 27)
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
        
        # x_H = x[:,:4]
        # x_N = x[:,4:]
        
        # x_H = self.softplus_H(x_H)
        # x_N = self.softplus_N(x_N)
        
        
        # x_final =  torch.cat([x_H, x_N], dim=1)
        
        # batch_size = x_final.size(0)
        # z = z.unsqueeze(0).expand(batch_size, -1).to(device)
        
        # DChap_pred = self.double_chapman(x_final, z)
        return x
        








class MLP19ION(nn.Module):
    
    def __init__(self):
        super(MLP19ION, self).__init__()
        
        
        
        # Dropout probability
        self.dropout_prob = 0.5
        
        # Layers
        self.layers = nn.Sequential(nn.Linear(262, 2048),
                                    nn.BatchNorm1d(2048),
                                    nn.ReLU(),
                                    nn.Dropout(p=self.dropout_prob),
                                    
                                    nn.Linear(2048, 4096),
                                    nn.BatchNorm1d(4096),
                                    nn.ReLU(),
                                    nn.Dropout(p=self.dropout_prob),
                                    
                                    nn.Linear(4096, 2048),
                                    nn.BatchNorm1d(2048),
                                    nn.ReLU(),
                                    nn.Dropout(p=self.dropout_prob),
                                    
                                    nn.Linear(2048, 1024),
                                    nn.BatchNorm1d(1024),
                                    nn.ReLU(),
                                    nn.Dropout(p=self.dropout_prob),
                                    
                                    nn.Linear(1024, 512),
                                    nn.BatchNorm1d(512),
                                    nn.ReLU(),
                                    
                                    nn.Linear(512, 27),
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
        
        # x_H = x[:,:4]
        # x_N = x[:,4:]
        
        # x_H = self.softplus_H(x_H)
        # x_N = self.softplus_N(x_N)
        
        
        # x_final =  torch.cat([x_H, x_N], dim=1)
        
        # batch_size = x_final.size(0)
        # z = z.unsqueeze(0).expand(batch_size, -1).to(device)
        
        # DChap_pred = self.double_chapman(x_final, z)
        return x




















