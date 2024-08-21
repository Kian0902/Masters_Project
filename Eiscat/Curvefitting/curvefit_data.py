# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 10:31:39 2024

@author: Kian Sartipzadeh
"""




import os
import pickle

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn





class Curvefit(nn.Module):
    """
    Class for curvefitting using a Neural Network. 
    """
    
    def __init__(self):
        super(Curvefit, self).__init__()
        
        # Setting up parameters to be optimized
        self.HE_below = nn.Parameter(torch.tensor([np.log(5)], dtype=torch.float64), requires_grad=True)
        self.HE_above = nn.Parameter(torch.tensor([np.log(10)], dtype=torch.float64), requires_grad=True)
        self.HF_below = nn.Parameter(torch.tensor([np.log(25)], dtype=torch.float64), requires_grad=True)
        self.HF_above = nn.Parameter(torch.tensor([np.log(40)], dtype=torch.float64), requires_grad=True)


    def forward(self, x, nE_peak, nF_peak, zE_peak, zF_peak):
        """
        Forward propagation function.
        
        Input (type) | DESCRIPTION
        ------------------------------------------------
        x            |
        nE_peak      |
        nF_peak      |
        zE_peak      |
        zF_peak      |
        
        Return (type) | DESCRIPTION
        ------------------------------------------------
        neE + neF     | 
        """

        # Converting numpy.float64 to torch.float64
        nE_peak = torch.tensor([nE_peak], dtype=torch.float64)
        nF_peak = torch.tensor([nF_peak], dtype=torch.float64)
        
        zE_peak = torch.tensor([zE_peak], dtype=torch.float64)
        zF_peak = torch.tensor([zF_peak], dtype=torch.float64)
        
        # Getting rid of log
        HE_below = torch.exp(self.HE_below)
        HE_above = torch.exp(self.HE_above)
        HF_below = torch.exp(self.HF_below)
        HF_above = torch.exp(self.HF_above)
        
        # Appending scale heights corresponding on altitude
        HE = torch.where(x < zE_peak, HE_below, HE_above)
        HF = torch.where(x < zF_peak, HF_below, HF_above)
        
        
        # Defining chapman electron density profile for both regions
        neE = nE_peak * torch.exp(1 - (x - zE_peak) / HE - torch.exp(-((x - zE_peak) / HE))) 
        neF = nF_peak * torch.exp(1 - (x - zF_peak) / HF - torch.exp(-((x - zF_peak) / HF)))
        
        return neE + neF




# Importing processed dataset
custom_file_path = "C:\\Users\\kian0\\OneDrive\\Desktop\\UiT Courses\\FYS_3931\\Scripts\\Masters_Project\\Eiscat\\Curvefitting\\Sorted_data.pkl"
with open(custom_file_path, 'rb') as f:
    dataset = pickle.load(f)







































