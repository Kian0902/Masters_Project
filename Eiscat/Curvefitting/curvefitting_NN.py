# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 10:29:57 2024

@author: Kian Sartipzadeh
"""


import numpy as np

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








