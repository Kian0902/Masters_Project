# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 13:12:37 2025

@author: Kian Sartipzadeh
"""


import os
import numpy as np
import matplotlib.pyplot as plt


import torch
from torch.utils.data import Dataset



class StoreDataset(Dataset):
    def __init__(self, data, targets):
        self.targets = torch.tensor(np.array(targets), dtype=torch.float32)
        self.targets = torch.tensor(np.array(targets), dtype=torch.float32)
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        
        return self.data[idx], self.targets[idx]

















