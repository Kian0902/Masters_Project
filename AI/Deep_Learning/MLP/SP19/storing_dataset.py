# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 14:55:24 2024

@author: Kian Sartipzadeh
"""

import torch
from torch.utils.data import Dataset


class StoreDataset(Dataset):
    def __init__(self, data, targets):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

































