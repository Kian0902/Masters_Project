# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 14:42:19 2024

@author: Kian Sartipzadeh
"""



import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import numpy as np

from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


X, y = make_blobs(n_samples=1000, centers=2, n_features=2)


# Standardize
scaler = StandardScaler()
X = scaler.fit_transform(X)


# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)







# class MultiLayeredPerceptron(nn.Module):
#     def __init__(self):
#         super(MultiLayeredPerceptron, self).__init__()
        
#         self.FC1 = nn.Linear(19, 100)
#         self.FC2 = nn.Linear(100, out_features)
        


























