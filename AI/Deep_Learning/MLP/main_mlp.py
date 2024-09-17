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
import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


X, y = make_blobs(n_samples=1000, centers=2, n_features=2)


# Standardize
scaler = StandardScaler()
X = scaler.fit_transform(X)


# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=1)


plt.figure()
plt.title("Blobs dataset")
plt.scatter(X_train[:,0][y_train==0], X_train[:,1][y_train==0], color="C0", label="Class 1")
plt.scatter(X_train[:,0][y_train==1], X_train[:,1][y_train==1], color="C1", label="Class 2")
plt.xlabel("X1")
plt.ylabel("X2")
plt.legend()
plt.show()




























