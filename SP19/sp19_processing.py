# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 13:32:17 2024

@author: Kian Sartipzadeh
"""

custom_header=['DoY/366', 'ToD/1440', 'Solar_Zenith/44', 'Kp', 'R', 'Dst',
                'ap', 'F10_7', 'AE', 'AL', 'AU', 'PC_potential', 'Lyman_alpha',
                'Bx', 'By', 'Bz', 'dBx', 'dBy', 'dBz']


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.neighbors import LocalOutlierFactor




class Processing:
    
    def __init__(self, dataset):
        self.dataset = dataset
    
    
    def filter_missing_values(self):
        
        if self.dataset.isnull().values.any():
            self.dataset = self.dataset.interpolate(method='linear', axis=0, limit_direction='forward')
        
        else:
            print("No missing values detected!")
    
    
    
    def plot_hist(self, feature=None):
        
        data = self.dataset.to_numpy()
        
        print(len(data))
        
        sns.histplot(data[:,feature], bins=int(len(data)/1000), kde=True, color="C0")
        plt.xlabel('Values')
        plt.ylabel('Density')
        plt.title(f'Feature {feature}')
        plt.show()
        

data = np.load("sp19_features.npy")


a=80000

X = pd.DataFrame(data[:a])


A = Processing(X)
A.filter_missing_values()

for i in range(0, 20):
    A.plot_hist(feature=i)










































