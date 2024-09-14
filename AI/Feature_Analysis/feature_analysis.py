# -*- coding: utf-8 -*-
"""
Created on Sat Sep 14 16:07:10 2024

@author: Kian Sartipzadeh
"""


import numpy as np


class FeatureAnalysis:
    
    
    def __init__(self, dataset):
        
        self.dataset = dataset
    
    
    
    def correlation_matrix(self):
        
        corr_matrix = np.corrcoef(self.dataset, rowvar=False)
        
        return corr_matrix
    
    
    
    
    























