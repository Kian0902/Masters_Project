# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 15:10:52 2024

@author: Kian Sartipzadeh
"""

import os
import pickle

import numpy as np
from scipy.io import loadmat
from scipy.interpolate import interp1d



class DataFiltering:
    """
    Class for clipping and filtering dict data.
    """
    def __init__(self, dataset):
        """
        Attributes (type)  | DESCRIPTION
        ------------------------------------------------
        dataset (dict)     | Single dictionary containing processed data.
        """
        self.dataset = dataset
    
    
    
    def filter_range(self, key: str, min_val: float, max_val: float):
        """
        Filters out values that lie outside an interval [min_val, max_mal]
        specified by the user.
        
        Input (type)    | DESCRIPTION
        ------------------------------------------------
        key     (str)   | Key of the dictionary
        min_val (float) | Lower limit of interval
        max_val (float) | Upper limit of interval
        """
        
        
        # Mask for filtered values. False outside and True inside interval
        mask = np.any((self.dataset[key] >= min_val) & (self.dataset[key] <= max_val), axis=1)
        
        for key in list(self.dataset.keys()):
            print(self.dataset[key].shape)
            print(mask.shape)
            self.dataset[key] = self.dataset[key][mask,:]
            print(self.dataset[key].shape)
        
        
        
        
        
        # print(type(mask))
        # mask = mask.ravel()

        
        # # Applying mask to all keys
        # for k in list(self.dataset.keys()):
            
        #     # print(self.dataset[k].shape)
        #     self.dataset[k] = self.dataset[k][mask, :]
        #     break


























































