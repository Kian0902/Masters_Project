# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 14:56:19 2024

@author: Kian Sartipzadeh
"""





import os
import pickle

import numpy as np
from scipy.io import loadmat
from scipy.interpolate import interp1d







class DataFiltering:
    """
    This class is for clipping and filtering data with dictionaries. 
    """
    def __init__(self, dataset):
        """
        Attributes (type)  | DESCRIPTION
        ------------------------------------------------
        dataset (dict)     | Dictionary containing the data.
        """
        self.dataset = dataset
    
    
    def filter_range(self, key: str, min_val: float, max_val: float):
        """
        Filters out values that lie outside an interval [min_val, max_mal]
        specified by the user. This is assuming that each key has an ndarray
        as value. In other words,  key:ndarray
        
        Input (type)    | DESCRIPTION
        ------------------------------------------------
        key     (str)   | Key of the dictionary
        min_val (float) | Lower limit of interval
        max_val (float) | Upper limit of interval
        """
        
        # Mask for filtered values. False outside and True inside interval
        mask = (self.dataset[key] >= min_val) & (self.dataset[key] <= max_val)
        mask = mask.ravel()
        

        # Applying mask to all keys
        for k in list(self.dataset.keys())[1:]:
            self.dataset = self.dataset[k][mask, 0]

        

        












