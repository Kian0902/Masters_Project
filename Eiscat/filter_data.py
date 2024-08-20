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
    Class for clipping and filtering data with dictionaries. 
    """
    def __init__(self, dataset):
        """
        Attributes (type)  | DESCRIPTION
        ------------------------------------------------
        dataset (dict)     | Single dictionary containing the matlab data.
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
            self.dataset[k] = self.dataset[k][mask, 0]
            
            
        

    def handle_nan(self, replace_val):
        """
        Function for handling arrays with nan values. This method interpolates
        and/or extrapolates way nans. If more than half of the array contains
        nans, then user has the choise of which value to fill the entire array
        with.
        
        Input (type)            | DESCRIPTION
        ------------------------------------------------
        replace_val (int/float) | Value to replace nans with
        """
        
        # Looping through keys
        # print(len(self.dataset.keys()))
        for key in self.dataset:
            
            
            
            
            # If any values within key is nan
            if np.isnan(self.dataset[key]).any() == True:
                nan_indices = np.isnan(self.dataset[key]) # Find the indices of NaN values
                
                
                # If more than half of array contains nan
                """ This is so that faulty data can be detected"""
                if len(nan_indices[nan_indices == True]) > len(self.dataset[key])/2:
                    self.dataset[key].fill(replace_val)


                # If array contains any other instances of nans
                else:
                    
                    self.dataset[key][nan_indices] = interp1d(np.arange(len(self.dataset[key]))[~nan_indices], 
                                              self.dataset[key][~nan_indices], 
                                              kind='linear', fill_value="extrapolate")(np.arange(len(self.dataset[key]))[nan_indices])
                    
                    
            # If nan is not detected
            else:
                pass
                
                
        # # Allow the user to return data after applying method
        # if return_data == True:
        #     return self.data


    def return_data(self):
        """
        Returns self.data
        """
        return self.dataset
        












