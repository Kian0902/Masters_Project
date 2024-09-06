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



class EISCATDataFilter:
    """
    Class for clipping and filtering dict data.
    """
    def __init__(self, dataset: dict, filt_range: bool=False, filt_nan: bool=False, filt_outlier: bool=False):
        """
        Attributes (type)  | DESCRIPTION
        ------------------------------------------------
        dataset (dict)     | Single dictionary containing processed data.
        filt_range (bool)  | Whether to apply range filtering.
        filt_nan   (bool)  | Whether to apply NaN filtering.
        range_key (str)    | Key to apply range filtering on.
        min_val (float)    | Minimum value for range filtering. Default = 90km
        max_val (float)    | Maximum value for range filtering. Default = 400km
        """
        self.dataset = dataset
        self.apply_range_filter = filt_range
        self.apply_nan_filter = filt_nan
        self.apply_outlier_filter = filt_outlier
    
    
    def batch_filtering(self, min_val=90, max_val=400):
        """
        Function for applying the filtering to the entire dataset by looping
        through the global keys (days).
        """
        # Loop through day
        for key in list(self.dataset.keys()):
            
            # Filter range
            if self.apply_range_filter:
                self.dataset[key] = self.filter_range(self.dataset[key], 'r_h', min_val, max_val)
            
            # Filter nans
            if self.apply_nan_filter:
                self.dataset[key] = self.filter_nan(self.dataset[key])
        
        
            # Filter outliers
            if self.apply_outlier_filter:
                ...
        
        
    
    def filter_range(self, data: dict, key: str, min_val: float, max_val: float):
        """
        Filters out values that lie outside an interval [min_val, max_mal]
        specified by the user.
        It is assumed that each key in self.dataset is assigned a (N x M)
        numpy.ndarray.
        
        Input (type)    | DESCRIPTION
        ------------------------------------------------
        key     (str)   | Key of the dictionary
        min_val (float) | Lower limit of interval
        max_val (float) | Upper limit of interval
        """
        
        
        # Mask for filtered values. False outside and True inside interval
        mask = np.any((data[key] >= min_val) & (data[key] <= max_val), axis=1)
        
        # Applying mask to all keys except "r_time"
        for key in list(data.keys())[1:]:
            data[key] = data[key][mask,:]
        
        return data
        
        
    def filter_nan(self, data: dict, replace_val=111):
        """
        Function for handling numpy.ndarrays with NaN values. This method
        interpolates and/or extrapolates way NaNs. If more than half of the
        array contains NaNs, then user has the choice of picking a fill value
        to replace the entire array. This is so it becomes easier later to 
        distinguish faulty arrays from healthy ones.
        
        Input (type)            | DESCRIPTION
        ------------------------------------------------
        replace_val (int/float) | Value to replace nans with
        """
        
        # Looping through keys except "r_time"
        for key in list(data.keys())[1:]:
            X = data[key]
            
            # Check if there are any NaNs in the data
            if np.isnan(X).any():
                # Find the indices of NaN values
                nan_mask = np.isnan(X)
                
                # If more than half of the elements in the array are NaN
                if np.sum(nan_mask) > (X.shape[0] * X.shape[1]) / 2:
                    # Fill the entire array with replace_val
                    data[key].fill(replace_val)
                else:
                    # Interpolate/extrapolate the NaNs for each column (minute)
                    x = np.arange(X.shape[0])  # Indexes for rows (altitude measurements)
                    
                    for j in range(X.shape[1]):  # Iterate over each column (minute)
                        col = X[:, j]
                        valid_mask = ~nan_mask[:, j]
                        if np.any(valid_mask):  # Check if there's at least one valid point
                            interp_func = interp1d(x[valid_mask], col[valid_mask], 
                                                   kind='linear', fill_value="extrapolate")
                            col[~valid_mask] = interp_func(x[~valid_mask])
                        else:
                            # If the entire column is NaN, fill it with replace_val
                            X[:, j].fill(replace_val)
                    
                    data[key] = X
       
        return data
    
    
    
    
    
    def return_data(self):
        """
        Returns self.data
        """
        return self.dataset





























