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
    def __init__(self, dataset: dict):
        """
        Attributes (type)  | DESCRIPTION
        ------------------------------------------------
        dataset (dict)     | Single dictionary containing processed data.
        """
        self.dataset = dataset

        
    
    def batch_filtering(self):
        
        # Loop through day
        for key in list(self.dataset.keys()):
            
            filt_range_data = self.filter_range(self.dataset[key], 'r_h', 90, 400)
            filt_nan_data   = self.filter_nan(filt_range_data)
            self.dataset[key] = filt_nan_data
            
        
    
    
    
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
    
    
    # def filter_nan(self, replace_val=111):
    #     """
    #     Function for handling numpy.ndarrays with NaN values. This method
    #     interpolates and/or extrapolates way NaNs. If more than half of the
    #     array contains NaNs, then user has the choice of picking a fill value
    #     to replace the entire array. This is so it becomes easier later to 
    #     distinguish faulty arrays from healthy ones.
        
    #     Input (type)            | DESCRIPTION
    #     ------------------------------------------------
    #     replace_val (int/float) | Value to replace nans with
    #     """
        
    #     # Looping through keys except "r_time"
    #     for key in list(self.dataset.keys())[1:]:
    #         data = self.dataset[key]
            
    #         # Check if there are any NaNs in the data
    #         if np.isnan(data).any():
    #             # Find the indices of NaN values
    #             nan_mask = np.isnan(data)
                
    #             # If more than half of the elements in the array are NaN
    #             if np.sum(nan_mask) > (data.shape[0] * data.shape[1]) / 2:
    #                 # Fill the entire array with replace_val
    #                 self.dataset[key].fill(replace_val)
    #             else:
    #                 # Interpolate/extrapolate the NaNs for each column (minute)
    #                 x = np.arange(data.shape[0])  # Indexes for rows (altitude measurements)
                    
    #                 for j in range(data.shape[1]):  # Iterate over each column (minute)
    #                     col = data[:, j]
    #                     valid_mask = ~nan_mask[:, j]
    #                     if np.any(valid_mask):  # Check if there's at least one valid point
    #                         interp_func = interp1d(x[valid_mask], col[valid_mask], 
    #                                                kind='linear', fill_value="extrapolate")
    #                         col[~valid_mask] = interp_func(x[~valid_mask])
    #                     else:
    #                         # If the entire column is NaN, fill it with replace_val
    #                         data[:, j].fill(replace_val)
                    
    #                 self.dataset[key] = data




    def return_data(self):
        """
        Returns self.data
        """
        return self.dataset








