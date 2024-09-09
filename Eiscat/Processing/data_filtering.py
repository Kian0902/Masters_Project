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
from scipy.signal import medfilt
import matplotlib.pyplot as plt

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
        
        
    
    def batch_filtering(self, min_val=90, max_val=400, dataset_outliers=None, filter_size=3):
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
            if self.apply_outlier_filter and dataset_outliers is not None:
                self.dataset[key] = self.filter_outlier(self.dataset[key], dataset_outliers[key], filter_size)
            
    
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
    
    
    
    def filter_outlier(self, data: dict, outlier_indices, filter_size: int=3):
        """
        This method detects and replaces outliers in radar measurement data
        using a median filter. Padding is applied in the N (altitude) dimension
        to handle edge effects, ensuring the original shape of the data is
        preserved after filtering.
        
        Input (type)            | DESCRIPTION
        ------------------------------------------------
        data (dict)             | Radar measurement data with shape (N, M).
        outlier_indices (array) | Indices of columns containing outliers.
        filter_size (int)       | Median filter kernel size. Default is 3.
        
        
        Return (type)  | DESCRIPTION
        ------------------------------------------------
        data (dict)    | Data with outliers replaced by median filtered values.
        """
        # Check if outlier_indices is empty
        if outlier_indices.size == 0:
            return data
        
        
        pad_size = filter_size // 2  # half of filters floored
        
        
        for key in list(data.keys())[2:]:
            X = data[key]
            
            X_padded = np.pad(X, ((pad_size, pad_size), (0, 0)), mode='edge')

            X_medfilt = medfilt(X_padded, kernel_size = filter_size)

            # Remove the padding after filtering
            X_medfilt = X_medfilt[pad_size:-pad_size, :]

 
            for idx in outlier_indices:
                
                X[:, idx] = X_medfilt[:, idx]
            
            data[key] = X
        
        return data
    
    
    def plot_data(self, outlier_indices):
        # Determine the number of columns for the plot grid
        num_plots = len(outlier_indices)
        
        # Create a plotting grid with 2 rows and num_plots columns
        fig, axes = plt.subplots(2, num_plots, figsize=(num_plots * 4, 8))
        
        # Loop over each outlier index to plot
        for i, idx in enumerate(outlier_indices):
            for key in list(self.dataset.keys())[2:]:
                X = self.dataset[key]
                
                # Plot the original data (outliers)
                axes[0, i].plot(X[:, idx], label=f"Outlier at index {idx}")
                axes[0, i].set_title(f"Original Data (Outlier at {idx})")
                axes[0, i].set_xlabel("Altitude")
                axes[0, i].set_ylabel("Electron Density")
                
                # Plot the filtered data
                X_filtered = self.filter_outlier({key: X}, np.array([idx]))[key]
                axes[1, i].plot(X_filtered[:, idx], label=f"Filtered at index {idx}", color='orange')
                axes[1, i].set_title(f"Filtered Data (at {idx})")
                axes[1, i].set_xlabel("Altitude")
                axes[1, i].set_ylabel("Electron Density")
                
                # Show legends
                axes[0, i].legend()
                axes[1, i].legend()
    
        plt.tight_layout()
        plt.show()

    
    
    def return_data(self):
        """
        Returns self.data
        """
        return self.dataset





























