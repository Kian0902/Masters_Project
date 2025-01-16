# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 18:14:11 2025

@author: Kian Sartipzadeh
"""


import numpy as np
from scipy.interpolate import interp1d
 


def filter_range(data: dict, key: str, min_val: float, max_val: float):
    
    # Mask for filtered values. False outside and True inside interval
    mask = np.any((data[key] >= min_val) & (data[key] <= max_val), axis=1)
    
    
    # Applying mask to all keys except "r_time"
    for key in list(data.keys())[1:]:
        data[key] = data[key][mask,:]
    
    return data


def interpolate_data(data: dict, r_h):
    
    r_new = r_h.flatten()
    M = data['r_param'].shape[1]  # Num of measurement
    
    # Initialize the interpolated array with NaNs
    r_param_new = np.full((len(r_new), M), np.nan)
    
    
    # Loop over each measurement
    for m in range(M):
        y = data['r_param'][:, m]
        x = data['r_h'].flatten()
        
        
        # Remove NaNs
        valid_mask = ~np.isnan(y)
        x_valid = x[valid_mask]
        y_valid = y[valid_mask]
        
        
        # If the are enough valid values to interpolate
        if len(x_valid) >= 2:
            # Create interpolation function
            interp_func = interp1d(x_valid, y_valid, kind='linear', bounds_error=False, fill_value=np.nan)
            
            # Interpolate to actual_altitude
            y_new = interp_func(r_new)
            
            # Store the interpolated values
            r_param_new[:, m] = y_new
        
        
        # If the are not enough valid values to interpolate
        else:
            # Not enough data to interpolate
            print(f"Measurement {m} skipped due to insufficient data.")
    
    
    # Assign the arrays to the nested dictionary
    X_new = {"r_time": data["r_time"],
             "r_h": r_h,
             "r_param": r_param_new
             }

    return X_new










