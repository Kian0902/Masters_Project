# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 15:10:52 2024

@author: Kian Sartipzadeh
"""
import os
import pickle
import warnings
import numpy as np
import matplotlib.pyplot as plt

from scipy.io import loadmat
from scipy.interpolate import interp1d
from scipy.signal import medfilt
from data_utils import inspect_dict


class EISCATDataFilter:
    """
    Class for clipping and filtering dict data.
    """
    def __init__(self, dataset: dict, filt_range: bool=False, filt_nan: bool=False, filt_interpolate: bool=False, filt_outlier: bool=False):
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
        self.apply_interpolate_filter = filt_interpolate
        self.apply_outlier_filter = filt_outlier
        
        
    def _get_reference_alt(self, ref_alt: int):
        reference_altitudes = None
        for date, data in self.dataset.items():
            if len(data["r_h"]) == ref_alt:
                reference_altitudes = data["r_h"].flatten()
                break
        
        if reference_altitudes is None:
            raise ValueError("No day with 27 altitude measurements found")
        else:
            return reference_altitudes
    
    
    def filter_interpolate(self, reference_alt: int=27):
        
        reference_altitudes = self._get_reference_alt(reference_alt)
        dates_to_remove = []

        # Iterate over each day in X_avg
        for date in list(self.dataset.keys()):
            data = self.dataset[date]
            r_h = data["r_h"].flatten()   # Original altitudes (shape: (N,))
            r_param = data["r_param"]     # Electron density measurements (shape: (N, M))
            r_error = data["r_error"]     # Error values (shape: (N, M))
            
            # Check if current data is invalid
            if r_param.size == 0 or r_param.shape[0] == 0 or r_param.shape[1] == 0:
                print(f"{date} has invalid r_param shape {r_param.shape} before interpolation. Removing.")
                warnings.warn(f"Data for {date} is empty (contains arrays with length 0) and will be removed.")
                dates_to_remove.append(date)
                continue
            
            # Interpolating electron density (r_param) to reference_altitudes
            interpolated_r_param = np.zeros((len(reference_altitudes), r_param.shape[1]))
            interpolated_r_error = np.zeros((len(reference_altitudes), r_error.shape[1]))
            
            for i in range(r_param.shape[1]):
                # Interpolating each column of r_param and r_error
                interpolated_r_param[:, i] = np.interp(reference_altitudes, r_h, r_param[:, i])
                interpolated_r_error[:, i] = np.interp(reference_altitudes, r_h, r_error[:, i])
            
            # Check if interpolated data is invalid
            if interpolated_r_param.size == 0 or interpolated_r_param.shape[0] == 0 or interpolated_r_param.shape[1] == 0:
                print(f"{date} has invalid interpolated r_param shape {interpolated_r_param.shape}. Removing.")
                warnings.warn(f"Interpolated data for {date} is empty and will be removed.")
                dates_to_remove.append(date)
            else:
                # Update dataset with interpolated data
                self.dataset[date] = {
                    "r_time": data["r_time"],
                    "r_h": reference_altitudes.reshape(-1, 1),
                    "r_param": interpolated_r_param,
                    "r_error": interpolated_r_error,
                }
        
        # Remove all invalid dates
        for date in dates_to_remove:
            del self.dataset[date]
    
    
    def batch_filtering(self, min_val=90, max_val=400, dataset_outliers=None, num_of_ref_alt=None, filter_size=3, plot_after_each_day=False):
        """
        Function for applying the filtering to the entire dataset by looping
        through the global keys (days).
        """
        
        # Loop through each day
        for key in list(self.dataset.keys()):
            
            # Store the original data separately for plotting
            original_data = {k: v.copy() for k, v in self.dataset[key].items()}
            
            # Apply range filter
            if self.apply_range_filter:
                self.dataset[key] = self.filter_range(self.dataset[key], 'r_h', min_val, max_val)
            
            # Apply NaN filter
            if self.apply_nan_filter:
                self.dataset[key] = self.filter_nan(self.dataset[key])
            
            # Apply outlier filter
            if self.apply_outlier_filter and dataset_outliers is not None:
                self.dataset[key] = self.filter_outlier(self.dataset[key], dataset_outliers[key], filter_size)
            
            # Check if data is invalid after filtering
            data = self.dataset.get(key)
            if data is None:
                continue  # Already removed
            
            # Check for invalid r_param after all filters
            r_param = data['r_param']
            if r_param.size == 0 or r_param.shape[0] == 0 or r_param.shape[1] == 0:
                print(f"Removing {key} due to empty/invalid r_param after batch filtering.")
                del self.dataset[key]
                continue
            
            # Plotting after filtering for each day if requested
            if plot_after_each_day:
                self.plot_data(original_data, self.dataset[key], dataset_outliers[key], key)
        
        # Apply interpolation filter if needed
        if self.apply_interpolate_filter:
            self.filter_interpolate(num_of_ref_alt)
    
    
    
    
    def filter_range(self, data: dict, key: str, min_val: float, max_val: float):
        # Existing implementation
        mask = np.any((data[key] >= min_val) & (data[key] <= max_val), axis=1)
        for k in list(data.keys())[1:]:
            data[k] = data[k][mask,:]
        return data
    
    def filter_nan(self, data: dict, replace_val=111):
        # Existing implementation
        for key in list(data.keys())[1:]:
            X = data[key]
            if np.isnan(X).any():
                nan_mask = np.isnan(X)
                if np.sum(nan_mask) > (X.size / 2):
                    X.fill(replace_val)
                else:
                    x = np.arange(X.shape[0])
                    for j in range(X.shape[1]):
                        col = X[:, j]
                        valid_mask = ~nan_mask[:, j]
                        if np.any(valid_mask):
                            interp_func = interp1d(x[valid_mask], col[valid_mask], kind='linear', fill_value="extrapolate")
                            col[~valid_mask] = interp_func(x[~valid_mask])
                        else:
                            X[:, j].fill(replace_val)
                    data[key] = X
        return data
    
    def filter_outlier(self, data: dict, outlier_indices, filter_size: int=3, save_plot: bool=False):
        # Existing implementation
        if outlier_indices.size == 0:
            return data
        pad_size = filter_size // 2
        for key in list(data.keys())[2:]:
            X = data[key]
            X_padded = np.pad(X, ((pad_size, pad_size), (0, 0)), mode='edge')
            X_medfilt = medfilt(X_padded, kernel_size=filter_size)
            X_medfilt = X_medfilt[pad_size:-pad_size, :]
            for idx in outlier_indices:
                X[:, idx] = X_medfilt[:, idx]
            data[key] = X
        return data
    
    def plot_data(self, original_data, filtered_data, outlier_indices, date):
        """
        Plots the original and filtered data for specific outlier indices in a grid of subplots.

        This function generates a 2xN grid of plots, where N is the number of outlier indices provided.
        The top row displays the original data, and the bottom row displays the filtered data. Each column
        corresponds to a specific outlier index, showing how the data around that index compares between 
        the original and filtered datasets.
        
        Input (type)                | DESCRIPTION
        ------------------------------------------------
        original_data (dict)        | A dictionary containing the original dataset. 
                                    | It must include 'r_param' (electron density) and 'r_h' (altitude) as keys.
        filtered_data (dict)        | A dictionary containing the filtered dataset. 
                                    | It must include 'r_param' (electron density) and 'r_h' (altitude) as keys.
        outlier_indices (list[int]) | A list of indices corresponding to outliers in the dataset. 
                                    | These indices will determine which data points are plotted.
        """
        
        
        # Check if there are no outliers
        if len(outlier_indices) == 0:
            print("No outliers to plot.")
            return
            
        # Determine the number of columns for the plot grid
        num_plots = len(outlier_indices)
        
        # Create a plotting grid with 2 rows and num_plots columns
        fig, axes = plt.subplots(2, num_plots, figsize=(num_plots * 4, 8))
        
        # If there's only one outlier, adjust axes to be a 2D list for consistency
        if num_plots == 1:
            axes = np.array([[axes[0]], [axes[1]]])
        
        
        # Plot the original data (outliers)
        X_orig = original_data['r_param']
        X_filt = filtered_data['r_param']
        
        
        # Loop over each outlier index to plot
        for i, idx in enumerate(outlier_indices):
            
            axes[0, i].plot(X_orig[:, idx - 1], original_data['r_h'].flatten(), label=f"Data at {idx-1}", color="C0", zorder=0)
            axes[0, i].plot(X_orig[:, idx], original_data['r_h'].flatten(), label="Outlier", color="C1", zorder=2)
            axes[0, i].plot(X_orig[:, idx + 1], original_data['r_h'].flatten(), label=f"Data at {idx+1}", color="C2", zorder=1)
            axes[0, i].set_title(f"Original Data (Outlier at {idx})")
            axes[0, i].set_ylabel("Altitude")
            axes[0, i].set_xlabel("Electron Density")
            axes[0, i].set_xscale('log')
    
            # Plot the filtered data
            axes[1, i].plot(X_filt[:, idx - 1], filtered_data['r_h'].flatten(), label=f"Data at {idx-1}", color="C0", zorder=0)
            axes[1, i].plot(X_filt[:, idx], filtered_data['r_h'].flatten(), label="Filtered", color="C1", zorder=2)
            axes[1, i].plot(X_filt[:, idx + 1], filtered_data['r_h'].flatten(), label=f"Data at {idx+1}", color="C2", zorder=1)
            axes[1, i].set_title(f"Filtered Data (at {idx})")
            axes[1, i].set_ylabel("Altitude")
            axes[1, i].set_xlabel("Electron Density")
            axes[1, i].set_xscale('log')
    
            # Show legends
            axes[0, i].legend()
            axes[1, i].legend()
    
        # Add a global title for the entire figure
        fig.suptitle(f'Original vs Filtered Data.\nDate {date}', fontsize=16)
        plt.tight_layout()
        plt.subplots_adjust(top=0.87)  # Adjust the top to make room for the suptitle
        
        plt.show()
    
    def return_data(self):
        return self.dataset




























