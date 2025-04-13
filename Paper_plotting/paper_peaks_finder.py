# -*- coding: utf-8 -*-
"""
Created on Sat Dec  7 15:02:44 2024

@author: Kian Sartipzadeh
"""


import numpy as np
from scipy.signal import find_peaks

class IonosphericPeakFinder:
    def __init__(self, radar_data):
        """
        Initialize the IonosphericPeakFinder class with nested radar data for multiple days.
        :param radar_data: Nested dictionary with daily radar data.
        """
        self.radar_data = radar_data

    def find_peaks(self, dataset, e_prom, f_prom):
        """
        Find peaks for a single day's dataset.
        :param dataset: Dictionary containing radar data for a single day with keys 'r_time', 'r_h', 'r_param'.
        :return: Arrays of peak altitudes and corresponding parameters.
        """
        r_time = dataset['r_time']      # Arrays with dates and times (M, 6)
        r_h = dataset['r_h'].flatten()  # Flatten altitude points (N,)
        r_param = dataset['r_param']    # Electron density profiles (N, M)
        
        h_peaks = np.full((2, r_param.shape[1]), np.nan)  # Initialize with np.nan
        r_peaks = np.full((2, r_param.shape[1]), np.nan)  # Initialize with np.nan
        
        # Defining E and F-region altitude ranges
        e_reg = (195 >= r_h) & (92 < r_h)
        f_reg = (325 >= r_h) & (195 < r_h)
        
        for m in range(r_param.shape[1]):
            column = r_param[:, m]
            
            if np.all(np.isnan(column)):
                # If the entire column is NaN, peaks remain np.nan
                continue
            
            # Ignore NaN values when finding peaks
            valid_e_reg = e_reg & ~np.isnan(column)
            valid_f_reg = f_reg & ~np.isnan(column)
            
            # Finding E-region peaks
            if np.any(valid_e_reg):
                e_peaks, e_properties = find_peaks(
                    column[valid_e_reg], prominence=e_prom  # Adjust prominence threshold as needed
                )
                if e_peaks.size > 0:
                    e_peak_index = e_properties['prominences'].argmax()
                    h_peaks[0, m] = r_h[valid_e_reg][e_peaks][e_peak_index]
                    r_peaks[0, m] = column[valid_e_reg][e_peaks][e_peak_index]
            
            # Finding F-region peaks
            if np.any(valid_f_reg):
                f_peaks, f_properties = find_peaks(
                    column[valid_f_reg], prominence=f_prom  # Adjust prominence threshold as needed
                )
                if f_peaks.size > 0:
                    f_peak_index = f_properties['prominences'].argmax()
                    h_peaks[1, m] = r_h[valid_f_reg][f_peaks][f_peak_index]
                    r_peaks[1, m] = column[valid_f_reg][f_peaks][f_peak_index]
        
        return h_peaks, r_peaks

    def get_peaks(self, e_prom=0.1, f_prom=0.1):
        """
        Process the radar data for all days to find peaks and corresponding altitudes.
        :return: New nested dictionary with additional keys for peak altitudes and values.
        """
        processed_data = {}
        
        for date, dataset in self.radar_data.items():
            h_peaks, r_peaks = self.find_peaks(dataset, e_prom, f_prom)
            
            # Copy original data and add new keys for peaks
            processed_data[date] = {
                'r_time': dataset['r_time'],
                'r_h': dataset['r_h'],
                'r_param': dataset['r_param'],
                'r_h_peak': h_peaks,
                'r_param_peak': r_peaks
            }
            
        return processed_data





class IonosphericProfileInterpolator:
    def __init__(self, radar_data):
        """
        Initialize with nested radar data.
        :param radar_data: Nested dictionary where each key represents a day.
            Each day's data must include the following keys with the corresponding shapes:
                - r_time      : numpy array with shape (M, 6)
                - r_h         : numpy array with shape (N, 1)
                - r_param     : numpy array with shape (N, M)
                - r_error     : numpy array with shape (N, M)
                - num_avg_samp: numpy array with shape (M,)
        """
        self.radar_data = radar_data

    def _interpolate_array(self, data, n):
        """
        Interpolates a one-dimensional array to have len(data) * n points using linear interpolation.
        This method uses the original array indices (0 to len(data)-1) as the independent variable.
        It handles NaNs by only interpolating between valid (non-NaN) points.
        
        :param data: 1D numpy array.
        :param n: Multiplicative factor for increasing the number of data points.
        :return: Interpolated 1D numpy array of length len(data)*n.
        """
        orig_len = len(data)
        x_orig = np.arange(orig_len)
        new_len = orig_len * n
        x_new = np.linspace(0, orig_len - 1, new_len)

        # If all points are NaN, return an array of NaNs
        if np.all(np.isnan(data)):
            return np.full(new_len, np.nan)

        # Use only valid (non-NaN) points
        valid = ~np.isnan(data)
        if np.sum(valid) < 2:
            # If there's fewer than 2 valid points, fill new array with that value (or NaN)
            fill_value = data[valid][0] if np.any(valid) else np.nan
            return np.full(new_len, fill_value)

        return np.interp(x_new, x_orig[valid], data[valid])

    def interpolate_data(self, n):
        """
        Interpolates the data along the N dimension for the keys 'r_h', 'r_param', and 'r_error'.
        The original dictionary structure is preserved; only the arrays for these keys are updated.
        The new arrays will have their first dimension increased by a factor of n (i.e., new N = N * n).
        
        :param n: Integer multiplier (n >= 1) for the number of vertical data points.
        :return: New nested dictionary with updated interpolated arrays replacing the original ones.
        """
        new_data = {}
        for date, dataset in self.radar_data.items():
            # Make a copy of the dataset so as not to alter the original.
            new_dataset = dataset.copy()

            # --- Interpolate r_h ---
            # r_h has shape (N, 1), so we flatten it for interpolation,
            # then reshape back to (N*n, 1).
            r_h = dataset['r_h'].flatten()  # shape (N,)
            r_h_interp = self._interpolate_array(r_h, n).reshape(-1, 1)
            new_dataset['r_h'] = r_h_interp

            # --- Interpolate r_param ---
            # r_param has shape (N, M): interpolate along the N axis for each of M columns.
            r_param = dataset['r_param']
            N, M = r_param.shape
            new_N = N * n
            r_param_interp = np.empty((new_N, M))
            r_param_interp[:] = np.nan
            for col in range(M):
                r_param_interp[:, col] = self._interpolate_array(r_param[:, col], n)
            new_dataset['r_param'] = r_param_interp


            # Copy over the updated dataset for the current date.
            new_data[date] = new_dataset
        
        return new_data

# --- Example Usage ---
# Suppose 'my_radar_data' is your original dictionary structured like:
# {
#     '2025-04-10': {
#         'r_time': time_array,
#         'r_h': altitude_array (27 points),
#         'r_param': electron_density_profiles (27 x M)
#     },
#     ...
# }
#
# Create an instance of the interpolator and get the interpolated data:
#
# interpolator = IonosphericProfileInterpolator(my_radar_data)
# new_radar_data = interpolator.interpolate_data(n=3)  # This will return data with 27*3 points.
#
# The new dictionary structure for each day retains the original keys and adds
# 'r_h_interp' and 'r_param_interp'.








