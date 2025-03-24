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
        e_reg = (190 >= r_h) & (92 < r_h)
        f_reg = (325 >= r_h) & (190 < r_h)
        
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











