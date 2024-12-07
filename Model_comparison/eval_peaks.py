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

    def find_peaks(self, dataset):
        """
        Find peaks for a single day's dataset.
        :param dataset: Dictionary containing radar data for a single day with keys 'r_time', 'r_h', 'r_param'.
        :return: Arrays of peak altitudes and corresponding parameters.
        """
        
        r_h = dataset['r_h'].flatten()  # Flatten altitude points (N,)
        r_param = dataset['r_param']    # Electron density profiles (N, M)

        h_peaks = np.zeros((2, r_param.shape[1]))  # Shape (P, M), where P=2 for E and F peaks
        r_peaks = np.zeros((2, r_param.shape[1]))

        # Defining E and F-region altitude ranges
        e_reg = (160 >= r_h) & (90 < r_h)
        f_reg = (325 >= r_h) & (160 < r_h)

        for m in range(r_param.shape[1]):
            # Finding E and F-region peaks
            e_peaks, e_properties = find_peaks(r_param[e_reg, m], prominence=True)
            f_peaks, f_properties = find_peaks(r_param[f_reg, m], prominence=True)

            # Handling E-region
            if e_peaks.size > 0:
                e_peak_index = e_properties['prominences'].argmax()
                h_peaks[0, m] = r_h[e_reg][e_peaks][e_peak_index]
                r_peaks[0, m] = r_param[e_reg, m][e_peaks][e_peak_index]
            else:
                h_peaks[0, m] = r_h[e_reg][r_param[e_reg, m].argmax()]
                r_peaks[0, m] = r_param[e_reg, m].max()

            # Handling F-region
            if f_peaks.size > 0:
                f_peak_index = f_properties['prominences'].argmax()
                h_peaks[1, m] = r_h[f_reg][f_peaks][f_peak_index]
                r_peaks[1, m] = r_param[f_reg, m][f_peaks][f_peak_index]
            else:
                h_peaks[1, m] = r_h[f_reg][r_param[f_reg, m].argmax()]
                r_peaks[1, m] = r_param[f_reg, m].max()

        return h_peaks, r_peaks

    def gets_peaks(self):
        """
        Process the radar data for all days to find peaks and corresponding altitudes.
        :return: New nested dictionary with additional keys for peak altitudes and values.
        """
        processed_data = {}

        for date, dataset in self.radar_data.items():
            h_peaks, r_peaks = self.find_peaks(dataset)

            # Copy original data and add new keys for peaks
            processed_data[date] = {
                'r_time': dataset['r_time'],
                'r_h': dataset['r_h'],
                'r_param': dataset['r_param'],
                'r_h_peak': h_peaks,
                'r_param_peak': r_peaks
            }

        return processed_data











