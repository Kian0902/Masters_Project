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
        Initializes the peak finder class.
        
        Parameters:
            radar_data (dict): Nested dictionary with keys as dates and values as dictionaries containing 
                              'r_time', 'r_h', and 'r_param'.
        """
        self.radar_data = radar_data

    def find_peaks(self, r_h, r_param, prominence=1e10):
        """
        Finds peaks for a single day of radar data.
        
        Parameters:
            r_h (numpy array): Array of altitude points (N, 1).
            r_param (numpy array): Array of radar parameters (N, M).
            prominence (float): Prominence for peak detection (default is 1e10).
        
        Returns:
            list of dict: A list containing dictionaries with keys 'E_region', 'F_region', and 'E+F_region',
                          each containing the peak altitude and value for each measurement.
        """
        results = []
        for measurement_idx in range(r_param.shape[1]):
            profile = r_param[:, measurement_idx]
            peaks, properties = find_peaks(profile, prominence=prominence)

            e_region_peak = None
            f_region_peak = None
            e_f_region_peak = None

            for peak_idx in peaks:
                altitude = r_h[peak_idx, 0]
                if 90 <= altitude <= 150:  # E-region altitude range
                    e_region_peak = (altitude, profile[peak_idx])
                elif 150 < altitude <= 400:  # F-region altitude range
                    f_region_peak = (altitude, profile[peak_idx])
                if 90 <= altitude <= 400:  # Combined E+F-region range
                    if e_f_region_peak is None or profile[peak_idx] > e_f_region_peak[1]:
                        e_f_region_peak = (altitude, profile[peak_idx])

            results.append({
                'E_region': e_region_peak,
                'F_region': f_region_peak,
                'E+F_region': e_f_region_peak,
            })

        return results

    def get_peaks(self):
        """
        Analyze all radar data to find peaks for each day and measurement.
        
        Returns:
            dict: Nested dictionary with dates as keys and values as lists of peak information for each measurement.
        """
        all_peaks = {}
        for date, data in self.radar_data.items():
            r_h = data['r_h']
            r_param = data['r_param']
            peaks = self.find_peaks(r_h, r_param)
            all_peaks[date] = peaks

        return all_peaks




















