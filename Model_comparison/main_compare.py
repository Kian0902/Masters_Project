# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 16:50:34 2024

@author: Kian Sartipzadeh
"""


import pickle
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
from storing_dataset import Matching3Pairs, Store3Dataset


from eval_utils import save_dict, load_dict, apply_log10, add_key_from_dict_to_dict, convert_pred_to_dict, convert_ionograms_to_dict, from_csv_to_numpy, from_strings_to_array, from_strings_to_datetime, filter_artist_times, inspect_dict
from eval_plotting import plot_compare, plot_compare_all, plot_compare_r2, plot_results, RadarPlotter
from eval_predict import apply_model
from hnn_model import CombinedNetwork
from sklearn.metrics import r2_score
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
        r_time = dataset['r_time']      # Arrays with dates and times (M, 6)
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



# class IonosphericPeakFinder:
#     def __init__(self, dataset):
#         """
#         Initialize the IonosphericPeakFinder class with daily radar data.
#         :param daily_data: Dictionary containing daily radar data with keys 'r_time', 'r_h', 'r_param'.
#         """
#         self.r_time = dataset['r_time']      # Arrays with dates and times (M, 6)
#         self.r_h = dataset['r_h'].flatten()  # Flatten altitude points (N,)
#         self.r_param = dataset['r_param']    # Electron density profiles (N, M)
    
    
#     def plot(self, ne, ne_peaks, z, z_peaks):
        
        
#         fig, ax = plt.subplots(figsize=(6, 6))
        
#         ax.plot(ne, z, color="C0")
#         ax.scatter(ne_peaks[0], z_peaks[0], label="E", color="C1")
#         ax.scatter(ne_peaks[1], z_peaks[1], label="F", color="red")
#         ax.grid(True)
#         plt.show()
        
#         # plt.scatter(x, y)
    
#     # def plot_filt(self, x, x_filt, z):
#     #     fig, ax = plt.subplots(figsize=(6, 6))
        
#     #     ax.plot(x, z, color="C0")
#     #     ax.plot(x_filt, z, color="C1")
#     #     ax.grid(True)
#     #     plt.show()
    
    
    
#     # def filter_savgol(self):
        
#     #     for m in range(self.r_param.shape[1]):
        
#     #         r_param_savgol = savgol_filter(self.r_param[:, m], window_length=5, polyorder=2)
#     #         r_param = self.r_param[:, m]
            
            
#     #         self.plot_filt(r_param, r_param_savgol, self.r_h)
            
#     #         # break
    
#     def find_peaks(self):

#         # Defining E and F-region altitudes
#         e_reg = (160 >= self.r_h) & (90 < self.r_h)
#         f_reg = (325 >= self.r_h) & (160 < self.r_h)
        
#         for m in range(self.r_param.shape[1]):
        
#             # Finding E and F-region peaks
#             e_peaks, e_properties = find_peaks(self.r_param[e_reg, m], prominence=True)
#             f_peaks, f_properties = find_peaks(self.r_param[f_reg, m], prominence=True)
            
            
#             # Handling E-region
#             if e_peaks.size > 0:
#                 e_peak_index = e_properties['prominences'].argmax()
#                 e_peaks_h = self.r_h[e_reg][e_peaks][e_peak_index]
#                 e_peaks_param = self.r_param[e_reg, m][e_peaks][e_peak_index]
#             else:
#                 e_peaks_h = self.r_h[e_reg][self.r_param[e_reg, m].argmax()]
#                 e_peaks_param = self.r_param[e_reg, m].max()
            
            
            
#             # Handling F-region
#             if f_peaks.size > 0:
#                 f_peak_index = f_properties['prominences'].argmax()
#                 f_peaks_h = self.r_h[f_reg][f_peaks][f_peak_index]
#                 f_peaks_param = self.r_param[f_reg, m][f_peaks][f_peak_index]
#             else:
#                 f_peaks_h = self.r_h[f_reg][self.r_param[f_reg, m].argmax()]
#                 f_peaks_param = self.r_param[f_reg, m].max()

            
#             h_peaks  = np.array([e_peaks_h, f_peaks_h])
#             r_peaks = np.array([e_peaks_param, f_peaks_param])
            
#             self.plot(self.r_param[:, m], r_peaks, self.r_h, h_peaks)
            



            # break
        # return e_peaks_z, e_peaks_ne, f_peaks_z, f_peaks_ne
    


# Test data folder names
test_ionogram_folder = "testing_data/test_ionogram_folder"
test_radar_folder = "testing_data/test_eiscat_folder"        # These are the true data
test_sp19_folder = "testing_data/test_geophys_folder"





# Initializing class for matching pairs
Pairs = Matching3Pairs(test_ionogram_folder, test_radar_folder, test_sp19_folder)


# Returning matching sample pairs
rad, ion, sp, radar_times = Pairs.find_pairs(return_date=True)
r_t = from_strings_to_datetime(radar_times)
r_times = from_strings_to_array(radar_times)

rad = np.abs(rad)
rad[rad < 1e5] = 1e6


# Storing the sample pairs
A = Store3Dataset(ion, sp, np.log10(rad), transforms.Compose([transforms.ToTensor()]))

# Path to trained weights
weights_path = 'HNN_v1_best_weights.pth'



X_pred = apply_model(A, CombinedNetwork(), weights_path)
X_hnn = convert_pred_to_dict(r_t, r_times, X_pred)


X_true = from_csv_to_numpy(test_radar_folder)[0]
X_eis = convert_pred_to_dict(r_t, r_times, X_true)


X_art = load_dict("processed_artist_test_days.pkl")
X_art = filter_artist_times(X_eis, X_hnn, X_art)


# Adding 'r_h' from eiscat to all dicts
Eiscat_support = load_dict("X_avg_test_data")
X_eis = add_key_from_dict_to_dict(Eiscat_support, X_eis)
X_hnn = add_key_from_dict_to_dict(Eiscat_support, X_hnn)
# X_art = add_key_from_dict_to_dict(Eiscat_support, X_art)


# X_eis = apply_log10(X_eis)


X_ion = convert_ionograms_to_dict(ion, X_eis)


eis_peak_finder = IonosphericPeakFinder(X_eis)
X_eis = eis_peak_finder.gets_peaks()

hnn_peak_finder = IonosphericPeakFinder(X_hnn)
X_hnn = hnn_peak_finder.gets_peaks()

art_peak_finder = IonosphericPeakFinder(X_art)
X_art = art_peak_finder.gets_peaks()







inspect_dict(X_art)


# # Times must be in the format yyyymmdd_hhmm
# selected_dates_and_times = ["20190105_1045", "20190105_1400", "20190105_2030",
#                             "20191215_2030", "20191215_2115", "20191215_2245",
#                             "20200227_0145", "20200227_1200", "20200227_1615"]

# # selected_dates_and_times = ["20190105_1045", "20190105_1800", "20190105_2030"]
# selected_datetimes = from_strings_to_datetime(selected_dates_and_times)



# for day in X_hnn:
#     radar_plotter = RadarPlotter(X_eis[day], X_hnn[day], X_art_new[day], X_ion[day])
#     radar_plotter.select_measurements_by_datetime(selected_datetimes)
#     radar_plotter.plot_compare_all()
#     radar_plotter.plot_compare_closest()
#     radar_plotter.plot_selected_measurements()
#     radar_plotter.plot_error_profiles()
#     radar_plotter.plot_ionogram_measurements_and_errors()
    




















