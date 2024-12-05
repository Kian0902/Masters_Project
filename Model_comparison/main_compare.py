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


from eval_utils import save_dict, load_dict, convert_pred_to_dict, from_csv_to_numpy, from_strings_to_array, from_strings_to_datetime, filter_artist_times, inspect_dict
from eval_plotting import plot_compare, plot_compare_all, plot_compare_r2, plot_results, RadarPlotter
from eval_predict import apply_model
from hnn_model import CombinedNetwork
from sklearn.metrics import r2_score


    


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

# Predictions of ne
X_pred = apply_model(A, CombinedNetwork(), weights_path)

# Convert to dict with days
X_hnn = convert_pred_to_dict(r_t, r_times, X_pred)


X_true = from_csv_to_numpy(test_radar_folder)[0]
X_eis = convert_pred_to_dict(r_t, r_times, X_true)
X_art = load_dict("processed_artist_test_days.pkl")
Eiscat_support = load_dict("X_avg_test_data")


X_art_new = filter_artist_times(X_eis, X_hnn, X_art)




# for day in X_hnn:
#     plot_compare_all(Eiscat_support[day], X_eis[day], X_hnn[day], X_art_new[day])
    


# Times must be in the format yyyymmdd_hhmm
selected_dates_and_times = ["20190105_1045", "20190105_1800", "20190105_2030",
                            "20191215_2030", "20191215_2115", "20191215_2245",
                            "20200227_0145", "20200227_1200", "20200227_2030",]
selected_datetimes = from_strings_to_datetime(selected_dates_and_times)


for day in X_hnn:
    radar_plotter = RadarPlotter(Eiscat_support[day], X_eis[day], X_hnn[day], X_art_new[day])
    radar_plotter.select_measurements_by_datetime(selected_datetimes)  # Select 3 measurements
    radar_plotter.plot_compare_all()  # Plot all radar data with selected measurements highlighted
    radar_plotter.plot_selected_measurements()  # Plot the selected measurements
    






















