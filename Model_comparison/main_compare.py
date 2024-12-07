# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 16:50:34 2024

@author: Kian Sartipzadeh
"""

import numpy as np

from torchvision import transforms
from storing_dataset import Matching3Pairs, Store3Dataset


from eval_utils import save_dict, load_dict, apply_log10, add_key_from_dict_to_dict, convert_pred_to_dict, convert_ionograms_to_dict, from_csv_to_numpy, from_strings_to_array, from_strings_to_datetime, filter_artist_times, inspect_dict
from eval_plotting import RadarPlotter
from eval_predict import apply_model
from eval_peaks import IonosphericPeakFinder
from hnn_model import CombinedNetwork



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
X_art = add_key_from_dict_to_dict(Eiscat_support, X_art)


# X_eis = apply_log10(X_eis)


X_ion = convert_ionograms_to_dict(ion, X_eis)


eis_peak_finder = IonosphericPeakFinder(X_eis)
X_eis = eis_peak_finder.get_peaks()

hnn_peak_finder = IonosphericPeakFinder(X_hnn)
X_hnn = hnn_peak_finder.get_peaks()

art_peak_finder = IonosphericPeakFinder(X_art)
X_art = art_peak_finder.get_peaks()






# Times must be in the format yyyymmdd_hhmm
selected_dates_and_times = ["20190105_1045", "20190105_1400", "20190105_2030",
                            "20191215_2030", "20191215_2115", "20191215_2245",
                            "20200227_0145", "20200227_1200", "20200227_1615"]

# selected_dates_and_times = ["20190105_1045", "20190105_1800", "20190105_2030"]
selected_datetimes = from_strings_to_datetime(selected_dates_and_times)



for day in X_hnn:
    radar_plotter = RadarPlotter(X_eis[day], X_hnn[day], X_art[day], X_ion[day])
    # radar_plotter.plot_all_peaks()
    radar_plotter.plot_compare_all_peaks()
    # radar_plotter.plot_compare_all_peak_regions()
    # break
    # radar_plotter.select_measurements_by_datetime(selected_datetimes)
    # radar_plotter.plot_compare_all()
    # radar_plotter.plot_compare_closest()
    # radar_plotter.plot_selected_measurements()
    # radar_plotter.plot_error_profiles()
    # radar_plotter.plot_ionogram_measurements_and_errors()
    




















