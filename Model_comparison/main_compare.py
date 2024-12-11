# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 16:50:34 2024

@author: Kian Sartipzadeh
"""

import numpy as np

from eval_utils import save_dict, load_dict, apply_log10, revert_log10, add_key_from_dict_to_dict, convert_pred_to_dict, convert_ionograms_to_dict, from_csv_to_numpy, from_strings_to_array, from_strings_to_datetime, filter_artist_times, inspect_dict
from eval_plotting import RadarPlotter
from eval_peaks import IonosphericPeakFinder




# Importing model prediction and corresponding eiscat, artist and ionograms
X_eis = load_dict("X_eis.pkl")
X_kian = load_dict("X_kian.pkl")
X_art = load_dict("X_art.pkl")
X_ion = load_dict("X_ion.pkl")





# eis_peak_finder = IonosphericPeakFinder(X_eis)
# X_eis = eis_peak_finder.get_peaks()

# kian_peak_finder = IonosphericPeakFinder(X_kian)
# X_kian = kian_peak_finder.get_peaks()

# art_peak_finder = IonosphericPeakFinder(X_art)
# X_art = art_peak_finder.get_peaks()







# # # Times must be in the format yyyymmdd_hhmm
selected_dates_and_times = ["20190105_1045", "20190105_1200", "20190105_1400", "20190105_1800", "20190105_2030",
                            "20191215_2030", "20191215_2115", "20191215_2145", "20191215_2215", "20191215_2245",
                            "20200227_0145", "20200227_0900", "20200227_1100", "20200227_1200", "20200227_1615"]


# selected_dates_and_times = ["20190105_2015"]

# selected_dates_and_times = ["20190105_1045", "20190105_1800", "20190105_2030"]
selected_datetimes = from_strings_to_datetime(selected_dates_and_times)



for day in X_kian:
    radar_plotter = RadarPlotter(X_eis[day], X_kian[day], X_art[day], X_ion[day])
    # radar_plotter.plot_compare_all_interactive()
    # radar_plotter.plot_compare_r2()
    radar_plotter.plot_compare_all_r2_window()
    # radar_plotter.plot_error_and_chi2()
    # radar_plotter.select_measurements_by_datetime(selected_datetimes)
    # radar_plotter.plot_compare_all_r2()
    
    # radar_plotter.plot_compare_closest()
    # radar_plotter.plot_selected_measurements()
    # radar_plotter.plot_selected_measurements_std()
    # radar_plotter.plot_error_profiles()
    # radar_plotter.plot_ionogram_measurements_and_errors()
    
    # radar_plotter.plot_all_peaks()
    # radar_plotter.plot_compare_all_peaks()
    # radar_plotter.plot_compare_all_peak_regions()
    # break
















