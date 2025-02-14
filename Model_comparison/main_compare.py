# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 16:50:34 2024

@author: Kian Sartipzadeh
"""

import numpy as np

from eval_utils import *
# from eval_plotting import RadarPlotter
from eval_paper_plotting import PaperPlotter, AdvancedPaperPlotter
from eval_interactive import RadarInteractive
from eval_peaks import IonosphericPeakFinder
from eval_utils import inspect_dict


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, Normalize
from matplotlib.dates import DateFormatter
from matplotlib.patches import Patch
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
from matplotlib.widgets import Cursor
from datetime import datetime

from eval_utils import from_array_to_datetime, get_altitude_r2_score, get_measurements_r2_score, get_altitude_r2_score_nans, get_measurements_r2_score_nans
import random
from scipy.stats import linregress
import seaborn as sns
# For all plots: sns.set(style="dark", context=None, palette=None)
# For single plot: with sns.axes_style("dark"):

from sklearn.metrics import r2_score

from matplotlib.ticker import FuncFormatter

# Define a custom formatter for the x-axis ticks
def format_ticks(x, _):
    if x.is_integer():
        return f'{int(x)}'  # Convert integers to display without decimals
    return f'{x}'  # Keep floats as is

formatter = FuncFormatter(format_ticks)






def align_artist_with_eiscat(eiscat_dict, artist_dict):
    """
    Aligns the Artist 4.5 dictionary with the EISCAT UHF dictionary by adding NaN-filled columns
    for missing timestamps in 'r_param'.
    
    Parameters:
    eiscat_dict (dict): Dictionary containing EISCAT UHF data.
    artist_dict (dict): Dictionary containing Artist 4.5 data.
    
    Returns:
    dict: Updated Artist 4.5 dictionary with aligned 'r_param' shapes.
    """
    updated_artist_dict = {}
    
    for date in eiscat_dict:
        if date in artist_dict:
            eiscat_times = eiscat_dict[date]['r_time']  # Shape (M, 6)
            artist_times = artist_dict[date]['r_time']  # Shape (K, 6)
            eiscat_h = eiscat_dict[date]['r_h']         # Shape (N, 1)
            artist_h = artist_dict[date]['r_h']         # Shape (N, 1)
            
            # Ensure altitudes match
            if not np.array_equal(eiscat_h, artist_h):
                raise ValueError(f"Altitude values do not match for {date}")
            
            # Find missing timestamps
            missing_indices = []
            for i, time in enumerate(eiscat_times):
                if not any(np.array_equal(time, t) for t in artist_times):
                    missing_indices.append(i)
            
            # Create new aligned 'r_param'
            M = eiscat_times.shape[0]  # Number of timestamps in EISCAT
            K = artist_times.shape[0]  # Number of timestamps in Artist 4.5
            N = eiscat_h.shape[0]      # Number of altitude levels
            
            # Initialize a new 'r_param' with NaNs
            new_r_param = np.full((N, M), np.nan)
            
            # Insert existing Artist 4.5 data into new array at corresponding indices
            matched_indices = [np.where((eiscat_times == t).all(axis=1))[0][0] for t in artist_times]
            new_r_param[:, matched_indices] = artist_dict[date]['r_param']
            
            # Update the dictionary
            updated_artist_dict[date] = {
                'r_time': eiscat_times.copy(),  # Use EISCAT times
                'r_h': artist_h.copy(),         # Keep the same altitudes
                'r_param': new_r_param          # Updated r_param with NaNs for missing times
            }
        else:
            # If the date is missing in Artist 4.5, create a full NaN entry
            M = eiscat_dict[date]['r_time'].shape[0]
            N = eiscat_dict[date]['r_h'].shape[0]
            updated_artist_dict[date] = {
                'r_time': eiscat_dict[date]['r_time'].copy(),
                'r_h': eiscat_dict[date]['r_h'].copy(),
                'r_param': np.full((N, M), np.nan)
            }
    
    return updated_artist_dict


s = "X"
# s = "hyp"

# Importing model prediction and corresponding eiscat, artist and ionograms
X_eis = load_dict("testing_data_shutup_rusland/X_eis.pkl")
X_kian = load_dict("testing_data_shutup_rusland/"+s+"_kian.pkl")
X_art = load_dict("testing_data_shutup_rusland/X_art.pkl")
X_iri = load_dict("testing_data_shutup_rusland/X_iri.pkl")
X_ion = load_dict("testing_data_shutup_rusland/"+s+"_ion.pkl")
X_geo = load_dict("testing_data_shutup_rusland/X_geo.pkl")

X_iri = filter_artist_times(X_eis, X_kian, X_iri)

X_art = align_artist_with_eiscat(X_eis, X_art)





# # for day in ['2020-2-27', '2019-12-15', '2019-1-5']:

# # day = '2019-1-5'
# # day = '2019-12-15'
# # day = '2020-2-27'
# # day = '2012-1-17'


# # paper_plotter = PaperPlotter(X_eis[day], X_kian[day], X_art[day], X_iri[day], X_ion[day], X_geo[day])
# # paper_plotter.plot_compare_ne()
# # paper_plotter.plot_compare_error()



days = ['2020-2-27', '2012-1-17', '2012-1-20']


advances_paper_plotter = AdvancedPaperPlotter(X_eis, X_kian, X_art, X_iri, X_ion, X_geo)
advances_paper_plotter.plot_vertical_compare_3days_ne(days)



# # X_EISCAT, X_KIAN, X_Artist, X_IRI, X_Ionogram, X_GEO
# for day in X_eis:
    # radar_plotter = RadarInteractive(X_eis[day], X_kian[day], X_art[day], X_iri[day], X_ion[day], X_geo[day])
    # radar_plotter.plot_compare_all(selected_measurements=False)
    # radar_plotter.plot_compare_all_r2_window(alt_range=[None, None], time_range=[None, None])
    # radar_plotter.plot_interactive()
    # radar_plotter.plot_compare_r2() 
# radar_plotter.select_measurements_by_datetime(selected_datetimes)
# radar_plotter.plot_compare_all(selected_measurements=True)
# radar_plotter.plot_selected_measurements_std()


























# prominence_values = np.arange(0.001, 0.1, 0.001)

# peak_finder = IonosphericPeakFinder(X_kian)
# best_p, best_s = peak_finder.grid_search(prominence_values, reference_data=X_eis)

# print(best_p, best_s)


# f_prom = 0.01
# e_prom = 0.01



# eis_peak_finder = IonosphericPeakFinder(X_eis)
# X_eis = eis_peak_finder.get_peaks(e_prom, f_prom)

# kian_peak_finder = IonosphericPeakFinder(X_kian)
# X_kian = kian_peak_finder.get_peaks(e_prom, f_prom)

# art_peak_finder = IonosphericPeakFinder(X_art)
# X_art = art_peak_finder.get_peaks(e_prom, f_prom)







# # # Times must be in the format yyyymmdd_hhmm
# selected_dates_and_times = ["20190105_0945", "20190105_1245", "20190105_1830", "20190105_2030",
#                             "20191215_2000", "20191215_2030", "20191215_2115", "20191215_2215",
#                             "20200227_0715", "20200227_1100", "20200227_1330", "20200227_1615"]

# selected_datetimes = from_strings_to_datetime(selected_dates_and_times)



# day = '2019-1-5'
# day = '2019-12-15'
# day = '2020-2-27'

# alt_range, time_range = None, None
# alt_range, time_range = [95, 350], ["20190105_0900", "20190105_1400"]
# alt_range, time_range = [100, 260], ["20191215_2045", "20191215_2300"]
# alt_range, time_range = [130, 400], ["20200227_0545", "20200227_1715"]

# radar_plotter = RadarPlotter(X_eis[day], X_kian[day], X_art[day], X_iri[day], X_ion[day])

# radar_plotter.select_measurements_by_datetime(selected_datetimes)
# radar_plotter.plot_compare_all(selected_measurements=False)
# radar_plotter.plot_ionogram_measurements_and_errors()
# radar_plotter.plot_compare_all_interactive()
# radar_plotter.plot_compare_r2()
# radar_plotter.plot_compare_all(True)
# radar_plotter.plot_compare_all_r2()
# radar_plotter.plot_compare_all_r2_window(alt_range=alt_range, time_range=time_range)
# radar_plotter.plot_chi_squared()

# radar_plotter.plot_compare_all_r2()

# radar_plotter.plot_compare_closest()
# radar_plotter.plot_selected_measurements()
# radar_plotter.plot_selected_measurements_std()
# radar_plotter.plot_error_profiles()
# radar_plotter.plot_ionogram_measurements_and_errors()

# radar_plotter.plot_peaks_all_measurements()
# radar_plotter.plot_compare_all_peak_heights()
# radar_plotter.plot_compare_all_peak_densities()
# radar_plotter.plot_compare_all_peak_altitudes()












