# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 16:50:34 2024

@author: Kian Sartipzadeh
"""

import numpy as np

from eval_utils import save_dict, load_dict, apply_log10, revert_log10, add_key_from_dict_to_dict, convert_pred_to_dict, convert_ionograms_to_dict, from_csv_to_numpy, from_array_to_datetime, from_strings_to_array, from_strings_to_datetime, filter_artist_times, inspect_dict
from eval_plotting import RadarPlotter
from eval_interactive import RadarInteractive
from eval_peaks import IonosphericPeakFinder



s = "X"
# s = "hyp"

# Importing model prediction and corresponding eiscat, artist and ionograms
X_eis = load_dict("X_eis.pkl")
X_kian = load_dict(s+"_kian.pkl")
X_art = load_dict("X_art.pkl")
X_iri = load_dict("X_iri.pkl")
X_ion = load_dict(s+"_ion.pkl")
X_geo = load_dict("X_geo.pkl")

X_iri = filter_artist_times(X_eis, X_kian, X_iri)



# day = '2019-1-5'
# day = '2019-12-15'
day = '2020-2-27'

interactive_plotter = RadarInteractive(X_eis[day], X_kian[day], X_art[day], X_iri[day], X_ion[day], X_geo[day])
interactive_plotter.plot_interactive()
# interactive_plotter.plot_proximity()




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

# radar_plotter = RadarPlotter(X_eis[day], X_kian[day], X_art[day], X_ion[day])

# print(X_eis[day]['r_h'])


# radar_plotter.select_measurements_by_datetime(selected_datetimes)
# radar_plotter.plot_compare_all(selected_measurements=True)
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












