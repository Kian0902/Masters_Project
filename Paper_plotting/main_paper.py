# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 16:50:34 2024

@author: Kian Sartipzadeh
"""


import numpy as np
from paper_paper_plotting import PaperPlotter
from paper_peaks_finder import IonosphericPeakFinder, IonosphericProfileInterpolator
from paper_utils import save_dict, load_dict, inspect_dict, merge_nested_dict, merge_nested_pred_dict, align_artist_with_eiscat


import numpy as np

def align_X_eis(X_eis, X_ion, X_geo, X_kian):
    """
    Aligns the X_eis dictionary to match the dates and time steps of model dictionaries.
    Additionally, adds the "r_h" key from X_eis to the model dictionaries for common dates.
    
    Parameters:
    - X_eis (dict): Nested dictionary with radar measurements.
    - X_ion (dict): Nested dictionary with ion model estimations. Modified in place to add "r_h".
    - X_geo (dict): Nested dictionary with geo model estimations. Modified in place to add "r_h".
    - X_kian (dict): Nested dictionary with kian model estimations. Modified in place to add "r_h".
    
    Returns:
    - tuple: (new_X_eis, X_ion, X_geo, X_kian)
        - new_X_eis: Aligned X_eis with only common dates and arrays aligned to model time steps.
        - X_ion: Modified ion model dictionary with "r_h" added for common dates.
        - X_geo: Modified geo model dictionary with "r_h" added for common dates.
        - X_kian: Modified kian model dictionary with "r_h" added for common dates.
    """
    # Find dates common to X_eis and all models
    common_dates = set(X_eis.keys()) & set(X_ion.keys()) & set(X_geo.keys()) & set(X_kian.keys())
    
    # Initialize new dictionary for aligned X_eis
    new_X_eis = {}
    
    for date in common_dates:
        # Reference time steps from X_ion (assuming all models have identical r_time)
        ref_time = X_ion[date]["r_time"]  # Shape: (M, 6)
        M = ref_time.shape[0]
        
        # X_eis time steps
        eis_time = X_eis[date]["r_time"]  # Shape: (M_eis, 6)
        
        # Create a mapping from X_eis time tuples to their indices
        eis_time_dict = {tuple(row): j for j, row in enumerate(eis_time)}
        
        # Initialize new arrays with NaN
        new_r_param = np.full((27, M), np.nan)      # Shape: (27, M)
        new_r_error = np.full((27, M), np.nan)      # Shape: (27, M)
        new_num_avg_samp = np.full((M,), np.nan)    # Shape: (M,)
        
        # Fill in data where time steps match
        for i, ref_row in enumerate(ref_time):
            t = tuple(ref_row)
            if t in eis_time_dict:
                j = eis_time_dict[t]
                new_r_param[:, i] = X_eis[date]["r_param"][:, j]
                new_r_error[:, i] = X_eis[date]["r_error"][:, j]
                new_num_avg_samp[i] = X_eis[date]["num_avg_samp"][j]
        
        # Build the new entry for this date in new_X_eis
        new_X_eis[date] = {
            "r_time": ref_time.copy(),          # Use model’s time steps
            "r_h": X_eis[date]["r_h"].copy(),  # Keep original altitudes
            "r_param": new_r_param,             # Aligned parameters
            "r_error": new_r_error,             # Aligned errors
            "num_avg_samp": new_num_avg_samp   # Aligned sample counts
        }
        
        # Add "r_h" to the models for this date
        r_h = X_eis[date]["r_h"].copy()
        X_ion[date]["r_h"] = r_h
        X_geo[date]["r_h"] = r_h
        X_kian[date]["r_h"] = r_h
    
    # Return the aligned X_eis and the modified model dictionaries
    return new_X_eis, X_ion, X_geo, X_kian





X_EISCAT   = load_dict('X_true_eiscat')
X_KIANNET  = load_dict('X_pred_deep_kiannet')
X_IONOCNN  = load_dict('X_pred_ionocnn')
X_GEODMLP  = load_dict('X_pred_geodmlp')
X_ARTIST   = load_dict('X_pred_artist.pkl')
X_ech      = load_dict('X_pred_echaim.pkl') 


X_eis, X_ion, X_geo, X_kian = align_X_eis(X_EISCAT, X_IONOCNN, X_GEODMLP, X_KIANNET)

X_art = align_artist_with_eiscat(X_eis, X_ARTIST)

plotter = PaperPlotter(X_eis, X_kian, X_ion, X_geo, X_art, X_ech)
# plotter.plot_altitude_metrics()
# plotter.plot_metrics_vs_time_compare_all()
# plotter.plot_metrics_vs_time_compare_error_all()
# plotter.plot_metrics_vs_altitude()
# metrics = plotter.plot_kde_altitude(330, True)



# plotter.plot_compare_all()
# plotter.plot_compare_all_error()
# plotter.plot_compare_error_all()
# plotter.plot_best_model()
# plotter.plot_r2_rmse_dist()

# prominence_values = np.arange(0.001, 0.1, 0.001)

# peak_finder = IonosphericPeakFinder(X_kian)
# best_p, best_s = peak_finder.grid_search(prominence_values, reference_data=X_eis)

# print(best_p, best_s)





f_prom = 0.01
e_prom = 0.01

n_inter = 1

eis_interpolate = IonosphericProfileInterpolator(X_eis)
eis_peak_finder = IonosphericPeakFinder(eis_interpolate.interpolate_data(n=n_inter))
X_eis = eis_peak_finder.get_peaks(e_prom, f_prom)

kian_interpolate = IonosphericProfileInterpolator(X_kian)
kian_peak_finder = IonosphericPeakFinder(kian_interpolate.interpolate_data(n=n_inter))
X_kian = kian_peak_finder.get_peaks(e_prom, f_prom)

ion_interpolate = IonosphericProfileInterpolator(X_ion)
ion_peak_finder = IonosphericPeakFinder(ion_interpolate.interpolate_data(n=n_inter))
X_ion = ion_peak_finder.get_peaks(e_prom, f_prom)

geo_interpolate = IonosphericProfileInterpolator(X_geo)
geo_peak_finder = IonosphericPeakFinder(geo_interpolate.interpolate_data(n=n_inter))
X_geo = geo_peak_finder.get_peaks(e_prom, f_prom)

art_interpolate = IonosphericProfileInterpolator(X_art)
art_peak_finder = IonosphericPeakFinder(art_interpolate.interpolate_data(n=n_inter))
X_art = art_peak_finder.get_peaks(e_prom, f_prom)

ech_interpolate = IonosphericProfileInterpolator(X_ech)
ech_peak_finder = IonosphericPeakFinder(ech_interpolate.interpolate_data(n=n_inter))
X_ech = ech_peak_finder.get_peaks(e_prom, f_prom)




peak_plotter = PaperPlotter(X_eis, X_kian, X_ion, X_geo, X_art, X_ech)
# peak_plotter.plot_compare_all_error()
# peak_plotter.plot_peaks()
peak_plotter.plot_compare_all_peak_densities(True)
# peak_plotter.plot_kde_comparison()
# peak_plotter.plot_combined_peak_densities()
# peak_plotter.plot_compare_all_peak_altitudes(True)



