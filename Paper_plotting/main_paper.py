# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 16:50:34 2024

@author: Kian Sartipzadeh
"""


import numpy as np
from paper_paper_plotting import PaperPlotter
from paper_utils import load_dict, inspect_dict, merge_nested_dict, merge_nested_pred_dict


def align_X_eis(X_eis, X_ion, X_geo, X_kian):
    """
    Aligns the X_eis dictionary to match the dates and time steps of model dictionaries.
    
    Parameters:
    - X_eis (dict): Nested dict with radar measurements.
    - X_ion, X_geo, X_kian (dict): Nested dicts with model estimations.
    
    Returns:
    - dict: New X_eis with only common dates and arrays aligned to model time steps.
    """
    # Find dates common to X_eis and all models
    common_dates = set(X_eis.keys()) & set(X_ion.keys()) & set(X_geo.keys()) & set(X_kian.keys())
    
    # Initialize new dictionary
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
        
        # Build the new entry for this date
        new_X_eis[date] = {
            "r_time": ref_time.copy(),          # Use model’s time steps
            "r_h": X_eis[date]["r_h"].copy(),  # Keep original altitudes
            "r_param": new_r_param,             # Aligned parameters
            "r_error": new_r_error,             # Aligned errors
            "num_avg_samp": new_num_avg_samp   # Aligned sample counts
        }
    
    return new_X_eis





X_eis  = load_dict('X_2012_one_week')
X_ion  = load_dict('X_pred_one_week_ionocnn')
X_geo  = load_dict('X_pred_one_week_geodmlp')
X_kian = load_dict('X_pred_one_week_kiannet')

X_eis_new = align_X_eis(X_eis, X_ion, X_geo, X_kian)





plotter = PaperPlotter(X_eis_new, X_kian, X_ion, X_geo)



plotter.plot_compare_all()






