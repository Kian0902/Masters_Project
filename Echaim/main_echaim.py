# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 15:11:24 2024

@author: Kian Sartipzadeh
"""

import os
import pickle
import pandas as pd

from echaim_sorting import EChaimSorter
from echaim_utils import save_data_dict, load_data_dict, merge_files
from echaim_plotting import EChaimPlotting
from echaim_interpolate import filter_range, interpolate_data


def execute_data_sorting(file_path:str):
    
    radar_processor = EChaimSorter(file_path)
    radar_processor.load_data()
    
    radar_processor.process_data()
    
    X_echaim = radar_processor.processed_data
    save_data_dict(X_echaim, file_name="processed_steps/echaim_sorted_data")
    return X_echaim



def execute_data_merging(file_path:str = None, show_plot=False):
    f = file_path or "processed_steps/echaim_sorted_data"
        
    
    X_echaim = load_data_dict(f)
    X_EISCAT = load_data_dict(file_name="X_kian")
    
    combined_echaim_data = {}  # Dictionary to store all corresponding X_echaim data
    
    for day in X_EISCAT:
        print(day)
        combined_echaim_data[day] = X_echaim[day]  # Add data for the current day to the dictionary
        
        if show_plot:
            EChaimPlotting(X_EISCAT[day], X_echaim[day]).plot_eiscat_vs_echaim()
        
    save_data_dict(combined_echaim_data, file_name="echaim_sorted_combined_data.pkl")
    return combined_echaim_data
    
    
    
    
    # # --- Uncomment for interpolating and saving the data ---
    # X_echaim = load_data_dict("echaim_combined_days.pkl")
    # X_EISCAT = load_data_dict(file_name="X_kian")
    
    
    # echaim_processed = {}
    
    # for day in X_echaim:
    #     X_uhf = X_EISCAT[day]
    #     X_art = X_echaim[day]
    #     # plot_data(X)
    #     r_uhf = X_uhf['r_h']
    #     # plot_eiscat_vs_echaim(X_uhf, X_art)
        
    #     X_filt = filter_range(X_art, 'r_h', 90, 400)
    #     # plot_eiscat_vs_echaim(X_uhf, X_filt)
        
        
    #     X_inter = interpolate_data(X_filt, r_uhf)
    #     EChaimPlotting(X_uhf, X_inter).plot_eiscat_vs_echaim()
    #     echaim_processed[day] = X_inter
    
    # save_data_dict(echaim_processed, file_name="echaim_combined_interpolated_days.pkl")
    # # ------------------------------------------------------



if __name__ == "__main__":
    file = 'echaim_data.csv'
    execute_data_sorting(file)
    execute_data_merging()
    
