# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 15:11:24 2024

@author: Kian Sartipzadeh
"""

from echaim_sorting import EChaimSorter
from echaim_utils import save_data, load_data
from echaim_plotting import EChaimPlotting
from echaim_interpolate import filter_range, interpolate_data


def execute_data_sorting(file_path:str, save_to_file=True):
    
    radar_processor = EChaimSorter(file_path)
    radar_processor.load_data()
    
    radar_processor.process_data()
    
    X_echaim = radar_processor.processed_data
    
    if save_to_file:
        save_data(X_echaim, file_name="processed_steps/echaim_sorted_data")
    return X_echaim



def execute_data_merging(file_path:str = None, save_to_file=True, show_plot=False):
    f = file_path or "processed_steps/echaim_sorted_data.pkl"
        
    
    X_echaim = load_data(f)
    X_EISCAT = load_data(file_name="X_eiscat_control.pkl")
    
    combined_echaim_data = {}  # Dictionary to store all corresponding X_echaim data
    
    for day in X_echaim:
        combined_echaim_data[day] = X_echaim[day]  # Add data for the current day to the dictionary
        
        if show_plot:
            EChaimPlotting(X_EISCAT[day], X_echaim[day]).plot_eiscat_vs_echaim()
    if save_to_file:
        save_data(combined_echaim_data, file_name="processed_steps/echaim_sorted_combined_data")
    return combined_echaim_data
    
    
def execute_data_interpolating(file_path:str = None, save_to_file=True, show_plot=False):
    f = file_path or "processed_steps/echaim_sorted_combined_data.pkl"
    
    X_echaim = load_data(f)
    X_EISCAT = load_data(file_name="X_kian")
    
    echaim_processed = {}
    for day in X_echaim:
        X_uhf = X_EISCAT[day]
        X_art = X_echaim[day]
        
        r_uhf = X_uhf['r_h']
        X_filt = filter_range(X_art, 'r_h', 90, 400)
        X_inter = interpolate_data(X_filt, r_uhf)
        
        echaim_processed[day] = X_inter
        if show_plot:
            EChaimPlotting(X_uhf, X_inter).plot_eiscat_vs_echaim()
    
    if save_to_file:
        save_data(echaim_processed, file_name="processed_steps/echaim_sorted_combined_interpolated_data")
    return echaim_processed





if __name__ == "__main__":
    file = 'echaim_data.csv'
    execute_data_sorting(file)
    execute_data_merging()
    final_data_version = execute_data_interpolating()
    save_data(final_data_version, file_name="processed_echaim_data")








