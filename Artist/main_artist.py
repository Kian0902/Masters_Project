# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 15:11:24 2024

@author: Kian Sartipzadeh
"""

import os
import pickle
import pandas as pd

from artist_sorting import ArtistSorter
from artist_utils import save_data, load_data, merge_files
from artist_plotting import ArtistPlotting
from artist_interpolate import filter_range, interpolate_data



def execute_merge_files(folder_path:str):
    f = folder_path or "artist_data_folder"
    merge_files(f)
    

def execute_data_sorting(file_path:str, save_to_file=True):
    
    radar_processor = ArtistSorter(file_path)
    radar_processor.load_data()
    
    radar_processor.process_data()
    
    X_artist = radar_processor.processed_data
    
    if save_to_file:
        save_data(X_artist, file_name="processed_steps/artist_merged_sorted_data")
    return X_artist


def execute_data_merging(file_path:str = None, save_to_file=True, show_plot=False):
    f = file_path or "processed_steps/artist_merged_sorted_data.pkl"
        
    X_artist = load_data(f)
    X_EISCAT = load_data(file_name="X_eiscat_test_data.pkl")
    
    combined_artist_data = {}
    for day in X_artist:
        combined_artist_data[day] = X_artist[day]  # Add data for the current day to the dictionary
        
        if show_plot:
            ArtistPlotting(X_EISCAT[day], X_artist[day]).plot_eiscat_vs_artist()
    if save_to_file:
        save_data(combined_artist_data, file_name="processed_steps/artist_sorted_combined_data")
    return combined_artist_data


def execute_data_interpolating(file_path:str = None, save_to_file=True, show_plot=False):
    f = file_path or "processed_steps/artist_sorted_combined_data.pkl"
    
    X_artist = load_data(f)
    X_EISCAT = load_data(file_name="X_eiscat_test_data.pkl")
    
    artist_processed = {}
    for day in X_EISCAT:
        X_uhf = X_EISCAT[day]
        X_art = X_artist[day]
        
        r_uhf = X_uhf['r_h']
        X_filt = filter_range(X_art, 'r_h', 90, 400)
        X_inter = interpolate_data(X_filt, r_uhf)
        
        artist_processed[day] = X_inter
        if show_plot:
            ArtistPlotting(X_uhf, X_inter).plot_eiscat_vs_artist()
    
    if save_to_file:
        save_data(artist_processed, file_name="processed_steps/artist_sorted_combined_interpolated_data")
    return artist_processed


if __name__ == "__main__":
    
    # # --- Uncomment for merging yearly artist files---
    # folder_path = "artist_data_folder"
    # execute_merge_files(folder_path)
    # # -----------------------------------------
    
    file = 'artist_merged.csv'
    execute_data_sorting(file)
    execute_data_merging()
    final_data_version = execute_data_interpolating()
    save_data(final_data_version, file_name="processed_artist_data")
    










