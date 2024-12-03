# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 15:11:24 2024

@author: Kian Sartipzadeh
"""

import os
import pickle
import pandas as pd

from artist_sorting import ArtistSorter
from artist_utils import save_data_dict, load_data_dict, merge_files, plot_data, plot_eiscat_vs_artist





if __name__ == "__main__":
    
    # --- Uncomment for merging artist files---
    # merge_files(folder_path="Artist_folder")
    # -----------------------------------------
    
    
    # --- Uncomment for processing artist file ---
    # file_path = 'Artist_folder/artist_merged.csv'
    # radar_processor = ArtistSorter(file_path)
    # radar_processor.load_data()
    
    # print("processing data")
    # radar_processor.process_data()
    # print("processing done!")
    
    # X_artist = radar_processor.processed_data
    # save_data(X_artist, file_name="artist_processed_data")
    # ------------------------------------------------------
    
    
    
    
    X_artist = load_data_dict("artist_processed_data")
    X_EISCAT = load_data_dict(file_name="X_avg_test_data")
    
    
    combined_artist_data = {}  # Dictionary to store all corresponding X_artist data

    for day in X_EISCAT:
        plot_eiscat_vs_artist(X_EISCAT[day], X_artist[day])  # Optional, if visualization is required
        combined_artist_data[day] = X_artist[day]  # Add data for the current day to the dictionary
    
    # Save the combined data to a single file
    save_data_dict(combined_artist_data, file_name="artist_test_days.pkl")

    # print("All artist data saved to combined_artist_data.pkl")