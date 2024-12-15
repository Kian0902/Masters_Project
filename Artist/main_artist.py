# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 15:11:24 2024

@author: Kian Sartipzadeh
"""

import os
import pickle
import pandas as pd

from artist_sorting import ArtistSorter
from artist_utils import save_data_dict, load_data_dict, merge_files
from artist_plotting import ArtistPlotting
from artist_interpolate import filter_range, interpolate_data



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
    

    
    # --- Uncomment for processing all artist file ---
    # X_artist = load_data_dict("artist_processed_data")
    # X_EISCAT = load_data_dict(file_name="X_avg_test_data")
    
    
    # combined_artist_data = {}  # Dictionary to store all corresponding X_artist data

    # for day in X_EISCAT:
    #     plot_eiscat_vs_artist(X_EISCAT[day], X_artist[day])  # Optional, if visualization is required
    #     combined_artist_data[day] = X_artist[day]  # Add data for the current day to the dictionary
    
    # # Save the combined data to a single file
    # save_data_dict(combined_artist_data, file_name="artist_test_days.pkl")

    # print("All artist data saved to combined_artist_data.pkl")
    # ------------------------------------------------------
    
    
    
    
    # --- Uncomment for interpolating and saving the data ---
    # X_artist = load_data_dict("artist_test_days.pkl")
    X_EISCAT = load_data_dict(file_name="X_avg_test_data")


    # artist_processed = {}

    # for day in X_artist:
    #     X_uhf = X_EISCAT[day]
    #     X_art = X_artist[day]
    #     # plot_data(X)
    #     r_uhf = X_uhf['r_h']
    #     # plot_eiscat_vs_artist(X_uhf, X_art)
        
    #     X_filt = filter_range(X_art, 'r_h', 90, 400)
    #     # plot_eiscat_vs_artist(X_uhf, X_filt)
        
        
    #     X_inter = interpolate_data(X_filt, r_uhf)
    #     plot_eiscat_vs_artist(X_uhf, X_inter)
    #     artist_processed[day] = X_inter
    
    # save_data_dict(artist_processed, file_name="processed_artist_test_days.pkl")
    # ------------------------------------------------------
    
    
    
    X_Artist = load_data_dict("processed_artist_test_days.pkl")
    
    day = "2019-1-5"
    X_art = X_Artist[day]
    X_eis = X_EISCAT[day]
    
    
    plotting = ArtistPlotting(X_eis, X_art)
    
    plotting.plot_profiles()
    plotting.plot_eiscat_vs_artist()
    
    














