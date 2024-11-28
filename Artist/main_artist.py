# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 15:11:24 2024

@author: Kian Sartipzadeh
"""

import os
import pickle
import pandas as pd

from artist_sorting import ArtistSorter
from artist_plotting import plot1, plot2
from artist_utils import merge_files




def import_file(file_name):
    with open(file_name, 'rb') as f:
        dataset = pickle.load(f)
    return dataset

def save_data(dataset: dict, file_name: str):
    with open(file_name, 'wb') as file:
        pickle.dump(dataset, file)

if __name__ == "__main__":
    
    # merge_files(folder_path="Artist_folder")
    
    print("Start")
    
    file_path = 'Artist_folder/artist_merged.csv'
    radar_processor = ArtistSorter(file_path)
    radar_processor.load_data()
    
    print("processing data")
    radar_processor.process_data()
    print("processing done!")
    
    X_artist = radar_processor.processed_data
    X_EISCAT = import_file(file_name="X_avg_test_data")
    
    
    combined_artist_data = {}  # Dictionary to store all corresponding X_artist data

    for day in X_EISCAT:
        plot2(X_EISCAT[day], X_artist[day])  # Optional, if visualization is required
        combined_artist_data[day] = X_artist[day]  # Add data for the current day to the dictionary
    
    # Save the combined data to a single file
    save_data(combined_artist_data, file_name="artist_test_days.pkl")

    print("All artist data saved to combined_artist_data.pkl")