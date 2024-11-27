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



if __name__ == "__main__":
    
    # merge_files(folder_path="Artist_folder")
    

    # Example usage
    file_path = 'artist_merged.csv'
    radar_processor = ArtistSorter(file_path)
    radar_processor.load_data()
    radar_processor.process_data()
    
    X_artist = radar_processor.processed_data
    X_EISCAT = import_file(file_name="X_avg_test_data")
    
    
    
    
    for day in X_EISCAT:
        plot2(X_EISCAT[day], X_artist[day])










