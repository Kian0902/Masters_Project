# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 15:11:24 2024

@author: Kian Sartipzadeh
"""

import os
import pickle
# import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt

# from datetime import datetime
# from matplotlib.dates import DateFormatter

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
        plot1(X_artist[day])
        # plot2(X_EISCAT[day], X_artist[day])
        
        
    #     # for i in range(0, X_artist[day]['r_param'].shape[0]):
    #     #     plt.plot(np.log10(X_artist[day]['r_param'][i]), X_artist[day]['r_h'].flatten())
    #     # plt.show()
    











