# -*- coding: utf-8 -*-
"""
Created on Tue Sep 18 14:42:19 2024

@author: Kian Sartipzadeh
"""


import os
import numpy as np
import pandas as pd



def list_csv_files(folder_path):
    return [f for f in os.listdir(folder_path) if f.endswith('.csv')]

def get_filename_without_extension(filename):
    return os.path.splitext(filename)[0]


radar_names = list_csv_files("EISCAT_samples")
satellite_names = list_csv_files("SP19_samples")


# Extract base filenames without ".csv"
radar_filenames = set(get_filename_without_extension(f) for f in radar_names)
satellite_filenames = set(get_filename_without_extension(f) for f in satellite_names)

# Find the matching filenames
matching_filenames = sorted(list(satellite_filenames.intersection(radar_filenames)))





radar_data_list = []
satellite_data_list = []

for filename in matching_filenames:
    radar_file_path = os.path.join("EISCAT_samples", filename + ".csv")
    satellite_file_path = os.path.join("SP19_samples", filename + ".csv")
    
    
    radar_data = np.loadtxt(radar_file_path, delimiter=',')
    satellite_data = np.loadtxt(satellite_file_path, delimiter=',')
    
    
    radar_data_list.append(radar_data)
    satellite_data_list.append(satellite_data)


radar_data = np.array(radar_data_list)
satellite_data= np.array(satellite_data_list)


print(radar_data[0])
print(satellite_data[0])


np.save('eiscat_data.npy', radar_data)
np.save('sp19_data.npy', satellite_data)





































