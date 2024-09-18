# -*- coding: utf-8 -*-
"""
Created on Tue Sep 18 14:42:19 2024

@author: Kian Sartipzadeh
"""



import os
import pandas as pd




def list_csv_files(folder_path):
    return [f for f in os.listdir(folder_path) if f.endswith('.csv')]


def match_files(A_files, B_files):
    
    
    radar_dates = {f.split('.')[0]: f for f in A_files}
    satellite_dates = {f.split('.')[0]: f for f in B_files}
    
    
    matched_pairs = []
    for date in satellite_dates.keys():
        if date in radar_dates:
            matched_pairs.append((satellite_dates[date], radar_dates[date]))
    
    return matched_pairs



def load_data_pairs(radar_folder, satellite_folder, matched_pairs):
    paired_data = []
    
    i=1
    for satellite_file, radar_file in matched_pairs:
        satellite_data = pd.read_csv(os.path.join(satellite_folder, satellite_file))
        radar_data = pd.read_csv(os.path.join(radar_folder, radar_file))
        paired_data.append((satellite_data, radar_data))
        
        if i % 100 ==0:
            print(i)
        
        i+=1
        
    return paired_data

eiscat_files = list_csv_files("EISCAT_samples")
sp19_files = list_csv_files("SP19_samples")

pairs = match_files(eiscat_files, sp19_files)


dataset = load_data_pairs("EISCAT_samples", "SP19_samples", pairs)




print(len(pairs))
print(len(dataset))












