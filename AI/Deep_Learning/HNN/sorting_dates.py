# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 10:27:24 2024

@author: Kian Sartipzadeh
"""


# 20180917_1615


import os
import pickle
import numpy as np
import pandas as pd



def save(dataset, save_param, output_dir):
    # iterate over dates
    for date, data in dataset.items():
        
        print(f"{date}")
        
        r_time = data['r_time']
        r_h = data['r_h']
        
        r_rad = data[save_param]
        # r_param = data['r_param']
        # r_error = data['r_error']
        
        
        # iterate over each 15 min time
        for i in range(r_time.shape[0]):
            
            sample_r_rad = r_rad[:, i]
            
            
            sample_df = pd.DataFrame({
                save_param: sample_r_rad
                }).T
            
            time_str = f"{r_time[i, 0]:04d}{r_time[i, 1]:02d}{r_time[i, 2]:02d}_{r_time[i, 3]:02d}{r_time[i, 4]:02d}"
            csv_filename = os.path.join(output_dir, f"{time_str}.csv")
            sample_df.to_csv(csv_filename, index=False, header=False)
        






filename = "X_avg_test_data"

out_dir = "EISCAT_test_samples"

# Opening file
with open(filename, 'rb') as f:
    dataset = pickle.load(f)  # Loading dictionary



save(dataset, 'r_param', out_dir)











































































