# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 10:27:24 2024

@author: Kian Sartipzadeh
"""



import os
import pickle
import numpy as np
import pandas as pd


def import_file(file_name):
    with open(file_name, 'rb') as f:
        dataset = pickle.load(f)
    return dataset

def save_data(dataset: dict, file_name: str):
    with open(file_name, 'wb') as file:
        pickle.dump(dataset, file)



def make_samples(dataset: dict, output_dir="outputs"):
    # iterate over dates
    for date, data in dataset.items():
        
        print(f"{date}")
        
        r_time = data['r_time']
        r_h = data['r_h']
        r_param = data['r_param']
        
        
        
        
        # iterate over each 15 min time
        for i in range(r_time.shape[0]):
            
            sample_r_param = r_param[:, i]
            
            
            sample_df = pd.DataFrame({
                'r_param': sample_r_param
                }).T
            
            time_str = f"{r_time[i, 0]:04d}{r_time[i, 1]:02d}{r_time[i, 2]:02d}_{r_time[i, 3]:02d}{r_time[i, 4]:02d}"
            csv_filename = os.path.join(output_dir, f"{time_str}.csv")
            sample_df.to_csv(csv_filename, index=False, header=False)





if __name__ == "__main__":
    
    
    filename = "X_avg_test_data"
    output_dir = "testing_data/test_eiscat_shutup_rusland"

    X = import_file(filename)
    make_samples(X, output_dir)

    
















































































