# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 15:58:51 2024

@author: Kian Sartipzadeh
"""



import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from datetime import datetime
from matplotlib.dates import DateFormatter



def load_data_dict(file_name):
    with open(file_name, 'rb') as f:
        dataset = pickle.load(f)
    return dataset

def save_data_dict(dataset: dict, file_name: str):
    with open(file_name, 'wb') as file:
        pickle.dump(dataset, file)
    
    print(f"Dict save as {file_name}")



def merge_files(folder_path):
    
    # Get all CSV files in the folder
    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    
    # List to hold DataFrames
    dataframes = []
    
    # Read each CSV file into a DataFrame and append to the list
    for file in csv_files:
        file_path = os.path.join(folder_path, file)
        df = pd.read_csv(file_path, keep_default_na=True)
        dataframes.append(df)
    
    # Concatenate all DataFrames
    merged_df = pd.concat(dataframes, ignore_index=True)
    
    # Save the merged DataFrame to a new CSV file
    merged_df.to_csv(folder_path + '/echaim_merged.csv', index=False)
    
    print("Files merged successfully!")






if __name__ == "__main__":
    print("...")


