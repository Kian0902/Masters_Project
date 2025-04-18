# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 15:58:51 2024

@author: Kian Sartipzadeh
"""



import os
import pickle
import numpy as np
import pandas as pd


def load_data(file_name):
    with open(file_name, 'rb') as f:
        dataset = pickle.load(f)
    return dataset

def save_data(dataset: dict, file_name: str):
    with open(file_name + ".pkl", 'wb') as file:
        pickle.dump(dataset, file)
    



def merge_files(folder_path: str):
    
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
    merged_df.to_csv('artist_merged.csv', index=False)
    


def inspect_dict(d, indent=0):
    """
    Recursively print all keys in a nested dictionary with the shape of their values.
    
    :param d: The dictionary to process.
    :param indent: The current indentation level (used for nested items).
    """
    for key, value in d.items():
        # Create indentation based on the depth level
        prefix = '  ' * indent
        
        # Determine the type and shape of the value
        if isinstance(value, dict):
            value_shape = f"dict with {len(value)} keys"
        elif isinstance(value, list):
            value_shape = f"list of length {len(value)}"
        elif isinstance(value, set):
            value_shape = f"set of length {len(value)}"
        elif isinstance(value, tuple):
            value_shape = f"tuple of length {len(value)}"
        elif isinstance(value, np.ndarray):
            value_shape = f"numpy array with shape {value.shape}"
        else:
            value_shape = f"type {type(value).__name__}"
        
        # Print the key along with its value shape
        print(f"{prefix}{key}: ({value_shape})")
        
        # If the value is also a dictionary, recursively call the function
        if isinstance(value, dict):
            inspect_dict(value, indent=indent + 1)


if __name__ == "__main__":
    print("...")


