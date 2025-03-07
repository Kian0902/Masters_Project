# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 10:59:31 2025

@author: kian0
"""


import os
import pickle 
import numpy as np
from tqdm import tqdm

def load_dict(file_name):
    with open(file_name, 'rb') as f:
        dataset = pickle.load(f)
    return dataset

def save_dict(dataset: dict, file_name: str):
    with open(file_name, 'wb') as file:
        pickle.dump(dataset, file)

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





def merge_days(folder_path, save=False, save_filename="X_merged_daily_sorted_ionosonde_data"):
    nested_dict = {}
    
    for filename in tqdm(os.listdir(folder_path)):
        file_path = os.path.join(folder_path, filename)
        
        # Check if it's a file 
        if os.path.isfile(file_path):
            date_key, _ = os.path.splitext(filename)  # Remove file extension
            
            with open(file_path, 'rb') as file:
                try:
                    data = pickle.load(file)
                    
                    # Ensure the expected keys exist in the data
                    if 'r_time' in data and 'r_param' in data:
                        nested_dict[date_key] = data
                    else:
                        print(f"Warning: File '{filename}' is missing expected keys.")
                except Exception as e:
                    print(f"Error loading '{filename}': {e}")
    
    
    if save:
        save_dict(nested_dict, save_filename)
    
    return nested_dict








