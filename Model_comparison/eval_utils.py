# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 16:54:01 2024

@author: Kian Sartipzadeh
"""

import os
import pickle
import numpy as np
from datetime import datetime
from collections import defaultdict



def load_dict(file_name):
    with open(file_name, 'rb') as f:
        dataset = pickle.load(f)
    return dataset

def save_dict(dataset: dict, file_name: str):
    with open(file_name, 'wb') as file:
        pickle.dump(dataset, file)




def from_strings_to_datetime(data_strings):
    data = from_strings_to_array(data_strings)
    
    # Convert time arrays to datetime objects
    r_time = np.array([datetime(year, month, day, hour, minute, second) 
                            for year, month, day, hour, minute, second in data])
    
    return r_time


def from_array_to_datetime(data):
    # Convert time arrays to datetime objects
    r_time = np.array([datetime(year, month, day, hour, minute, second) 
                            for year, month, day, hour, minute, second in data])
    
    return r_time



# Function to preprocess the input strings
def from_strings_to_array(date_strings):
    result = []
    for date_string in date_strings:
        date_part, time_part = date_string.split("_")
        year = int(date_part[:4])
        month = int(date_part[4:6])
        day = int(date_part[6:8])
        hour = int(time_part[:2])
        minute = int(time_part[2:4])
        second = 0  # Assuming seconds are always zero
        result.append(np.array([year, month, day, hour, minute, second]))
        
    return result





def from_csv_to_numpy(folder):
    list_files = [f for f in os.listdir(folder) if f.endswith('.csv')]
    
    data = []
    file_names = []
    for file in list_files:
        name = os.path.splitext(file)[0]
        file_names.append(name)
        
        file_path = os.path.join(folder, file)
        x = np.genfromtxt(file_path, dtype=np.float64, delimiter=",")
        data.append(x)
        
    return np.array(data), file_names



def from_csv_to_filename(folder):
    list_files = [os.path.splitext(f)[0] for f in os.listdir(folder) if f.endswith('.csv')]
    return list_files




def convert_pred_to_dict(r_t, r_times, ne_pred):
    # Initialize the nested dictionary
    nested_dict = defaultdict(lambda: {'r_time': [], 'r_param': []})
    
    
    for i in range(len(r_t)):
        
        # Extract the date in 'yyyy-m-dd' format
        # date_str = r_t[i].strftime('%Y-%m-%d')
        date_str = f"{r_t[i].year}-{r_t[i].month}-{r_t[i].day}"
    
        # Append the time and corresponding radar measurement to the nested dictionary
        nested_dict[date_str]['r_time'].append(r_times[i])
        nested_dict[date_str]['r_param'].append(ne_pred[i, :])
    
    for date in nested_dict:
        nested_dict[date]['r_time'] = np.array(nested_dict[date]['r_time'])
        nested_dict[date]['r_param'] = np.array(nested_dict[date]['r_param']).T
    
    
    # Convert defaultdict back to a regular dict if necessary
    nested_dict = dict(nested_dict)
    return nested_dict






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




























