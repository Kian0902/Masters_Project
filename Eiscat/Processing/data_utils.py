# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 17:24:01 2024

@author: Kian Sartipzadeh
"""

import os
import pickle
import numpy as np
from datetime import datetime



def load_dict(file_name):
    with open(file_name, 'rb') as f:
        dataset = pickle.load(f)
    return dataset

def save_dict(dataset: dict, file_name: str):
    with open(file_name, 'wb') as file:
        pickle.dump(dataset, file)





def get_day_data(dataset, day_idx):
    list_days = list(dataset.keys())
    day = list_days[day_idx]
    return dataset[day]


def get_day(dataset, day_idx):
    list_days = list(dataset.keys())
    day = list_days[day_idx]
    return {day: dataset[day]}



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








class MatchingFiles:
    """
    Class for handeling matching files between two folders.
    
    This becomes useful when faced with two data sources with matching
    filenames. Here the user has the option to delete the matching files
    from one or the other folders.
    
    Example: We have two folders containing radar data VHF and UHF but want
    to prioritize keeping the UHF data. Here, the sure has the option to delete
    the matching files in VHF folder.
    """
    def __init__(self, folder_1, folder_2):
        self.folder_1 = folder_1
        self.folder_2 = folder_2
    
    
    def list_mat_files(self, folder):
        return [f for f in os.listdir(folder) if f.endswith('.mat')]

    
    def get_matching_filenames(self):
        filenames_1 = self.list_mat_files(self.folder_1)
        filenames_2 = self.list_mat_files(self.folder_2)
        
        folder_1_filenames = set(os.path.splitext(f)[0] for f in filenames_1)
        folder_2_filenames = set(os.path.splitext(f)[0] for f in filenames_2)
        
        return sorted(list(folder_1_filenames.intersection(folder_2_filenames)))
        
    def remove_matching_vhf_files(self):
        matching_filenames = self.get_matching_filenames()
        for filename in matching_filenames:
            file_path = os.path.join(self.folder_1, f"{filename}.mat")
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"Removed: {file_path}")
            else:
                print(f"File not found: {file_path}")








