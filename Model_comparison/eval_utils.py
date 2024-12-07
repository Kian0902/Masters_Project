# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 16:54:01 2024

@author: Kian Sartipzadeh
"""

import os
import pickle
import numpy as np
from datetime import datetime
from collections import defaultdict, OrderedDict



def load_dict(file_name):
    with open(file_name, 'rb') as f:
        dataset = pickle.load(f)
    return dataset

def save_dict(dataset: dict, file_name: str):
    with open(file_name, 'wb') as file:
        pickle.dump(dataset, file)


# =============================================================================
#           Functions for handeling filenames and datetimes
#                               (Start)

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
#                                       (End)
# =============================================================================





# =============================================================================
#                    Functions for handeling dictionaries
#                                 (Start)

def add_key_from_dict_to_dict(from_X, to_X):
    # Reorder the keys in to_X to match the order in from_X
    for date_key in to_X.keys():
        if date_key in from_X and "r_h" in from_X[date_key]:
            # Add r_h from EISCAT_support
            to_X[date_key]["r_h"] = from_X[date_key]["r_h"]
            
            # Reorder keys to match the order in EISCAT_support
            ordered_keys = list(from_X[date_key].keys())  # Get the desired key order
            to_X[date_key] = OrderedDict((key, to_X[date_key][key]) for key in ordered_keys if key in to_X[date_key])
    
    return to_X






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




def convert_ionograms_to_dict(ionograms, eiscat_dict):
    # Initialize an empty dictionary for X_ion
    X_ion = {}

    # Iterate over the radar dictionary and construct the X_ion dictionary
    for date_key, radar_data in eiscat_dict.items():
        r_time = radar_data['r_time']  # This should be the array with shape (M, 6)
        num_measurements = r_time.shape[0]  # This gives us M, the number of measurements
        
        # Get the ionograms corresponding to the current date
        # Assuming `ion` is a list of ionograms measured in the same order as `r_times`
        ionogram_list = []
        for idx in range(num_measurements):
            ionogram_list.append(ionograms.pop(0))  # Pop the next ionogram corresponding to this measurement
        
        # Add the data to X_ion
        X_ion[date_key] = {
            'r_time': r_time,
            'r_param': np.array(ionogram_list, dtype=object)  # r_param is an array of shape (M,) containing ionograms
        }

    return X_ion




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





def filter_artist_times(dict_eis: dict, dict_hnn: dict, dict_art: dict):
    filtered_art = {}

    for date in dict_art:
        # Convert 'r_time' arrays to datetime objects
        r_time_eis = from_array_to_datetime(dict_eis[date]['r_time'])
        r_time_hnn = from_array_to_datetime(dict_hnn[date]['r_time'])
        r_time_art = from_array_to_datetime(dict_art[date]['r_time'])
        
        # Find common timestamps
        common_times = np.intersect1d(np.intersect1d(r_time_eis, r_time_hnn), r_time_art)
        
        # Get indices of common timestamps in radar3's 'r_time'
        common_indices = np.isin(r_time_art, common_times)

        # Filter 'r_time' and 'r_param' arrays in radar3
        filtered_art[date] = {
            'r_time': dict_art[date]['r_time'][common_indices],
            'r_h': dict_art[date]['r_h'],
            'r_param': dict_art[date]['r_param'][:, common_indices]
        }
    
    return filtered_art



def apply_log10(radar_data):
    """
    Apply np.log10 to the 'r_param' key values in the radar data dictionary.
    :param radar_data: Dictionary containing radar data with dates as keys and sub-dictionaries
                       with keys 'r_time', 'r_h', 'r_param'.
    :return: A new dictionary with the same structure but with 'r_param' transformed by np.log10.
    """
    transformed_data = {}
    for day, data in radar_data.items():
        transformed_data[day] = {
            'r_time': data['r_time'],
            'r_h': data['r_h'],
            'r_param': np.log10(data['r_param'])
        }
    return transformed_data



#                                       (End)
# =============================================================================



















