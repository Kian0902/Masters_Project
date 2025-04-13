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
from sklearn.metrics import r2_score


def load_dict(file_name):
    with open(file_name, 'rb') as f:
        dataset = pickle.load(f)
    return dataset

def save_dict(dataset: dict, file_name: str):
    with open(file_name, 'wb') as file:
        pickle.dump(dataset, file)




def get_altitude_r2_score(data_eis, data_pred):
    """
    Calculate R2 scores for each altitude point between EISCAT reference data 
    and HNN predictions.
    
    Parameters:
        eiscat_data (numpy.ndarray): Array of shape (N, M), where N is the number of altitude heights
                                     and M is the number of time points. Contains the reference electron
                                     densities from EISCAT.
        hnn_predictions (numpy.ndarray): Array of shape (N, M), where N is the number of altitude heights
                                         and M is the number of time points. Contains the predicted electron
                                         densities from the HNN model.
                                         
    Returns:
        numpy.ndarray: Array of shape (N,), where each element is the R2 score for the corresponding altitude.
    """
    # Validate input shapes
    if data_eis.shape != data_pred.shape:
        raise ValueError("EISCAT data and predictions must have the same shape.")
    
    # Number of altitudes
    num_altitudes = data_eis.shape[0]
    
    # Initialize an array to store R2 scores for each altitude
    r2_scores = np.zeros(num_altitudes)
    
    # Calculate R2 scores for each altitude
    for i in range(num_altitudes):
        # Extract data for altitude i across all time points
        true_altitude = data_eis[i, :]
        pred_altitude = data_pred[i, :]
        
        # Compute R2 score and store it
        r2_scores[i] = r2_score(true_altitude, pred_altitude)
    
    return r2_scores


def get_measurements_r2_score(data_eis, data_pred):
    """
    Calculate R2 scores for each altitude point between EISCAT reference data 
    and HNN predictions.
    
    Parameters:
        eiscat_data (numpy.ndarray): Array of shape (N, M), where N is the number of altitude heights
                                     and M is the number of time points. Contains the reference electron
                                     densities from EISCAT.
        hnn_predictions (numpy.ndarray): Array of shape (N, M), where N is the number of altitude heights
                                         and M is the number of time points. Contains the predicted electron
                                         densities from the HNN model.
                                         
    Returns:
        numpy.ndarray: Array of shape (N,), where each element is the R2 score for the corresponding altitude.
    """
    
    data_eis = data_eis.T
    data_pred = data_pred.T
    
    # Validate input shapes
    if data_eis.shape != data_pred.shape:
        raise ValueError("EISCAT data and predictions must have the same shape.")
    
    # Number of altitudes
    num_measurements = data_eis.shape[0]
    
    # Initialize an array to store R2 scores for each altitude
    r2_scores = np.zeros(num_measurements)
    
    # Calculate R2 scores for each altitude
    for i in range(num_measurements):
        # Extract data for altitude i across all time points
        true_altitude = data_eis[i, :]
        pred_altitude = data_pred[i, :]
        
        # Compute R2 score and store it
        r2_scores[i] = r2_score(true_altitude, pred_altitude)
    
    return r2_scores.T








def get_altitude_r2_score_nans(data_eis, data_pred):
    """
    Calculate R2 scores for each altitude point between EISCAT reference data 
    and HNN predictions, ignoring NaN values.
    
    Parameters:
        data_eis (numpy.ndarray): Array of shape (N, M), where N is the number of altitude heights
                                  and M is the number of time points. Contains the reference electron
                                  densities from EISCAT.
        data_pred (numpy.ndarray): Array of shape (N, M), where N is the number of altitude heights
                                   and M is the number of time points. Contains the predicted electron
                                   densities from the HNN model.
                                         
    Returns:
        numpy.ndarray: Array of shape (N,), where each element is the R2 score for the corresponding altitude.
                       If no valid data points exist for an altitude, the R2 score is set to `np.nan`.
    """
    # Validate input shapes
    if data_eis.shape != data_pred.shape:
        raise ValueError("EISCAT data and predictions must have the same shape.")
    
    # Number of altitudes
    num_altitudes = data_eis.shape[0]
    
    # Initialize an array to store R2 scores for each altitude
    r2_scores = np.full(num_altitudes, np.nan)  # Default to NaN for altitudes with insufficient data
    
    # Calculate R2 scores for each altitude
    for i in range(num_altitudes):
        # Extract data for altitude i across all time points
        true_altitude = data_eis[i, :]
        pred_altitude = data_pred[i, :]
        
        # Filter out NaN values
        valid_indices = ~np.isnan(true_altitude) & ~np.isnan(pred_altitude)
        true_valid = true_altitude[valid_indices]
        pred_valid = pred_altitude[valid_indices]
        
        # Compute R2 score only if there are sufficient valid points
        if len(true_valid) > 1:  # At least two points are needed for R2 calculation
            r2_scores[i] = r2_score(true_valid, pred_valid)
    
    return r2_scores



def get_measurements_r2_score_nans(data_eis, data_pred):
    """
    Calculate R2 scores for each altitude point between EISCAT reference data 
    and HNN predictions, ignoring NaN values.
    
    Parameters:
        data_eis (numpy.ndarray): Array of shape (N, M), where N is the number of altitude heights
                                  and M is the number of time points. Contains the reference electron
                                  densities from EISCAT.
        data_pred (numpy.ndarray): Array of shape (N, M), where N is the number of altitude heights
                                   and M is the number of time points. Contains the predicted electron
                                   densities from the HNN model.
                                         
    Returns:
        numpy.ndarray: Array of shape (N,), where each element is the R2 score for the corresponding altitude.
                       If no valid data points exist for an altitude, the R2 score is set to `np.nan`.
    """
    
    data_eis = data_eis.T
    data_pred = data_pred.T
    
    # Validate input shapes
    if data_eis.shape != data_pred.shape:
        raise ValueError("EISCAT data and predictions must have the same shape.")
    
    # Number of altitudes
    num_measurements = data_eis.shape[0]
    
    # Initialize an array to store R2 scores for each altitude
    r2_scores = np.full(num_measurements, np.nan)  # Default to NaN for altitudes with insufficient data
    
    # Calculate R2 scores for each altitude
    for i in range(num_measurements):
        # Extract data for altitude i across all time points
        true_altitude = data_eis[i, :]
        pred_altitude = data_pred[i, :]
        
        # Filter out NaN values
        valid_indices = ~np.isnan(true_altitude) & ~np.isnan(pred_altitude)
        true_valid = true_altitude[valid_indices]
        pred_valid = pred_altitude[valid_indices]
        
        # Compute R2 score only if there are sufficient valid points
        if len(true_valid) > 1:  # At least two points are needed for R2 calculation
            r2_scores[i] = r2_score(true_valid, pred_valid)
    
    return r2_scores.T











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

def add_key_from_dict_to_dict(from_X, to_X, key):
    # Reorder the keys in to_X to match the order in from_X
    for date_key in to_X.keys():
        if date_key in from_X and key in from_X[date_key]:
            # Add key from EISCAT_support
            to_X[date_key][key] = from_X[date_key][key]
            
            # Reorder keys to match the order in EISCAT_support
            ordered_keys = list(from_X[date_key].keys())  # Get the desired key order
            to_X[date_key] = OrderedDict((key, to_X[date_key][key]) for key in ordered_keys if key in to_X[date_key])
    
    return to_X


def add_key_with_matching_times(from_X, to_X, key):
    """
    Adds a key from `from_X` to `to_X` and aligns the data based on matching datetime objects in 'r_time'.

    Args:
        from_X (dict): Source dictionary.
        to_X (dict): Target dictionary to which the key is added.
        key (str): The key to be transferred and aligned.

    Returns:
        dict: Updated target dictionary with the aligned key added.
    """
    for date_key in to_X.keys():
        if date_key in from_X and key in from_X[date_key]:
            # Extract datetime arrays from both dictionaries
            from_times = from_X[date_key]['r_time']
            to_times = to_X[date_key]['r_time']

            # Find matching datetime indices
            from_indices = {tuple(time): idx for idx, time in enumerate(from_times)}
            matching_indices = [from_indices[tuple(time)] for time in to_times if tuple(time) in from_indices]

            # Filter the values of the key in from_X to match the matching indices
            if matching_indices:
                aligned_values = from_X[date_key][key][:, matching_indices]
                to_X[date_key][key] = aligned_values
            else:
                print(f"No matching times found for date {date_key}")

            # Optional: Reorder keys in to_X[date_key] to match from_X
            ordered_keys = list(from_X[date_key].keys())
            to_X[date_key] = OrderedDict((k, to_X[date_key][k]) for k in ordered_keys if k in to_X[date_key])

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




def convert_geophys_to_dict(geophys, eiscat_dict):
    # Initialize an empty dictionary for X_ion
    X_geo = {}

    # Iterate over the radar dictionary and construct the X_ion dictionary
    for date_key, radar_data in eiscat_dict.items():
        r_time = radar_data['r_time']  # This should be the array with shape (M, 6)
        num_measurements = r_time.shape[0]  # This gives us M, the number of measurements
        
        # Get the ionograms corresponding to the current date
        # Assuming `ion` is a list of ionograms measured in the same order as `r_times`
        geophys_list = []
        for idx in range(num_measurements):
            geophys_list.append(geophys.pop(0))  # Pop the next ionogram corresponding to this measurement
        
        # Add the data to X_ion
        X_geo[date_key] = {
            'r_time': r_time,
            'r_param': np.array(geophys_list).T  # r_param is an array of shape (M,) containing ionograms
        }

    return X_geo









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




def align_artist_with_eiscat(eiscat_dict, artist_dict):
    """
    Aligns the Artist 4.5 dictionary with the EISCAT UHF dictionary by adding NaN-filled columns
    for missing timestamps in 'r_param'.
    
    Parameters:
    eiscat_dict (dict): Dictionary containing EISCAT UHF data.
    artist_dict (dict): Dictionary containing Artist 4.5 data.
    
    Returns:
    dict: Updated Artist 4.5 dictionary with aligned 'r_param' shapes.
    """
    updated_artist_dict = {}
    
    for date in eiscat_dict:
        if date in artist_dict:
            eiscat_times = eiscat_dict[date]['r_time']  # Shape (M, 6)
            artist_times = artist_dict[date]['r_time']  # Shape (K, 6)
            eiscat_h = eiscat_dict[date]['r_h']         # Shape (N, 1)
            artist_h = artist_dict[date]['r_h']         # Shape (N, 1)
            
            # Ensure altitudes match
            if not np.array_equal(eiscat_h, artist_h):
                raise ValueError(f"Altitude values do not match for {date}")
            
            # Find missing timestamps
            missing_indices = []
            for i, time in enumerate(eiscat_times):
                if not any(np.array_equal(time, t) for t in artist_times):
                    missing_indices.append(i)
            
            # Create new aligned 'r_param'
            M = eiscat_times.shape[0]  # Number of timestamps in EISCAT
            K = artist_times.shape[0]  # Number of timestamps in Artist 4.5
            N = eiscat_h.shape[0]      # Number of altitude levels
            
            # Initialize a new 'r_param' with NaNs
            new_r_param = np.full((N, M), np.nan)
            
            # Insert existing Artist 4.5 data into new array at corresponding indices
            matched_indices = [np.where((eiscat_times == t).all(axis=1))[0][0] for t in artist_times]
            new_r_param[:, matched_indices] = artist_dict[date]['r_param']
            
            # Update the dictionary
            updated_artist_dict[date] = {
                'r_time': eiscat_times.copy(),  # Use EISCAT times
                'r_h': artist_h.copy(),         # Keep the same altitudes
                'r_param': new_r_param          # Updated r_param with NaNs for missing times
            }
        else:
            # If the date is missing in Artist 4.5, create a full NaN entry
            M = eiscat_dict[date]['r_time'].shape[0]
            N = eiscat_dict[date]['r_h'].shape[0]
            updated_artist_dict[date] = {
                'r_time': eiscat_dict[date]['r_time'].copy(),
                'r_h': eiscat_dict[date]['r_h'].copy(),
                'r_param': np.full((N, M), np.nan)
            }
    
    return updated_artist_dict






def merge_nested_peak_dict(nested_dict):
    all_r_time = []
    all_r_param = []
    all_r_param_peak = []
    all_r_h_peak = []
    
    sorted_keys = sorted(nested_dict.keys(), key=lambda d: datetime.strptime(d, '%Y-%m-%d'))
    
    # Loop over keys in sorted order (optional, if order matters)
    for key in sorted_keys:
        all_r_time.append(nested_dict[key]['r_time'])
        all_r_param.append(nested_dict[key]['r_param'])
        all_r_param_peak.append(nested_dict[key]['r_param_peak'])
        all_r_h_peak.append(nested_dict[key]['r_h_peak'])
        h=nested_dict[key]['r_h']
        
    # Concatenate arrays along axis=0
    merged_r_time = np.concatenate(all_r_time, axis=0)
    merged_r_param = np.concatenate(all_r_param, axis=1)
    merged_r_param_peak = np.concatenate(all_r_param_peak, axis=1)
    merged_r_h_peak = np.concatenate(all_r_h_peak, axis=1)
    
    
    # Return the merged dictionary under the key "All"
    return {"All": {"r_time": merged_r_time, "r_h": h, "r_param": merged_r_param, "r_param_peak": merged_r_param_peak, "r_h_peak": merged_r_h_peak}}



def merge_nested_peak_pred_dict(nested_dict):
    all_r_time = []
    all_r_param = []
    all_r_param_peak = []
    all_r_h_peak = []
    
    sorted_keys = sorted(nested_dict.keys(), key=lambda d: datetime.strptime(d, '%Y-%m-%d'))
    
    # Loop over keys in sorted order (optional, if order matters)
    for key in sorted_keys:
        all_r_time.append(nested_dict[key]['r_time'])
        all_r_param.append(nested_dict[key]['r_param'])
        all_r_param_peak.append(nested_dict[key]['r_param_peak'])
        all_r_h_peak.append(nested_dict[key]['r_h_peak'])
        
    # Concatenate arrays along axis=0
    merged_r_time = np.concatenate(all_r_time, axis=0)
    merged_r_param = np.concatenate(all_r_param, axis=1)
    merged_r_param_peak = np.concatenate(all_r_param_peak, axis=1)
    merged_r_h_peak = np.concatenate(all_r_h_peak, axis=1)
    
    
    # Return the merged dictionary under the key "All"
    return {"All": {"r_time": merged_r_time, "r_param": merged_r_param, "r_param_peak": merged_r_param_peak, "r_h_peak": merged_r_h_peak}}




def merge_nested_dict(nested_dict):
    all_r_time = []
    all_r_param = []
    
    sorted_keys = sorted(nested_dict.keys(), key=lambda d: datetime.strptime(d, '%Y-%m-%d'))
    
    # Loop over keys in sorted order (optional, if order matters)
    for key in sorted_keys:
        all_r_time.append(nested_dict[key]['r_time'])
        all_r_param.append(nested_dict[key]['r_param'])
        h=nested_dict[key]['r_h']
        
    # Concatenate arrays along axis=0
    merged_r_time = np.concatenate(all_r_time, axis=0)
    merged_r_param = np.concatenate(all_r_param, axis=1)
    
    # Return the merged dictionary under the key "All"
    return {"All": {"r_time": merged_r_time, "r_h": h, "r_param": merged_r_param}}


def merge_nested_pred_dict(nested_dict):
    all_r_time = []
    all_r_param = []
    
    sorted_keys = sorted(nested_dict.keys(), key=lambda d: datetime.strptime(d, '%Y-%m-%d'))
    
    # Loop over keys in sorted order (optional, if order matters)
    for key in sorted_keys:
        all_r_time.append(nested_dict[key]['r_time'])
        all_r_param.append(nested_dict[key]['r_param'])

        
    # Concatenate arrays along axis=0
    merged_r_time = np.concatenate(all_r_time, axis=0)
    merged_r_param = np.concatenate(all_r_param, axis=1)
    
    # Return the merged dictionary under the key "All"
    return {"All": {"r_time": merged_r_time, "r_param": merged_r_param}}




def merge_nested_peak_dict(nested_dict):
    all_r_time = []
    all_r_param = []
    all_r_h = []
    all_r_peak_param = []
    sorted_keys = sorted(nested_dict.keys(), key=lambda d: datetime.strptime(d, '%Y-%m-%d'))
    
    # Loop over keys in sorted order (optional, if order matters)
    for key in sorted_keys:
        all_r_time.append(nested_dict[key]['r_time'])
        all_r_param.append(nested_dict[key]['r_param'])
        all_r_h.append(nested_dict[key]['r_h_peak'])
        all_r_peak_param.append(nested_dict[key]['r_param_peak'])
        h=nested_dict[key]['r_h']
        
    # Concatenate arrays along axis=0
    merged_r_time = np.concatenate(all_r_time, axis=0)
    merged_r_param = np.concatenate(all_r_param, axis=1)
    merged_r_h = np.concatenate(all_r_h, axis=1)
    merged_r_peak_param = np.concatenate(all_r_peak_param, axis=1)
    
    # Return the merged dictionary under the key "All"
    return {"All": {"r_time": merged_r_time, "r_h": h, "r_param": merged_r_param, "r_h_peak": merged_r_h, "r_param_peak": merged_r_peak_param}}





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
            'r_param': np.log10(data['r_param']),
            'r_h_peak': data['r_h_peak'],
            'r_param_peak': np.log10(data['r_param_peak'])
        }
    return transformed_data


def revert_log10(radar_data):
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
            'r_param': 10**data['r_param']
        }
    return transformed_data


#                                       (End)
# =============================================================================







if __name__ == "__main__":
    print("...")







