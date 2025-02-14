# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 16:39:02 2025

@author: Kian Sartipzadeh
"""

import os
import numpy as np
from datetime import datetime
from utils import save_dict, load_dict


def convert_xarray(data, save_path=None):

    # Initialize variables
    r_time = []
    r_h = None
    r_param = []

    # Process each xarray in the dataset
    for xarr in data:
        # Extract time and convert to the desired format
        time = np.datetime_as_string(xarr.time.values[0], unit="s")  # Ensure seconds precision
        dt = datetime.strptime(time, "%Y-%m-%dT%H:%M:%S")  # Parse datetime
        r_time.append([dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second])

        # Extract altitude and electron density
        if r_h is None:  # Set r_h once (assume it doesn't change across time steps)
            r_h = xarr.alt_km.values[:, np.newaxis]

        # Extract electron density ("ne") and append to r_param
        r_param.append(xarr.ne.values)

    # Convert r_time to numpy array with dtype np.int64
    r_time = np.array(r_time, dtype=np.int64)  # (M, 6)

    # Convert r_param to numpy array
    r_param = np.stack(r_param, axis=-1)  # (N, M)

    # Create the final dictionary
    custom_dict = {
        "r_time": r_time,
        "r_h": r_h,
        "r_param": r_param
    }

    # Save the dictionary (optional)
    if save_path:
        save_dict(custom_dict, save_path)

    return custom_dict





def sort_days(folder_path, save_path=None):

    nested_dict = {}
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".pickle") and file_name.startswith("iri_"):
            
            date_part = file_name.split("_")[1].replace(".pickle", "")
            
            # Convert the date to the desired format: "yyyy-m-d"
            formatted_date = f"{int(date_part[:4])}-{int(date_part[4:6])}-{int(date_part[6:])}"
            file_path = os.path.join(folder_path, file_name)
            data = load_dict(file_path)
            
            # Assign to the nested dictionary
            nested_dict[formatted_date] = data

    # Save the nested dictionary to a pickle file
    if save_path:
        save_dict(nested_dict, save_path)
        
    return nested_dict



if __name__=="__main__":
    
    # file_path = "IRI_unformatted/iri_data_20190105.pickle"
    
    # x = load_dict(file_path)
    
    # convert_xarray(x, "IRI_formatted/iri_20190105.pickle")
    folder_path = "IRI_formatted"
    
    sort_days(folder_path, save_path="X_IRI")
    
    nested_dict = load_dict("X_IRI")
    
    
    
    # print(nested_dict.keys())
    # print(nested_dict["2019-1-5"].keys())
