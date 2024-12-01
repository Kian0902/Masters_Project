# -*- coding: utf-8 -*-
"""
Created on Sun Dec  1 15:53:17 2024

@author: Kian Sartipzadeh
"""


import os
from datetime import datetime



def filename_to_datetime(folder_name):
    
    
    # List all files in the folder
    file_names = os.listdir(folder_name)

    # Convert filenames to datetime objects
    datetimes = []
    for file_name in file_names:
        if file_name.endswith(".csv"):  # Make sure it's a CSV file
        
            # Remove the file extension (.csv)
            base_name = os.path.splitext(file_name)[0]
            
            # Convert the filename to a datetime object
            dt = datetime.strptime(base_name, "%Y%m%d_%H%M")
            datetimes.append(dt)
    
    return datetimes









