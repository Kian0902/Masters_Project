# -*- coding: utf-8 -*-
"""
Created on Sun Dec  1 15:53:17 2024

@author: Kian Sartipzadeh
"""


import os
import pandas as pd
from datetime import datetime

from tqdm import tqdm


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





def create_samples_csv(dataset, output_dir="geophys_samples"):
    
    
    os.makedirs(output_dir, exist_ok=True)
    
    for i, row in tqdm(dataset.iterrows()):
        filename = row['DateTime'].strftime('%Y%m%d_%H%M') + '.csv'
        file_path = os.path.join(output_dir, filename)
        
        sample = row.drop(labels=['DateTime']).to_frame().transpose()
        sample.to_csv(file_path, index=False, header=False)
        # break
        
    print(f"CSV files saved in '{output_dir}'")






























