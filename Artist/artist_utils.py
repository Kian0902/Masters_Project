# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 15:58:51 2024

@author: Kian Sartipzadeh
"""



import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from datetime import datetime
from matplotlib.dates import DateFormatter




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
    merged_df.to_csv('artist_merged.csv', index=False)
    
    print("Files merged successfully!")







def plot1(data):
    """
    Plot a comparison of original and averaged data using pcolormesh.

    Input (type)                 | DESCRIPTION
    ------------------------------------------------
    original_data (dict)         | Dictionary containing the original data.
    """
    
    # Convert time arrays to datetime objects
    r_time = np.array([datetime(year, month, day, hour, minute, second) 
                            for year, month, day, hour, minute, second in data['r_time']])
    r_h = data['r_h']
    r_param = data['r_param']
    # r_error = data['r_error']
    
    # Date
    date_str = r_time[0].strftime('%Y-%m-%d')
    
    # Creating the plots
    fig, ax = plt.subplots(1, 2, figsize=(10, 8))
    fig.suptitle(f'Date: {date_str}', fontsize=15)
    fig.tight_layout()
    
    
    # Plotting original data
    pcm_ne=ax[0].pcolormesh(r_time, r_h.flatten(), np.log10(r_param), shading='auto', cmap='turbo', vmin=9, vmax=12)
    ax[0].set_title('EISCAT UHF', fontsize=17)
    ax[0].set_xlabel('Time [hours]')
    ax[0].set_ylabel('Altitude [km]')
    ax[0].xaxis.set_major_formatter(DateFormatter('%H:%M'))
    fig.autofmt_xdate()
    
    # Add colorbar for the original data
    cbar = fig.colorbar(pcm_ne, ax=ax[0], orientation='vertical')
    cbar.set_label(r'$log_{10}(n_e)$ [g/cm$^3$]', fontsize=17)
    
    for i in range(0, r_param.shape[0]):
        
        ax[1].plot(np.log10(r_param[i].T), r_h.flatten())

    
    
    # Display the plots
    plt.show()






def plot2(data1, data2):
    """
    Plot a comparison of original and averaged data using pcolormesh.

    Input (type)                 | DESCRIPTION
    ------------------------------------------------
    original_data (dict)         | Dictionary containing the original data.
    """
    
    # Convert time arrays to datetime objects
    r_time1 = np.array([datetime(year, month, day, hour, minute, second) 
                            for year, month, day, hour, minute, second in data1['r_time']])
    # Convert time arrays to datetime objects
    r_time2 = np.array([datetime(year, month, day, hour, minute, second) 
                            for year, month, day, hour, minute, second in data2['r_time']])
    r_h1 = data1['r_h']
    r_h2 = data2['r_h']
    r_param1 = data1['r_param']
    r_param2 = data2['r_param']
    
    
    # Date
    date_str = r_time1[0].strftime('%Y-%m-%d')
    
    
    # Creating the plots
    fig, ax = plt.subplots(1, 2, figsize=(10, 8), sharey=True)
    fig.suptitle(f'Date: {date_str}', fontsize=15)
    fig.tight_layout()
    fig.autofmt_xdate()
    x_limits = [r_time1[0], r_time1[-1]]
    

    # Plotting original data
    ne_EISCAT = ax[0].pcolormesh(r_time1, r_h1.flatten(), np.log10(r_param1), shading='auto', cmap='turbo', vmin=10, vmax=12)
    ax[0].set_title('EISCAT UHF', fontsize=17)
    ax[0].set_xlabel('Time [hours]')
    ax[0].set_ylabel('Altitude [km]')
    ax[0].xaxis.set_major_formatter(DateFormatter('%H:%M'))
    
    
    # Plotting original data
    ne_Artist = ax[1].pcolormesh(r_time2, r_h2.flatten(), np.log10(r_param2), shading='auto', cmap='turbo', vmin=10, vmax=12)
    ax[1].set_title('Artist', fontsize=17)
    ax[1].set_xlabel('Time [hours]')
    ax[1].xaxis.set_major_formatter(DateFormatter('%H:%M'))
    ax[1].set_xlim(x_limits)
    
    # Add colorbar for the original data
    cbar = fig.colorbar(ne_EISCAT, ax=ax[1], orientation='vertical', fraction=0.03, pad=0.04, aspect=44, shrink=3)
    cbar.set_label(r'$log_{10}(n_e)$ [g/cm$^3$]', fontsize=17)
    
    
    # Display the plots
    plt.show()


if __name__ == "__main__":
    print("...")


