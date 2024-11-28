# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 16:54:01 2024

@author: Kian Sartipzadeh
"""

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
from datetime import datetime

def import_file(file_name):
    with open(file_name, 'rb') as f:
        dataset = pickle.load(f)
    return dataset

def save_data(dataset: dict, file_name: str):
    with open(file_name, 'wb') as file:
        pickle.dump(dataset, file)





def from_strings_to_datetime(data_strings):
    
    
    data = from_strings_to_array(data_strings)
    
    
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








def plot_compare(ne_true, ne_pred, ne_art, r_time, art_time):
    
    
    
    
    
    # Eiscat altitudes
    r_h = np.array([[91.5687711 ],[ 94.57444598],[ 97.57964223],[100.57010953],
           [103.57141624],[106.57728701],[110.08393175],[114.60422289],
           [120.1185208 ],[126.61221111],[134.1346149 ],[142.53945817],
           [152.05174717],[162.57986185],[174.09833378],[186.65837945],
           [200.15192581],[214.62769852],[230.12198695],[246.64398082],
           [264.11728204],[282.62750673],[302.15668686],[322.70723831],
           [344.19596481],[366.64409299],[390.113117]])
    
    # Artist altitudes
    art_h = np.arange(80, 485, 5)
    
    date_str = r_time[0].strftime('%Y-%m-%d')


    # Creating the plots
    fig, ax = plt.subplots(1, 3, figsize=(14, 8), sharey=True)
    fig.tight_layout()
    fig.suptitle(f'Date: {date_str}', fontsize=15)
    
    
    
    x_limits = [r_time[0], r_time[-1]]
    print(x_limits)
    
    # Plotting original data
    ne_EISCAT = ax[0].pcolormesh(r_time, r_h.flatten(), ne_true.T, shading='auto', cmap='turbo', vmin=10, vmax=12)
    ax[0].set_title('EISCAT UHF', fontsize=17)
    ax[0].set_xlabel('Time [hours]')
    ax[0].set_ylabel('Altitude [km]')
    ax[0].xaxis.set_major_formatter(DateFormatter('%H:%M'))
    
    
    # Plotting original data
    ax[1].pcolormesh(r_time, r_h.flatten(), ne_pred.T, shading='auto', cmap='turbo', vmin=10, vmax=12)
    ax[1].set_title('Ours', fontsize=17)
    ax[1].set_xlabel('Time [hours]')
    ax[1].xaxis.set_major_formatter(DateFormatter('%H:%M'))
    
    
    # Plotting original data
    ax[2].pcolormesh(art_time, art_h, ne_art.T, shading='auto', cmap='turbo', vmin=10, vmax=12)
    ax[2].set_title('Artist 4.5', fontsize=17)
    ax[2].set_xlabel('Time [hours]')
    ax[2].xaxis.set_major_formatter(DateFormatter('%H:%M'))
    ax[2].set_xlim(x_limits)
    
    
    
    # Add colorbar for the original data
    cbar = fig.colorbar(ne_EISCAT, ax=ax[2], orientation='vertical', fraction=0.03, pad=0.04, aspect=44, shrink=3)
    cbar.set_label(r'$log_{10}(n_e)$ [g/cm$^3$]', fontsize=17)
    
    fig.autofmt_xdate()
    plt.show()



# (15,) <class 'numpy.ndarray'>
# 2019-03-07 20:15:00 <class 'datetime.datetime'>
# 2019-03-07
# [datetime.datetime(2019, 3, 7, 20, 15), datetime.datetime(2019, 3, 7, 23, 59)]


# (133,) <class 'numpy.ndarray'>
# 2019-03-07 20:15:00 <class 'datetime.datetime'>
# 2019-03-07
# [datetime.datetime(2019, 3, 7, 20, 15), datetime.datetime(2021, 1, 7, 0, 0)]





def plot_results(ne_pred, ne_true):
    
    
    r_h = np.array([[91.5687711 ],[ 94.57444598],[ 97.57964223],[100.57010953],
           [103.57141624],[106.57728701],[110.08393175],[114.60422289],
           [120.1185208 ],[126.61221111],[134.1346149 ],[142.53945817],
           [152.05174717],[162.57986185],[174.09833378],[186.65837945],
           [200.15192581],[214.62769852],[230.12198695],[246.64398082],
           [264.11728204],[282.62750673],[302.15668686],[322.70723831],
           [344.19596481],[366.64409299],[390.113117  ]])
    
    
    
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.suptitle('Model Results', fontsize=15)
    
    ax.set_title('EISCAT vs HNN', fontsize=15)
    ax.plot(ne_true, r_h.flatten(), color="C0", label="EISCAT_ne")
    ax.plot(ne_pred, r_h.flatten(), color="C1", label="Pred_ne")
    ax.set_xlabel("Electron Density  log10(ne)")
    ax.set_ylabel("Altitude (km)")
    ax.set_xlim(8.9, 12.1)
    ax.grid(True)
    ax.legend()
    
    plt.show()




