# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 13:44:20 2024

@author: Kian Sartipzadeh
"""


import numpy as np
import matplotlib.pyplot as plt

from datetime import datetime
from matplotlib.dates import DateFormatter





def plot(data):
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
    pcm_ne=ax[0].pcolormesh(r_time, r_h.flatten(), np.log10(r_param.T), shading='auto', cmap='turbo', vmin=9, vmax=12)
    ax[0].set_title('EISCAT UHF', fontsize=17)
    ax[0].set_xlabel('Time [hours]')
    ax[0].set_ylabel('Altitude [km]')
    ax[0].xaxis.set_major_formatter(DateFormatter('%H:%M'))
    fig.autofmt_xdate()
    
    # Add colorbar for the original data
    cbar = fig.colorbar(pcm_ne, ax=ax[0], orientation='vertical')
    cbar.set_label(r'$log_{10}(n_e)$ [g/cm$^3$]', fontsize=17)
    
    for i in range(0, r_param.shape[0]):
        
        ax[1].plot(np.log10(r_param[i]), r_h.flatten())

    
    
    # Display the plots
    plt.show()





