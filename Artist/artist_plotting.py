# -*- coding: utf-8 -*-
"""
Created on Sun Dec 15 12:26:32 2024

@author: Kian Sartipzadeh
"""

import numpy as np
import matplotlib.pyplot as plt

from datetime import datetime
from matplotlib.dates import DateFormatter




class ArtistPlotting:
    def __init__(self, X_EISCAT, X_Artist):
        X_EISCAT = self.X_EISCAT
        X_Artist = self.X_Artist

    

    def plot_data(self):
        """
        Plot a comparison of original and averaged data using pcolormesh.
    
        Input (type)                 | DESCRIPTION
        ------------------------------------------------
        original_data (dict)         | Dictionary containing the original data.
        """
        
        
        data = self.X_Artist
        
        # Convert time arrays to datetime objects
        r_time = np.array([datetime(year, month, day, hour, minute, second) 
                                for year, month, day, hour, minute, second in data['r_time']])
        r_h = data['r_h']
        r_param = data['r_param']
        
        
        
        # Date
        date_str = r_time[0].strftime('%Y-%m-%d')
        
        # Creating the plots
        fig, ax = plt.subplots(1, 2, figsize=(12, 7), gridspec_kw={'width_ratios': [0.6, 1]}, sharey=True)
        fig.suptitle(f'Date: {date_str}', fontsize=20, y=1.02)
        fig.tight_layout()
        # fig.autofmt_xdate()
        
        ax[0].set_title('All measurements', fontsize=17)
        ax[0].set_xlabel(r'$log_{10}(n_e)$ [n/cm$^3$]', fontsize=13)
        ax[0].set_ylabel('Altitude [km]', fontsize=15)
        ax[0].grid(True)
        ax[0].tick_params(axis='x', rotation=0)
        for i in range(0, r_param.shape[1]):
            
            ax[0].plot(np.log10(r_param[:, i].T), r_h.flatten())
            
        
        # Plotting original data
        pcm_ne=ax[1].pcolormesh(r_time, r_h.flatten(), np.log10(r_param), shading='auto', cmap='turbo', vmin=10, vmax=12)
        ax[1].set_title('EISCAT UHF', fontsize=17)
        ax[1].set_xlabel('Time [hours]', fontsize=13)
        ax[1].xaxis.set_major_formatter(DateFormatter('%H:%M'))
        ax[1].tick_params(axis='x', rotation=45)
        
        # Add colorbar for the original data
        cbar = fig.colorbar(pcm_ne, ax=ax[1], orientation='vertical')
        cbar.set_label(r'$log_{10}(n_e)$ [n/cm$^3$]', fontsize=17)
        
        
        # Display the plots
        plt.show()
        
    
    
    
    
    
    def plot_eiscat_vs_artist(self):
        """
        Plot a comparison of original and averaged data using pcolormesh.
    
        Input (type)                 | DESCRIPTION
        ------------------------------------------------
        original_data (dict)         | Dictionary containing the original data.
        """
        
        X_eiscat = self.X_EISCAT
        X_artist = self.X_Artist
        
        
        # Convert time arrays to datetime objects
        r_time1 = np.array([datetime(year, month, day, hour, minute, second) 
                                for year, month, day, hour, minute, second in X_eiscat['r_time']])
        # Convert time arrays to datetime objects
        r_time2 = np.array([datetime(year, month, day, hour, minute, second) 
                                for year, month, day, hour, minute, second in X_artist['r_time']])
        r_h1 = X_eiscat['r_h']
        r_h2 = X_artist['r_h']
        r_param1 = X_eiscat['r_param']
        r_param2 = X_artist['r_param']
        
        
        # Date
        date_str = r_time1[0].strftime('%Y-%m-%d')
        
        
        # Creating the plots
        fig, ax = plt.subplots(1, 2, figsize=(12, 8), sharey=True)
        fig.suptitle(f'Date: {date_str}', fontsize=20)
        fig.tight_layout()
        fig.autofmt_xdate()
        x_limits = [r_time1[0], r_time1[-1]]
        
    
        # Plotting original data
        ne_EISCAT = ax[0].pcolormesh(r_time1, r_h1.flatten(), np.log10(r_param1), shading='auto', cmap='turbo', vmin=10, vmax=12)
        ax[0].set_title('EISCAT UHF', fontsize=17)
        ax[0].set_xlabel('Time [hours]', fontsize=13)
        ax[0].set_ylabel('Altitude [km]', fontsize=15)
        ax[0].xaxis.set_major_formatter(DateFormatter('%H:%M'))
        
        
        # Plotting original data
        ne_Artist = ax[1].pcolormesh(r_time2, r_h2.flatten(), np.log10(r_param2), shading='auto', cmap='turbo', vmin=10, vmax=12)
        ax[1].set_title('Artist', fontsize=17)
        ax[1].set_xlabel('Time [hours]', fontsize=13)
        ax[1].xaxis.set_major_formatter(DateFormatter('%H:%M'))
        ax[1].set_xlim(x_limits)
        
        # Add colorbar for the original data
        cbar = fig.colorbar(ne_EISCAT, ax=ax[1], orientation='vertical', fraction=0.053, pad=0.04)
        cbar.set_label(r'$log_{10}(n_e)$ [n/cm$^3$]', fontsize=17)
        
        
        # Display the plots
        plt.show()