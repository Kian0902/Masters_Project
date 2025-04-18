# -*- coding: utf-8 -*-
"""
Created on Sun Dec 15 12:26:32 2024

@author: Kian Sartipzadeh
"""

import numpy as np
import matplotlib.pyplot as plt

from datetime import datetime
from matplotlib.dates import DateFormatter
from matplotlib.gridspec import GridSpec
import seaborn as sns
# For all plots: sns.set(style="dark", context=None, palette=None)
# For single plot: with sns.axes_style("dark"):



class EChaimPlotting:
    def __init__(self, X_EISCAT, X_ECHAIM):
        self.X_EISCAT = X_EISCAT
        self.X_ECHAIM = X_ECHAIM
        
    def plot_profiles(self):
        
        data = self.X_ECHAIM
        
        # Convert time arrays to datetime objects
        r_time = np.array([datetime(year, month, day, hour, minute, second) 
                                for year, month, day, hour, minute, second in data['r_time']])
        r_h = data['r_h']
        r_param = data['r_param']
        
        
        
        # Date
        # date_str = r_time[0].strftime('%Y-%m-%d')
        
        with sns.axes_style("dark"):
            # Creating the plots
            fig, ax = plt.subplots(figsize=(6, 6))
            # fig.suptitle(f'Date: {date_str}', fontsize=19, y=1.02)
            fig.tight_layout()
            # fig.autofmt_xdate()
            
            ax.set_title('All Profiles', fontsize=17)
            ax.set_xlabel(r'$log_{10}(n_e)$ $[n\,m^{-3}]$', fontsize=13)
            ax.set_ylabel('Altitude [km]', fontsize=15)
            ax.grid(True)
            ax.tick_params(axis='x', rotation=0)
            for i in range(0, r_param.shape[1]):
                
                ax.plot(np.log10(r_param[:, i].T), r_h.flatten())
                
            
            plt.show()   
    
    
    def plot_data(self):
        """
        Plot a comparison of original and averaged data using pcolormesh.
    
        Input (type)                 | DESCRIPTION
        ------------------------------------------------
        original_data (dict)         | Dictionary containing the original data.
        """
        
        
        data = self.X_ECHAIM
        
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
        
    
    
    
    
    
    def plot_eiscat_vs_echaim(self):
        """
        Plot a comparison of original and averaged data using pcolormesh.
    
        Input (type)                 | DESCRIPTION
        ------------------------------------------------
        original_data (dict)         | Dictionary containing the original data.
        """
        
        X_eiscat = self.X_EISCAT
        X_echaim = self.X_ECHAIM
        
        
        # Convert time arrays to datetime objects
        r_time1 = np.array([datetime(year, month, day, hour, minute, second) 
                                for year, month, day, hour, minute, second in X_eiscat['r_time']])
        # Convert time arrays to datetime objects
        r_time2 = np.array([datetime(year, month, day, hour, minute, second) 
                                for year, month, day, hour, minute, second in X_echaim['r_time']])
        r_h1 = X_eiscat['r_h']
        r_h2 = X_echaim['r_h']
        r_param1 = np.log10(X_eiscat['r_param'])
        r_param2 = X_echaim['r_param']
        
        
        
        # Date
        date_str = r_time1[0].strftime('%Y-%m-%d')
        
        
        # Create a grid layout
        fig = plt.figure(figsize=(12, 6))
        gs = GridSpec(1, 3, width_ratios=[1, 1, 0.05], wspace=0.1)
        
        # Shared y-axis setup
        ax0 = fig.add_subplot(gs[0])
        ax1 = fig.add_subplot(gs[1], sharey=ax0)
        cax = fig.add_subplot(gs[2])
        
        fig.suptitle(f'Date: {date_str}', fontsize=20, y=1.03)
        
        
        # # Creating the plots
        # fig, ax = plt.subplots(1, 2, figsize=(12, 8), sharey=True)
        # fig.suptitle(f'Date: {date_str}', fontsize=20)
        # fig.tight_layout()
        # fig.autofmt_xdate()
        x_limits = [r_time1[0], r_time1[-1]]
        
    
        # Plotting original data
        ne_EISCAT = ax0.pcolormesh(r_time1, r_h1.flatten(), r_param1, shading='auto', cmap='turbo', vmin=10, vmax=12)
        ax0.set_title('EISCAT UHF', fontsize=17)
        ax0.set_xlabel('Time [hh:mm]', fontsize=13)
        ax0.set_ylabel('Altitude [km]', fontsize=15)
        ax0.xaxis.set_major_formatter(DateFormatter('%H:%M'))
        
        
        # Plotting original data
        ne_ECHAIM = ax1.pcolormesh(r_time2, r_h2.flatten(), np.log10(r_param2), shading='auto', cmap='turbo', vmin=10, vmax=12)
        ax1.set_title('E-Chaim', fontsize=17)
        ax1.set_xlabel('Time [hh:mm]', fontsize=13)
        ax1.xaxis.set_major_formatter(DateFormatter('%H:%M'))
        ax1.tick_params(labelleft=False)
        ax1.set_xlim(x_limits)
        
        for ax in [ax0, ax1]:
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='center')
        
        
        # Add colorbar for the original data
        cbar = fig.colorbar(ne_EISCAT, cax=cax, orientation='vertical', fraction=0.053, pad=0.04)
        cbar.set_label(r'$log_{10}(n_e)$ $[n\,m^{-3}]$', fontsize=17)
        
        
        # Display the plots
        plt.show()
        
        
        
        
        
        
        
        
        
        
        
        