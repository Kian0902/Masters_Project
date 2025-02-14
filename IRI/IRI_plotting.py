# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 16:43:09 2025

@author: Kian Sartipzadeh
"""



import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.dates import DateFormatter

from datetime import datetime


from utils import from_array_to_datetime

class IRIPlotter:
    def __init__(self, X_IRI):
        self.X_IRI = X_IRI
        
    
    
    def plot_profile(self):
        
        r_iri = self.X_IRI["r_h"]
        ne_iri = self.X_IRI["r_param"][:, 55]
        
        with sns.axes_style("dark"):
            fig, ax = plt.subplots()
            
            ax.set_title("Electron Density Profile")
            ax.plot(ne_iri, r_iri, label=r'$n_e$')
            ax.set_xlabel(r'Electron Density [$n\,m^{-3}$]')
            ax.set_ylabel(r'$z$ [$km$]')
            ax.grid(True)
            ax.legend()
            plt.show()
            
            
    
    def plot_profiles(self):
        
        
        r_iri = self.X_IRI["r_h"]
        ne_iri = self.X_IRI["r_param"]
        
        with sns.axes_style("dark"):
            fig, ax = plt.subplots()
            
            ax.set_title("Electron Density Profiles")
            ax.plot(ne_iri, r_iri)
            ax.set_xlabel(r'Electron Density [$n\,m^{-3}$]')
            ax.set_ylabel(r'$z$ [$km$]')
            ax.grid(True)
            plt.show()
    


    def plot_day(self):
        iri_time = from_array_to_datetime(self.X_IRI["r_time"])
        r_iri = self.X_IRI["r_h"]
        ne_iri = np.log10(self.X_IRI["r_param"])
        
        
        date_str = iri_time[0].strftime('%Y-%m-%d')
        
        # Create a grid layout
        fig = plt.figure()#figsize=(15, 5))
        fig.suptitle(f'Date: {date_str}')
        gs = GridSpec(1, 2, width_ratios=[1, 0.05], wspace=0.1)
        
        
        ax0 = fig.add_subplot(gs[0])
        cax = fig.add_subplot(gs[1])
        
        
        ne_IRI = ax0.pcolormesh(iri_time, r_iri.flatten(), ne_iri, shading='auto', cmap='turbo', vmin=10, vmax=12)
        ax0.set_xlabel('Time [hh:mm]')
        ax0.set_ylabel('Altitude [km]')
        ax0.xaxis.set_major_formatter(DateFormatter('%H:%M'))
        # ax0.tick_params(labelleft=False)
        plt.setp(ax0.xaxis.get_majorticklabels(), rotation=45, ha='center')
          
        
        # Add colorbar
        cbar = fig.colorbar(ne_IRI, cax=cax, orientation='vertical')
        cbar.set_label(r'$log_{10}(n_e)$ [$n\,m^{-3}$]')
        plt.show()



    def plot_before_vs_after(self, X_after):
        """
        Plot a comparison of original and averaged data using pcolormesh.
    
        Input (type)                 | DESCRIPTION
        ------------------------------------------------
        original_data (dict)         | Dictionary containing the original data.
        """
        
        
        iri_time = from_array_to_datetime(self.X_IRI["r_time"])
        r_iri = self.X_IRI["r_h"]
        ne_iri = np.log10(self.X_IRI["r_param"])
        
        iri_time_inter = from_array_to_datetime(X_after["r_time"])
        r_iri_inter = X_after["r_h"]
        ne_iri_inter = np.log10(X_after["r_param"])
        
        
        # Date
        date_str = iri_time[0].strftime('%Y-%m-%d')
        
        
        # Create a grid layout
        fig = plt.figure(figsize=(12, 6))
        gs = GridSpec(1, 3, width_ratios=[1, 1, 0.05])
        
        # Shared y-axis setup
        ax0 = fig.add_subplot(gs[0])
        ax1 = fig.add_subplot(gs[1])
        cax = fig.add_subplot(gs[2])
        
        fig.suptitle(f'Date: {date_str}', fontsize=20, y=1.03)
        
        
        # Plotting original data
        ne_before = ax0.pcolormesh(iri_time, r_iri.flatten(), ne_iri, shading='auto', cmap='turbo', vmin=10, vmax=12)
        ax0.set_title('Before Downsampling', fontsize=17)
        ax0.set_xlabel('Time [hh:mm]', fontsize=13)
        ax0.set_ylabel('Altitude [km]', fontsize=15)
        ax0.xaxis.set_major_formatter(DateFormatter('%H:%M'))
        
        
        # Plotting original data
        ne_after = ax1.pcolormesh(iri_time_inter, r_iri_inter.flatten(), ne_iri_inter, shading='auto', cmap='turbo', vmin=10, vmax=12)
        ax1.set_title('After Downsampling', fontsize=17)
        ax1.set_xlabel('Time [hh:mm]', fontsize=13)
        ax1.xaxis.set_major_formatter(DateFormatter('%H:%M'))
        # ax1.set_xlim(x_limits)
        
        for ax in [ax0, ax1]:
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='center')
        
        
        # Add colorbar for the original data
        cbar = fig.colorbar(ne_before, cax=cax, orientation='vertical', fraction=0.053, pad=0.04)
        cbar.set_label(r'$log_{10}(n_e)$ $[n\,m^{-3}]$', fontsize=17)
        plt.show()











