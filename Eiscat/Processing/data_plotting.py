# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 17:16:54 2024

@author: Kian Sartipzadeh
"""



import numpy as np
import matplotlib.pyplot as plt
# from matplotlib.colors import ListedColormap, Normalize
from matplotlib.dates import DateFormatter
# import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
# from matplotlib.widgets import Cursor
from datetime import datetime

from data_utils import from_array_to_datetime, inspect_dict
import random

# import seaborn as sns
# sns.set(style="dark", context=None, palette=None)



class EISCATPlotter:
    def __init__(self, X_EISCAT):
        self.X_EISCAT = X_EISCAT
        self.selected_indices = []
    
    
    def plot_all_interval(self, interval: list):
        """
        Plot all measurements within the given time interval.

        Parameters:
        interval (list): A list of two strings representing the start and end of the time interval,
                         e.g., ["20190105_1045", "20190105_1200"].
        """
        if not interval or len(interval) != 2:
            print("Interval must contain two elements: [start_time, end_time].")
            return

        # Convert interval strings to datetime objects
        start_time = datetime.strptime(interval[0], "%Y%m%d_%H%M")
        end_time = datetime.strptime(interval[1], "%Y%m%d_%H%M")

        # Extract data
        r_time = from_array_to_datetime(self.X_EISCAT["r_time"])  # Array of datetime objects
        r_h = self.X_EISCAT["r_h"].flatten()  # Altitude points
        r_param = self.X_EISCAT["r_param"]  # Electron density data
        r_error = self.X_EISCAT["r_error"]  # Electron density data
        
        # Find indices within the specified time interval
        selected_indices = [i for i, t in enumerate(r_time) if start_time <= t <= end_time]

        if not selected_indices:
            print("No measurements found within the specified interval.")
            return

        # Plot each measurement
        plt.figure(figsize=(5, 7))
        for idx in selected_indices:
            plt.plot(r_param[:, idx], r_h, label=f"{r_time[idx].strftime('%H:%M:%S')}")
            plt.fill_betweenx(r_h, r_param[:, idx] - r_error[:, idx], r_param[:, idx] + r_error[:, idx], alpha=0.3)
        
        # Customize plot
        # plt.xscale("log")
        plt.xlim(xmin=0, xmax=1.3e11)
        plt.ylim(ymin=90, ymax=400)
        plt.xlabel("Electron Density (m^-3)", fontsize=14)
        plt.ylabel("Altitude (km)", fontsize=14)
        plt.title(f"EISCAT Measurements from {start_time.strftime('%Y-%m-%d %H:%M')} \
                  to {end_time.strftime('%Y-%m-%d %H:%M')}", fontsize=16)
        plt.legend(title="Measurement Time", loc="best", fontsize=10)
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    
    
    
    
    
    def select_measurements(self, n):
        """
        Randomly select n measurements and store their indices.
        """
        M = self.X_EISCAT["r_time"].shape[0]
        self.selected_indices = random.sample(range(M), n)
    
    
    def select_measurements_by_datetime(self, datetimes):
        """
        Select measurements by providing a list of datetime objects.
        """
        r_time = from_array_to_datetime(self.X_EISCAT["r_time"])
        
        self.selected_indices = [i for i, t in enumerate(r_time) if t in datetimes]
    
    
    
    def plot_selected_measurements(self):
        """
        Plot the selected measurements from all three radars in a 1xn grid of subplots.
        """
        if not self.selected_indices:
            print("No measurements selected. Please run select_measurements(n) or select_measurements_by_datetime(datetimes) first.")
            return
        
        
        
        r_time = from_array_to_datetime(self.X_EISCAT["r_time"])
        r_h = self.X_EISCAT["r_h"].flatten()
        n = len(self.selected_indices)
        
        fig, axes = plt.subplots(1, n, figsize=(5*n, 7), sharey=True)
        
        if n == 1:
            axes = [axes]  # Ensure axes is iterable if there's only one subplot
        
        for ax, idx in zip(axes, self.selected_indices):
            ax.plot(self.X_EISCAT["r_param"][:, idx], r_h, label='EISCAT', linestyle='-')
            
            time_str = r_time[idx].strftime('%H:%M')
            ax.set_xlabel(r'$n_e$ [$n/cm^3$]', fontsize=13)
            ax.set_title(f'Time: {time_str}', fontsize=15)
            ax.grid(True)
            ax.legend()
        
        axes[0].set_ylabel('Altitude [km]', fontsize=13)
        
        date_str = r_time[idx].strftime('%Y-%m-%d')
        fig.suptitle(f'EISCAT {n} Chosen Times\nDate: {date_str}', fontsize=20)
        plt.tight_layout()
        plt.show()
    
    
    

    
    
    
    def plot_day(self):
        
        
        r_time = from_array_to_datetime(self.X_EISCAT["r_time"])
        r_h = self.X_EISCAT["r_h"].flatten()
        
        ne_eis = np.log10(self.X_EISCAT["r_param"])
        ne_err = np.log10(self.X_EISCAT["r_error"])
        
        
        date_str = r_time[0].strftime('%Y-%m-%d')
        
        
        
        
        fig, ax = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        
        # Create a grid layout
        fig.suptitle(f'Date: {date_str}', fontsize=20, y=1.0)
        
        # Plotting EISCAT
        plot_eis = ax[0].pcolormesh(r_time, r_h, ne_eis, shading='auto', cmap='turbo', vmin=10, vmax=12)
        ax[0].set_title('EISCAT UHF', fontsize=17)
        # ax[0].set_xlabel('Time [hours]', fontsize=13)
        ax[0].set_ylabel('Altitude [km]', fontsize=15)
        ax[0].xaxis.set_major_formatter(DateFormatter('%H:%M'))
        ax[0].tick_params(labelbottom=False)
        fig.colorbar(plot_eis, ax=ax[0])
        plt.setp(ax[0].xaxis.get_majorticklabels(), rotation=45, ha='center')
        
        # Plotting DL model
        plot_err = ax[1].pcolormesh(r_time, r_h, ne_err, shading='auto', cmap='bwr', vmin=8, vmax=11)
        ax[1].set_title('Error', fontsize=17)
        ax[1].set_xlabel('Time [hours]', fontsize=13)
        ax[1].set_ylabel('Altitude [km]', fontsize=15)
        ax[1].xaxis.set_major_formatter(DateFormatter('%H:%M'))
        fig.colorbar(plot_err, ax=ax[1])
        plt.setp(ax[1].xaxis.get_majorticklabels(), rotation=45, ha='center')
        
        
        
        # Add colorbar
        # cbar = fig.colorbar(ne_EISCAT, cax=ax[0], orientation='vertical')
        # cbar.set_label(r'$log_{10}(n_e)$ [n/cm$^3$]', fontsize=17)
        
        
        plt.show()
    



    def plot_day_temp(self):
        
        
        r_time = from_array_to_datetime(self.X_EISCAT["r_time"])
        r_h = self.X_EISCAT["r_h"].flatten()
        
        ne_eis = np.log10(self.X_EISCAT["r_param"])
        ne_err = np.log10(self.X_EISCAT["r_error"])
        ne_systemp = self.X_EISCAT["r_systemp"]
        
        date_str = r_time[0].strftime('%Y-%m-%d')
        
        
        
        
        # Create a grid layout
        fig = plt.figure(figsize=(14, 9))
        gs = GridSpec(3, 2, width_ratios=[1, 0.02], hspace=0.3, wspace=0.05)
        
        # Create a grid layout
        fig.suptitle(f'Date: {date_str}', fontsize=20, y=1.0)
        
        # Shared y-axis setup
        ax0 = fig.add_subplot(gs[0, 0])
        ax1 = fig.add_subplot(gs[1, 0], sharex=ax0)
        ax2 = fig.add_subplot(gs[2, 0], sharex=ax0)
        cax0 = fig.add_subplot(gs[0, 1])
        cax1 = fig.add_subplot(gs[1, 1])
        
        
        # Plotting EISCAT
        plot_eis = ax0.pcolormesh(r_time, r_h, ne_eis, shading='auto', cmap='turbo', vmin=10, vmax=12)
        ax0.set_title('EISCAT UHF', fontsize=17)
        # ax0.set_xlabel('Time [hours]', fontsize=13)
        ax0.set_ylabel('Altitude [km]', fontsize=15)
        ax0.xaxis.set_major_formatter(DateFormatter('%H:%M'))
        ax0.tick_params(labelbottom=False)
        fig.colorbar(plot_eis, cax=cax0)
        # plt.setp(ax0.xaxis.get_majorticklabels(), rotation=45, ha='center')
        
        # Plotting DL model
        plot_err = ax1.pcolormesh(r_time, r_h, ne_err, shading='auto', cmap='bwr', vmin=8, vmax=11)
        ax1.set_title('Error', fontsize=17)
        # ax1.set_xlabel('Time [hours]', fontsize=13)
        ax0.set_ylabel('Altitude [km]', fontsize=15)
        ax1.xaxis.set_major_formatter(DateFormatter('%H:%M'))
        ax1.tick_params(labelbottom=False)
        fig.colorbar(plot_err, cax=cax1)
        # plt.setp(ax[1].xaxis.get_majorticklabels(), rotation=45, ha='center')
        
        
        # Plotting DL model
        ax2.plot(r_time, ne_systemp[0])
        ax2.set_title('System Temperature', fontsize=17)
        # ax2.set_xlabel('Time [hours]', fontsize=13)
        ax0.set_ylabel('Temp', fontsize=15)
        ax2.xaxis.set_major_formatter(DateFormatter('%H:%M'))
        # ax[2].tick_params(labelleft=False)
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='center')
        
        
        
        # Add colorbar
        # cbar = fig.colorbar(ne_EISCAT, cax=ax[0], orientation='vertical')
        # cbar.set_label(r'$log_{10}(n_e)$ [n/cm$^3$]', fontsize=17)
        
        
        plt.show()





