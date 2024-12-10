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
# import random

# import seaborn as sns
# sns.set(style="dark", context=None, palette=None)



class EISCATPlotter:
    def __init__(self, X_EISCAT):
        self.X_EISCAT = X_EISCAT
        self.selected_indices = []
        
        
        
        
    # def plot_eis(self, ax=None):
    #     r_time = from_array_to_datetime(self.X_EISCAT["r_time"])
    #     r_h = self.X_EISCAT["r_h"].flatten()
    #     ne_eis = np.log10(self.X_EISCAT["r_param"])
    
    #     if ax is None:
    #         fig, ax = plt.subplots()
    #     else:
    #         fig = None  # we only have ax, no new figure
    
    #     # Perform the plotting on the given ax
    #     p = ax.pcolormesh(r_time, r_h, ne_eis, shading='auto', cmap='turbo', vmin=10, vmax=12)
    #     ax.set_title('EISCAT UHF', fontsize=17)
    #     ax.set_xlabel('Time [hours]', fontsize=13)
    #     ax.set_ylabel('Altitude [km]', fontsize=15)
    #     ax.xaxis.set_major_formatter(DateFormatter('%H:%M'))
        
    #     if fig is not None:
    #         # Add colorbar and other features if we created a new figure
    #         fig.colorbar(p, ax=ax)
    #         plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='center')
    #         return fig, ax
    #     else:
    #         return ax
    
    
    # def plot_err(self, ax=None):
    #     r_time = from_array_to_datetime(self.X_EISCAT["r_time"])
    #     r_h = self.X_EISCAT["r_h"].flatten()
    #     ne_err = np.log10(self.X_EISCAT["r_error"])
    
    #     if ax is None:
    #         fig, ax = plt.subplots()
    #     else:
    #         fig = None
    
    #     # Perform the plotting on the given ax
    #     p = ax.pcolormesh(r_time, r_h, ne_err, shading='auto', cmap='bwr', vmin=8, vmax=11)
    #     ax.set_title('Error', fontsize=17)
    #     ax.set_xlabel('Time [hours]', fontsize=13)
    #     ax.xaxis.set_major_formatter(DateFormatter('%H:%M'))
    
    #     if fig is not None:
    #         fig.colorbar(p, ax=ax)
    #         plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='center')
    #         return fig, ax
    #     else:
    #         return ax
    
    
    # def plot_day(self):
    #     fig, axs = plt.subplots(1, 2, figsize=(16, 8), sharey=True)
        
    #     # Now both subplots will be drawn onto existing axes
    #     self.plot_eis(ax=axs[0])
    #     self.plot_err(ax=axs[1])
        
    #     # plt.tight_layout()
    #     plt.show()

        
        
        
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





