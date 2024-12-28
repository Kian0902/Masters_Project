# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 14:31:51 2024

@author: Kian Sartipzadeh
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, Normalize
from matplotlib.dates import DateFormatter
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
from matplotlib.widgets import Cursor
from datetime import datetime

from eval_utils import from_array_to_datetime, get_altitude_r2_score, get_measurements_r2_score, get_altitude_r2_score_nans, get_measurements_r2_score_nans
import random
from scipy.stats import linregress
import seaborn as sns
# For all plots: sns.set(style="dark", context=None, palette=None)
# For single plot: with sns.axes_style("dark"):

from sklearn.metrics import r2_score

from matplotlib.ticker import FuncFormatter

# Define a custom formatter for the x-axis ticks
def format_ticks(x, _):
    if x.is_integer():
        return f'{int(x)}'  # Convert integers to display without decimals
    return f'{x}'  # Keep floats as is

formatter = FuncFormatter(format_ticks)



class RadarPlotter:
    def __init__(self, X_EISCAT, X_HNN, X_Artist, X_Ionogram):
        self.X_EISCAT = X_EISCAT
        self.X_HNN = X_HNN
        self.X_Artist = X_Artist
        self.X_Ionogram = X_Ionogram
        self.selected_indices = []
    
    
    def plot_compare_all_r2_window(self):
        r_time = from_array_to_datetime(self.X_EISCAT["r_time"])
        art_time = from_array_to_datetime(self.X_Artist["r_time"])
        r_h = self.X_EISCAT["r_h"].flatten()
        
        ne_eis = np.log10(self.X_EISCAT["r_param"])
        ne_kian = self.X_HNN["r_param"]
        ne_art  = np.log10(self.X_Artist["r_param"])
        
        kian_r2_scores_alt = get_altitude_r2_score(ne_eis, ne_kian)
        kian_r2_scores_mea = get_measurements_r2_score(ne_eis, ne_kian)
        
        art_r2_scores_alt = get_altitude_r2_score_nans(ne_eis, ne_art)
        art_r2_scores_mea = get_measurements_r2_score_nans(ne_eis, ne_art)
        
        date_str = r_time[0].strftime('%Y-%m-%d')

        
        # Create a grid layout
        fig = plt.figure(figsize=(14, 12))
        gs = GridSpec(3, 4, width_ratios=[1, 0.3, 1, 0.05], height_ratios=[1, 0.3, 1], wspace=0.2, hspace=0.2)
        fig.suptitle(f'Date: {date_str}', fontsize=20, y=0.95)
        
        
        # 3rd row
        ax20 = fig.add_subplot(gs[2, 0]) # eis          (y-axis)
        ax21 = fig.add_subplot(gs[2, 1], sharey=ax20) # kian r2 eis  (y-axis)
        ax22 = fig.add_subplot(gs[2, 2], sharey=ax20) # kian         (y-axis)
        cax23 = fig.add_subplot(gs[2, 3])
        
        
        # 2nd row
        ax10 = fig.add_subplot(gs[1, 0], sharex=ax20) # kian r2 eis (x-axis)
        ax11 = fig.add_subplot(gs[1, 1]) # Nothing     (Middle)
        ax12 = fig.add_subplot(gs[1, 2], sharex=ax20) # kian r2 eis (x-axis)
        
        
        # 1st row
        ax00 = fig.add_subplot(gs[0, 0], sharex=ax20) # kian         (y-axis)
        ax01 = fig.add_subplot(gs[0, 1], sharey=ax00) # kian r2 eis  (y-axis)
        ax02 = fig.add_subplot(gs[0, 2], sharex=ax22, sharey=ax00) # eis          (y-axis)
        cax03 = fig.add_subplot(gs[0, 3])
        
        
        
        
        # =============================================================================
        #         1st Row
        
        
        # Plotting Kian-net
        ax00.pcolormesh(r_time, r_h, ne_kian, shading='auto', cmap='turbo', vmin=10, vmax=12)
        ax00.set_title('KIAN-Net', fontsize=17)
        ax00.set_ylabel('Altitude [km]', fontsize=15)
        ax00.xaxis.set_major_formatter(DateFormatter('%H:%M'))
        ax00.tick_params(labelbottom=False)
        
        
        # KIAN-Net r2-scores altitude
        ax01.plot(kian_r2_scores_alt, r_h, color='C0', label=r'$R^2$')
        ax01.set_title(r'$R^2$', fontsize=17)
        ax01.set_xlabel(r'$R^2$', fontsize=13)
        ax01.grid(True)
        # ax1.legend()
        ax01.set_xlim(xmin=-0.1, xmax=1.1)
        ax01.tick_params(labelleft=False)
        
        # Plotting EISCAT
        ne_EISCAT = ax02.pcolormesh(r_time, r_h, ne_eis, shading='auto', cmap='turbo', vmin=10, vmax=12)
        ax02.set_title('EISCAT UHF', fontsize=17)
        ax02.xaxis.set_major_formatter(DateFormatter('%H:%M'))
        ax02.tick_params(labelleft=False, labelbottom=False)
        
        #
        # =============================================================================
        
        
        # =============================================================================
        #         2nd Row
        
        
        # KIAN-Net r2-scores measurements
        ax10.plot(r_time, kian_r2_scores_mea, color='C0', label=r'$R^2$')
        # ax10.set_title(r'$R^2$', fontsize=17)
        ax10.set_ylabel(r'$R^2$', fontsize=13)
        ax10.grid(True)
        ax10.xaxis.set_major_formatter(DateFormatter('%H:%M'))
        ax10.set_ylim(ymin=-0.1, ymax=1.1)
        ax10.tick_params(labelbottom=False)
        
        
        # Plot in ax11
        ax11.axis('off')  # Turn the axis back on
        
        
        # Artist 4.5 r2-scores measurements
        ax12.plot(art_time, art_r2_scores_mea, color='C1', label=r'$R^2$')
        ax12.grid(True)
        ax12.set_ylim(ymin=-0.1, ymax=1.1)
        ax12.tick_params(labelbottom=False)
        ax12.yaxis.tick_right()  # Display ticks on the right
        ax12.yaxis.set_label_position("right")
        ax12.set_ylabel(r'$R^2$', fontsize=13, rotation=270, labelpad=20)
        
        #
        # =============================================================================
        
        
        # =============================================================================
        #         3rd Row
        
        
        # Plotting EISCAT
        ax20.pcolormesh(r_time, r_h, ne_eis, shading='auto', cmap='turbo', vmin=10, vmax=12)
        ax20.set_title('EISCAT UHF', fontsize=17)
        ax20.set_xlabel('Time [hours]', fontsize=13)
        ax20.set_ylabel('Altitude [km]', fontsize=15)
        ax20.xaxis.set_major_formatter(DateFormatter('%H:%M'))
        
        
        # Artist 4.5 r2-scores altitude
        ax21.plot(art_r2_scores_alt, r_h, color='C1', label=r'$R^2$')
        ax21.set_title(r'$R^2$', fontsize=17)
        ax21.set_xlabel(r'$R^2$', fontsize=13)
        ax21.grid(True)
        ax21.set_xlim(xmin=-0.1, xmax=1.1)
        ax21.tick_params(labelleft=False)
        
        
        # Plotting Artist 4.5
        ax22.pcolormesh(art_time, r_h, ne_art, shading='auto', cmap='turbo', vmin=10, vmax=12)
        ax22.set_title('Artist 4.5', fontsize=17)
        ax22.set_xlabel('Time [hours]', fontsize=13)
        ax22.xaxis.set_major_formatter(DateFormatter('%H:%M'))
        ax22.tick_params(labelleft=False)
        
        #
        # =============================================================================
        
        
        # Rotate x-axis labels
        for ax in [ax00, ax01, ax02, ax10, ax11, ax12, ax20, ax21, ax22]:
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='center')
          
        
        # Add colorbar
        cbar03 = fig.colorbar(ne_EISCAT, cax=cax03, orientation='vertical')
        cbar03.set_label(r'$log_{10}(n_e)$ [n/m$^3$]', fontsize=17)
        
        # Add colorbar to 
        cbar23 = fig.colorbar(ne_EISCAT, cax=cax23, orientation='vertical')
        cbar23.set_label(r'$log_{10}(n_e)$ [n/m$^3$]', fontsize=17)
        
        plt.show()
    
        
    def get_mape(self, y_true, y_pred):
        """
        Calculate the Mean Absolute Percentage Error (MAPE) for each altitude.
        
        Parameters:
        - y_true: 2D numpy array of true values (N x M, where N=altitudes, M=time points).
        - y_pred: 2D numpy array of predicted values (N x M).
        
        Returns:
        - mape_per_altitude: 1D numpy array of MAPE for each altitude (N).
        """
        epsilon = 1e-10  # To avoid division by zero
        mape_per_altitude = np.mean(np.abs((y_true - y_pred) / (y_true + epsilon)), axis=1) * 100
        return mape_per_altitude
    
    def get_nmse(self, y_true, y_pred):
        """
        Calculate the Normalized Mean Squared Error (NMSE) for each altitude.
        
        Parameters:
        - y_true: 2D numpy array of true values (N x M, where N=altitudes, M=time points).
        - y_pred: 2D numpy array of predicted values (N x M).
        
        Returns:
        - nmse_per_altitude: 1D numpy array of NMSE for each altitude (N).
        """
        nmse_per_altitude = np.sum((y_true - y_pred) ** 2, axis=1) / np.sum(y_true ** 2, axis=1)
        return nmse_per_altitude
        
    def get_rmse(self, ne_true, ne_pred):
        """
        Calculate the RMSE for each altitude.
    
        Parameters:
        - ne_actual: 2D numpy array of actual values (log10(n_e))
        - ne_predicted: 2D numpy array of predicted values (log10(n_e))
    
        Returns:
        - rmse: 1D numpy array of RMSE for each altitude
        """
        rmse = np.sqrt(np.mean((ne_true - ne_pred) ** 2, axis=1))
        return rmse
        
    
    def plot_compare_r2(self):
        r_time = from_array_to_datetime(self.X_EISCAT["r_time"])
        r_h = self.X_EISCAT["r_h"].flatten()
        
        ne_eis = np.log10(self.X_EISCAT["r_param"])
        ne_kian = self.X_HNN["r_param"]
        
        r2_scores = get_altitude_r2_score(ne_eis, ne_kian)
        rmse = self.get_rmse(ne_eis, ne_kian)
        mape = self.get_mape(ne_eis, ne_kian)
        nmse = self.get_nmse(ne_eis, ne_kian)
        
        # print(rmse)
        # print(mape)
        # print(nmse)
        
        date_str = r_time[0].strftime('%Y-%m-%d')

        
        # Creating the plots
        fig, ax = plt.subplots(1, 3, figsize=(16, 7), gridspec_kw={'width_ratios': [1, 0.25, 1]}, sharey=True)
        fig.suptitle(f'Date: {date_str}', fontsize=20, y=1.02)
        fig.tight_layout()

        # Plotting original data
        ne_EISCAT = ax[0].pcolormesh(r_time, r_h, ne_eis, shading='auto', cmap='turbo', vmin=10, vmax=12)
        ax[0].set_title('EISCAT UHF', fontsize=17)
        ax[0].set_xlabel('Time [hours]', fontsize=13)
        ax[0].set_ylabel('Altitude [km]', fontsize=15)
        ax[0].xaxis.set_major_formatter(DateFormatter('%H:%M'))

        # Plotting R2-scores and RMSE as a line plot
        ax[1].plot(r2_scores, r_h, color='C0', label=r'$R^2$')
        ax[1].plot(rmse, r_h, color='C1', label=r'$RMSE$')
        ax[1].set_title(r'Eval Metrics', fontsize=17)
        # ax[1].set_xlabel(r'$R^2$', fontsize=13)
        ax[1].grid()
        ax[1].legend()
        ax[1].set_xlim(xmin=-0.1, xmax=1.1)
        
        # Plotting predicted data
        ax[2].pcolormesh(r_time, r_h, ne_kian, shading='auto', cmap='turbo', vmin=10, vmax=12)
        ax[2].set_title('KIAN-Net', fontsize=17)
        ax[2].set_xlabel('Time [hours]', fontsize=13)
        ax[2].xaxis.set_major_formatter(DateFormatter('%H:%M'))
        
        # Rotate x-axis labels for ax[0] and ax[2]
        for i, a in enumerate(ax):
            if i != 1:  # Skip ax[1]
                plt.setp(a.xaxis.get_majorticklabels(), rotation=45, ha='center')
            
        
        # Add colorbar for the predicted data
        cbar = fig.colorbar(ne_EISCAT, ax=ax[2], orientation='vertical', fraction=0.048, pad=0.04)
        cbar.set_label(r'$log_{10}(n_e)$ [n/m$^3$]', fontsize=17, labelpad=15)
        
        plt.show()
    
    
    
    def chi_squared(self, X_eis, X_eis_err, X_kian):
        chi_2 =((10**X_eis - 10**X_kian)**2)/((10**X_eis_err)**2) 
        return chi_2
    
    def error_sigma(self, X_eis, X_kian):
        err =  ((X_kian - X_eis)**2)/(X_eis)**2
        return err
    
    
    def plot_error_and_chi2(self):
        r_time = from_array_to_datetime(self.X_EISCAT["r_time"])
        r_h = self.X_EISCAT["r_h"].flatten()
        
        ne_eis = np.log10(self.X_EISCAT["r_param"])
        ne_eis_err = np.log10(self.X_EISCAT["r_error"])
        ne_kian = self.X_HNN["r_param"]
        
        ne_chi_2 = self.chi_squared(ne_eis, ne_eis_err, ne_kian)
        ne_err = self.error_sigma(ne_eis, ne_kian)
        
        
        
        date_str = r_time[0].strftime('%Y-%m-%d')

        
        # Creating the plots
        fig, ax = plt.subplots(1, 2, figsize=(16, 7), sharey=True)
        fig.suptitle(f'Date: {date_str}', fontsize=20, y=1.02)
        fig.tight_layout()

        # Plotting original data
        plot_chi2 = ax[0].pcolormesh(r_time, r_h, ne_chi_2, shading='auto', cmap='bwr')
        ax[0].set_title(r'$\chi^2$-scores', fontsize=17)
        ax[0].set_xlabel('Time [hours]', fontsize=13)
        ax[0].set_ylabel('Altitude [km]', fontsize=15)
        ax[0].xaxis.set_major_formatter(DateFormatter('%H:%M'))
        
        
        # Plotting original data
        plot_err = ax[1].pcolormesh(r_time, r_h, ne_err, shading='auto', cmap='jet')
        ax[1].set_title(r'err', fontsize=17)
        ax[1].set_xlabel('Time [hours]', fontsize=13)
        # ax[1].set_ylabel('Altitude [km]', fontsize=15)
        ax[1].xaxis.set_major_formatter(DateFormatter('%H:%M'))
        
        
        # Rotate x-axis labels for ax[0] and ax[2]
        for i, a in enumerate(ax):
            if i != 1:  # Skip ax[1]
                plt.setp(a.xaxis.get_majorticklabels(), rotation=45, ha='center')
            
        
        # Add colorbar for the predicted data
        cbar_chi = fig.colorbar(plot_chi2, ax=ax[0], orientation='vertical', fraction=0.048, pad=0.04)
        cbar_chi.set_label(r'$log_{10}(n_e)$ [n/m$^3$]', fontsize=17, labelpad=15)
        
        # Add colorbar for the predicted data
        cbar_err = fig.colorbar(plot_err, ax=ax[1], orientation='vertical', fraction=0.048, pad=0.04)
        cbar_err.set_label(r'$log_{10}(n_e)$ [n/m$^3$]', fontsize=17, labelpad=15)
        
        
        plt.show()
    
    def plot_chi_squared(self):
        """
        Plot chi-squared values as a color plot and compute summary metrics.
        """
        r_time = from_array_to_datetime(self.X_EISCAT["r_time"])  # Convert time array to datetime
        r_h = self.X_EISCAT["r_h"].flatten()  # Altitudes
    
        # Log-transform observed and error data
        ne_eis = np.log10(self.X_EISCAT["r_param"])
        ne_eis_err = np.log10(self.X_EISCAT["r_error"])
        ne_kian = self.X_HNN["r_param"]  # Predicted data
    
        # Compute chi-squared values for all points
        ne_chi_2 = np.log10(self.chi_squared(ne_eis, ne_eis_err, ne_kian))
        
        
        
        # Compute chi-squared per altitude (mean over time for each altitude)
        chi_2_per_altitude = np.mean(ne_chi_2, axis=1)  # Average over columns (time)
    
        # Compute chi-squared per measurement (mean over altitude for each time)
        chi_2_per_measurement = np.mean(ne_chi_2, axis=0)  # Average over rows (altitude)
    
        date_str = r_time[0].strftime('%Y-%m-%d')
    
        # Creating the color plot for chi-squared per altitude and time
        fig, ax = plt.subplots(1, 1, figsize=(8, 7))
        fig.suptitle(f'Date: {date_str}', fontsize=20, y=1.02)
        fig.tight_layout()
    
        # Plot the full chi-squared color map
        plot_err = ax.pcolormesh(r_time, r_h, ne_chi_2, shading='auto', cmap='jet', vmin=0, vmax=2)
        ax.set_title(r'$\chi^2$-error', fontsize=17)
        ax.set_xlabel('Time [hours]', fontsize=13)
        ax.set_ylabel('Altitude [km]', fontsize=15)
        ax.xaxis.set_major_formatter(DateFormatter('%H:%M'))
    
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='center')
    
        # Add colorbar
        cbar_chi = fig.colorbar(plot_err, ax=ax, orientation='vertical', fraction=0.048, pad=0.04)
        cbar_chi.set_label(r'$\chi^2$', fontsize=17, labelpad=15)
    
        plt.show()
    
        # Plot chi-squared per altitude
        plt.figure(figsize=(8, 5))
        plt.plot(chi_2_per_altitude, r_h, marker='o')
        plt.title(r'$\chi^2$ per Altitude', fontsize=17)
        plt.xlabel('Altitude [km]', fontsize=13)
        plt.ylabel(r'$\chi^2$', fontsize=15)
        plt.grid()
        plt.show()
    
        # Plot chi-squared per measurement
        plt.figure(figsize=(8, 5))
        plt.plot(r_time, chi_2_per_measurement, marker='o')
        plt.title(r'$\chi^2$ per Measurement', fontsize=17)
        plt.xlabel('Time [hours]', fontsize=13)
        plt.ylabel(r'$\chi^2$', fontsize=15)
        plt.gca().xaxis.set_major_formatter(DateFormatter('%H:%M'))
        plt.grid()
        plt.show()
        
    # def plot_chi_squared(self):
    #     r_time = from_array_to_datetime(self.X_EISCAT["r_time"])
    #     r_h = self.X_EISCAT["r_h"].flatten()
        
    #     ne_eis = np.log10(self.X_EISCAT["r_param"])
    #     ne_eis_err = np.log10(self.X_EISCAT["r_error"])
    #     ne_kian = self.X_HNN["r_param"]
        
    #     # ne_chi_2 = self.chi_squared(ne_eis, ne_kian)
    #     ne_err = self.chi_squared(ne_eis, ne_eis_err, ne_kian)
        
        
    #     date_str = r_time[0].strftime('%Y-%m-%d')

        
    #     # Creating the plots
    #     fig, ax = plt.subplots(1, 1, figsize=(8, 7))
    #     fig.suptitle(f'Date: {date_str}', fontsize=20, y=1.02)
    #     fig.tight_layout()

    #     # Plotting original data
    #     plot_err = ax.pcolormesh(r_time, r_h, ne_err, shading='auto', cmap='jet', vmin=0.00001, vmax=0.005)
    #     ax.set_title(r'$\chi^2$-error', fontsize=17)
    #     ax.set_xlabel('Time [hours]', fontsize=13)
    #     ax.set_ylabel('Altitude [km]', fontsize=15)
    #     ax.xaxis.set_major_formatter(DateFormatter('%H:%M'))
        
        
    #     plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='center')
            
        
    #     # Add colorbar for the predicted data
    #     cbar_chi = fig.colorbar(plot_err, ax=ax, orientation='vertical', fraction=0.048, pad=0.04)
    #     cbar_chi.set_label(r'$\chi^2$', fontsize=17, labelpad=15)
        
    #     plt.show()
    
    
    def plot_compare_all(self, selected_measurements=False):
        r_time = from_array_to_datetime(self.X_EISCAT["r_time"])
        art_time = from_array_to_datetime(self.X_Artist["r_time"])
        r_h = self.X_EISCAT["r_h"].flatten()
        
        ne_eis = np.log10(self.X_EISCAT["r_param"])
        ne_hnn = self.X_HNN["r_param"]
        ne_art = np.log10(self.X_Artist["r_param"])
        
        date_str = r_time[0].strftime('%Y-%m-%d')
        
        # Create a grid layout
        fig = plt.figure(figsize=(16, 6))
        gs = GridSpec(1, 4, width_ratios=[1, 1, 1, 0.05], wspace=0.1)
        
        # Shared y-axis setup
        ax0 = fig.add_subplot(gs[0])
        ax1 = fig.add_subplot(gs[1], sharey=ax0)
        ax2 = fig.add_subplot(gs[2], sharey=ax0)
        cax = fig.add_subplot(gs[3])
        
        fig.suptitle(f'Date: {date_str}', fontsize=20, y=1.03)
        nl=""
        if selected_measurements:
            nl = "\n"
            # Highlight selected measurements
            for idx, line_num in zip(self.selected_indices, range(1, len(self.selected_indices) + 1)):
                for ax in [ax0, ax1, ax2]:
                    ax.axvline(r_time[idx], color='red', linestyle=':', linewidth=4)
                    # Add line numbers at the top of the plot
                    ax.text(
                        r_time[idx],
                        r_h[-1] + 15,  # Position above the plot
                        str(line_num),
                        color='red',
                        fontsize=15,
                        ha='center',
                        va='bottom',
                    )       
        
        
        # Plotting EISCAT
        ne_EISCAT = ax0.pcolormesh(r_time, r_h, ne_eis, shading='auto', cmap='turbo', vmin=10, vmax=12)
        ax0.set_title('EISCAT UHF'+nl, fontsize=17)
        ax0.set_xlabel('Time [hh:mm]', fontsize=13)
        ax0.set_ylabel('Altitude [km]', fontsize=15)
        ax0.xaxis.set_major_formatter(DateFormatter('%H:%M'))
        
        # Plotting DL model
        ax1.pcolormesh(r_time, r_h, ne_hnn, shading='auto', cmap='turbo', vmin=10, vmax=12)
        ax1.set_title('KIAN-Net'+nl, fontsize=17)
        ax1.set_xlabel('Time [hh:mm]', fontsize=13)
        ax1.xaxis.set_major_formatter(DateFormatter('%H:%M'))
        ax1.tick_params(labelleft=False)
        
        # Plotting Artist 4.5
        ax2.pcolormesh(art_time, r_h, ne_art, shading='auto', cmap='turbo', vmin=10, vmax=12)
        ax2.set_title('Artist 4.5'+nl, fontsize=17)
        ax2.set_xlabel('Time [hh:mm]', fontsize=13)
        ax2.xaxis.set_major_formatter(DateFormatter('%H:%M'))
        ax2.tick_params(labelleft=False)
        
        # Rotate x-axis labels
        for ax in [ax0, ax1, ax2]:
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='center')
        
        
        
        # Add colorbar
        cbar = fig.colorbar(ne_EISCAT, cax=cax, orientation='vertical')
        cbar.set_label(r'$log_{10}(n_e)$ $[n\,m^{-3}]$', fontsize=17)
        
        plt.show()
    
    
    def plot_compare_all_r2(self):
        r_time = from_array_to_datetime(self.X_EISCAT["r_time"])
        art_time = from_array_to_datetime(self.X_Artist["r_time"])
        r_h = self.X_EISCAT["r_h"].flatten()
        
        ne_eis = np.log10(self.X_EISCAT["r_param"])
        ne_kian = self.X_HNN["r_param"]
        ne_art = np.log10(self.X_Artist["r_param"])
        
        date_str = r_time[0].strftime('%Y-%m-%d')
        
        kian_r2_scores = get_altitude_r2_score(ne_eis, ne_kian)
        art_r2_scores = get_altitude_r2_score_nans(ne_eis, ne_art)
        # print(art_r2_scores)
        
        # Create a grid layout
        fig = plt.figure(figsize=(15, 5))
        gs = GridSpec(1, 6, width_ratios=[1, 0.25, 1, 0.25, 1, 0.05], wspace=0.1)
        
        # Shared y-axis setup
        ax0 = fig.add_subplot(gs[0])
        ax1 = fig.add_subplot(gs[1], sharey=ax0)
        ax2 = fig.add_subplot(gs[2], sharey=ax0)
        ax3 = fig.add_subplot(gs[3], sharey=ax0)
        ax4 = fig.add_subplot(gs[4], sharey=ax0)
        cax = fig.add_subplot(gs[5])
        
        fig.suptitle(f'Date: {date_str}', fontsize=20, y=1.03)
        
        
        # Plotting Kian-net
        ax0.pcolormesh(r_time, r_h, ne_kian, shading='auto', cmap='turbo', vmin=10, vmax=12)
        ax0.set_title('KIAN-Net', fontsize=17)
        ax0.set_xlabel('Time [hh:mm]', fontsize=13)
        ax0.set_ylabel('Altitude [km]', fontsize=15)
        ax0.xaxis.set_major_formatter(DateFormatter('%H:%M'))
        # ax0.tick_params(labelleft=False)
        
        
        # KIAN-Net r2-scores
        ax1.plot(kian_r2_scores, r_h, color='C0', label=r'$R^2$')
        ax1.set_title(r'$R^2$-score', fontsize=17)
        ax1.set_xlabel(r'$R^2$', fontsize=13)
        ax1.grid(True)
        ax1.set_xticks([0, 0.5, 1])
        ax1.xaxis.set_major_formatter(formatter) 
        ax1.set_xlim(xmin=-0.1, xmax=1.1)
        ax1.tick_params(labelleft=False)
        
        
        # Plotting EISCAT
        ne_EISCAT = ax2.pcolormesh(r_time, r_h, ne_eis, shading='auto', cmap='turbo', vmin=10, vmax=12)
        ax2.set_title('EISCAT UHF', fontsize=17)
        ax2.set_xlabel('Time [hh:mm]', fontsize=13)
        ax2.xaxis.set_major_formatter(DateFormatter('%H:%M'))
        ax2.tick_params(labelleft=False)
        
        
        # Artist 4.5 r2-scores
        ax3.plot(art_r2_scores, r_h, color='C1', label=r'$R^2$')
        ax3.set_title(r'$R^2$-score', fontsize=17)
        ax3.set_xlabel(r'$R^2$', fontsize=13)
        ax3.grid(True)
        ax3.set_xticks([0, 0.5, 1])
        ax3.xaxis.set_major_formatter(formatter) 
        ax3.set_xlim(xmin=-0.1, xmax=1.1)
        ax3.tick_params(labelleft=False)
        
        
        # Plotting Artist 4.5
        ax4.pcolormesh(art_time, r_h, ne_art, shading='auto', cmap='turbo', vmin=10, vmax=12)
        ax4.set_title('Artist 4.5', fontsize=17)
        ax4.set_xlabel('Time [hh:mm]', fontsize=13)
        ax4.xaxis.set_major_formatter(DateFormatter('%H:%M'))
        ax4.tick_params(labelleft=False)
        
        # Rotate x-axis labels
        for ax in [ax0, ax2, ax4]:
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='center')
          
        
        # Add colorbar
        cbar = fig.colorbar(ne_EISCAT, cax=cax, orientation='vertical')
        cbar.set_label(r'$log_{10}(n_e)$ [$n\,m^{-3}$]', fontsize=17)
        
        plt.show()
    
    
    
    def plot_compare_closest(self):
        r_time = from_array_to_datetime(self.X_EISCAT["r_time"])
        r_h = self.X_EISCAT["r_h"].flatten()
        
        ne_eis = np.log10(self.X_EISCAT["r_param"])
        ne_kian = self.X_HNN["r_param"]
        ne_art = np.log10(self.X_Artist["r_param"])
        
        ne_art = np.nan_to_num(ne_art, nan=0)
        
        
        h_peak_eis = self.X_EISCAT["r_h_peak"]
        h_peak_kian = self.X_HNN["r_h_peak"]
        h_peak_art = self.X_Artist["r_h_peak"]
        
        
        # Calculate absolute differences
        diff_kian = self.error_function(ne_eis, ne_kian)
        diff_art = self.error_function(ne_eis, ne_art)
        
        
        # Calculate the difference in magnitude to depict which one is closer
        diff_magnitude = diff_art - diff_kian
        
        date_str = r_time[0].strftime('%Y-%m-%d')
        
        
        
        # Create a grid layout
        fig = plt.figure(figsize=(16, 7))
        gs = GridSpec(1, 3, width_ratios=[1, 1, 0.05], wspace=0.1)
        
        # Shared y-axis setup
        ax0 = fig.add_subplot(gs[0])
        ax1 = fig.add_subplot(gs[1], sharey=ax0)
        cax = fig.add_subplot(gs[2])
        
        fig.suptitle(f'Date: {date_str}', fontsize=20, y=1.03)
        
        
        # Plotting EISCAT
        ax0.pcolormesh(r_time, r_h, ne_eis, shading='auto', cmap='turbo', vmin=10, vmax=12, zorder=2)
        ax0.set_title('EISCAT UHF', fontsize=17)
        ax0.set_xlabel('Time [hh:mm]', fontsize=13)
        ax0.set_ylabel('Altitude [km]', fontsize=15)
        ax0.xaxis.set_major_formatter(DateFormatter('%H:%M'))
        
        # Plotting E-peaks
        ax0.plot(r_time, h_peak_eis[0, :], marker='o', linestyle='-', linewidth=2, color="blue", markeredgecolor="black", markeredgewidth=1.5, label="UHF E-peak", zorder=6)
        ax0.plot(r_time, h_peak_kian[0, :], marker='o', linestyle='-', linewidth=2, color="C1", markeredgecolor="black", markeredgewidth=1.5, label="KIAN-Net E-peak", zorder=6)
        ax0.plot(r_time, h_peak_art[0, :], marker='o', linestyle='-', linewidth=2, color="lime", markeredgecolor="black", markeredgewidth=1.5, label="Artist 4.5 E-peak", zorder=6)
        
        # Plotting F-peaks
        ax0.plot(r_time, h_peak_eis[1, :], marker='o', linestyle='-', linewidth=2, markeredgecolor="black", markeredgewidth=1.5, color="blue", label="UHF E-peak", zorder=6)
        ax0.plot(r_time, h_peak_kian[1, :], marker='o', linestyle='-', linewidth=2, markeredgecolor="black", markeredgewidth=1.5, color="C1", label="KIAN-Net E-peak", zorder=6)
        ax0.plot(r_time, h_peak_art[1, :], marker='o', linestyle='-', linewidth=2, markeredgecolor="black", markeredgewidth=1.5, color="lime", label="Artist 4.5 E-peak", zorder=6)

        
        # Plot the comparison with magnitude differences
        mesh = ax1.pcolormesh(r_time, r_h, diff_magnitude, shading='auto', cmap='bwr', vmin=-0.1, vmax=0.1)
        ax1.set_title(f'KIAN-Net vs Artist 4.5   Proximity to EISCAT', fontsize=17)
        ax1.set_xlabel('Time [hours]', fontsize=13)
        ax1.xaxis.set_major_formatter(DateFormatter('%H:%M'))
        ax1.tick_params(labelleft=False)
        
        # Rotate x-axis labels
        for ax in [ax0, ax1]:
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='center')
        
        
        # Add colorbar
        cbar = fig.colorbar(mesh, cax=cax, orientation='vertical')
        cbar.set_label('(Artist 4.5 < 0 < KIAN-Net)', fontsize=13)
        
        # Add text at the top and bottom of the colorbar
        cbar.ax.text(1, 1.02, 'KIAN-Net', transform=cbar.ax.transAxes, ha='center', va='bottom', fontsize=13, weight='bold')
        cbar.ax.text(1, -0.02, 'Artist 4.5', transform=cbar.ax.transAxes, ha='center', va='top', fontsize=13, weight='bold')
        
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
        
        # sns.set(style="dark", context=None, palette=None)
        
        r_time = from_array_to_datetime(self.X_EISCAT["r_time"])
        r_h = self.X_EISCAT["r_h"].flatten()
        n = len(self.selected_indices)
        
        fig, axes = plt.subplots(1, n, figsize=(5*n, 7), sharey=True)
        
        if n == 1:
            axes = [axes]  # Ensure axes is iterable if there's only one subplot
        
        for ax, idx in zip(axes, self.selected_indices):
            ax.plot(np.log10(self.X_EISCAT["r_param"][:, idx]), r_h, label='EISCAT', linestyle='-')
            ax.plot(self.X_HNN["r_param"][:, idx], r_h, label='DL Model', linestyle='-')
            ax.plot(np.log10(self.X_Artist["r_param"][:, idx]), r_h, label='Artist 4.5', linestyle='-')
            
            time_str = r_time[idx].strftime('%H:%M')
            ax.set_xlabel(r'$log_{10}(n_e)$ [$n/m^3$]', fontsize=13)
            ax.set_title(f'Time: {time_str}', fontsize=15)
            ax.grid(True)
            ax.legend()
        
        axes[0].set_ylabel('Altitude [km]', fontsize=13)
        
        date_str = r_time[idx].strftime('%Y-%m-%d')
        fig.suptitle(f'EISCAT vs HNN vs Artist for {n} Chosen Times\nDate: {date_str}', fontsize=20)
        plt.tight_layout()
        plt.show()
        
        
        
        
    def plot_selected_measurements_std(self):
        """
        Plot the selected measurements from all three radars in a 1xn grid of subplots.
        """
        if not self.selected_indices:
            print("No measurements selected. Please run select_measurements(n) or select_measurements_by_datetime(datetimes) first.")
            return
        
        # sns.set(style="dark", context=None, palette=None)
        
        r_time = from_array_to_datetime(self.X_EISCAT["r_time"])
        r_h = self.X_EISCAT["r_h"].flatten()
        n = len(self.selected_indices)
        
        fig, axes = plt.subplots(1, n, figsize=(5*n, 7), sharey=True)
        
        if n == 1:
            axes = [axes]  # Ensure axes is iterable if there's only one subplot
        
        for ax, idx in zip(axes, self.selected_indices):
            
            ne_eis = self.X_EISCAT["r_param"][:, idx]
            ne_eis_err = self.X_EISCAT["r_error"][:, idx]
            
            ne_kian = self.X_HNN["r_param"][:, idx]
            ne_art = self.X_Artist["r_param"][:, idx]
            
            
            ax.plot(np.log10(ne_eis), r_h, label='EISCAT', linestyle='-')
            # ax.plot(10**ne_kian, r_h, label='KIAN-Net', linestyle='-')
            # ax.plot(ne_art, r_h, label='Artist 4.5', linestyle='-')
            
            # print(ne_eis)
            # print(ne_eis_err)
            
            # err = ne_eis - ne_eis_err
            
            # print(ne_eis - ne_eis_err)
            # print(ne_eis + ne_eis_err)
            # ax.plot(self.X_HNN["r_param"][:, idx], r_h, label='DL Model', linestyle='-')
            
            low = np.log10(ne_eis - ne_eis_err)
            high = np.log10(ne_eis + ne_eis_err)
            
            ax.fill_betweenx(r_h, low, high, alpha=0.3)
            # ax.set_xscale("log")
            
            
            time_str = r_time[idx].strftime('%H:%M')
            ax.set_xlabel(r'$log_{10}(n_e)$ [$n/m^3$]', fontsize=13)
            ax.set_title(f'Time: {time_str}', fontsize=15)
            ax.grid(True)
            ax.legend()
        
        axes[0].set_ylabel('Altitude [km]', fontsize=13)
        
        date_str = r_time[idx].strftime('%Y-%m-%d')
        fig.suptitle(f'EISCAT vs HNN vs Artist for {n} Chosen Times\nDate: {date_str}', fontsize=20)
        plt.tight_layout()
        plt.show()
    
    
    
    
    
    # =============================================================================
    #                        Plotting Normalized Errors
    
        
    def error_function(self, X_eiscat, X):
        error = np.abs(X_eiscat - X)/X_eiscat
        return error
    
    
    def calculate_errors(self, idx):
        eiscat_param = np.log10(self.X_EISCAT["r_param"][:, idx])
        hnn_param = self.X_HNN["r_param"][:, idx]
        artist_param = np.log10(self.X_Artist["r_param"][:, idx])
        
        # Check if all values are NaN
        if np.all(np.isnan(artist_param)):
            valid_artist_mask = np.full(artist_param.shape, False)
            artist_param = np.zeros_like(artist_param)
        else:
            valid_artist_mask = ~np.isnan(artist_param)
            artist_param = np.nan_to_num(artist_param, nan=0)
        
        error_hnn = self.error_function(eiscat_param, hnn_param)
        error_artist = self.error_function(eiscat_param, artist_param)
        
        return error_hnn, error_artist, valid_artist_mask
    
    def plot_error_profiles(self):
        if not self.selected_indices:
            print("No measurements selected. Please run select_measurements(n) or select_measurements_by_datetime(datetimes) first.")
            return

        r_time = from_array_to_datetime(self.X_EISCAT["r_time"])
        r_h = self.X_EISCAT["r_h"].flatten()
        n = len(self.selected_indices)

        fig, axes = plt.subplots(1, n, figsize=(5 * n, 7), sharey=True)

        if n == 1:
            axes = [axes]  # Ensure axes is iterable if there's only one subplot

        for ax, idx in zip(axes, self.selected_indices):
            # Calculate errors
            error_hnn, error_artist, valid_artist_mask = self.calculate_errors(idx)

            # Plot errors
            ax.plot(error_hnn, r_h, label='Error: EISCAT vs DL Model', linestyle='-', color='C1')
            if np.any(valid_artist_mask):
                ax.plot(error_artist[valid_artist_mask], r_h[valid_artist_mask], label='Error: EISCAT vs Artist 4.5', linestyle='-', color='green')
                # Plot red line for NaN indices from the last valid value or first valid value
                nan_indices = np.where(~valid_artist_mask)[0]
                last_valid_index = np.max(np.where(valid_artist_mask)[0]) if np.any(valid_artist_mask) else None
                first_valid_index = np.min(np.where(valid_artist_mask)[0]) if np.any(valid_artist_mask) else None
                for nan_idx in nan_indices:
                    if nan_idx < first_valid_index or nan_idx > last_valid_index:
                        # Draw a line from the closest valid value to the NaN index
                        closest_valid_index = first_valid_index if nan_idx < first_valid_index else last_valid_index
                        ax.plot([error_artist[closest_valid_index], error_artist[nan_idx]], [r_h[closest_valid_index], r_h[nan_idx]], 'r--', linewidth=2, label='Missing values' if nan_idx == nan_indices[0] else "")
            else:
                # Plot all error values if all are NaN
                ax.plot(error_artist, r_h,  'r-', linewidth=2, label='No Artist Data')
            
            time_str = r_time[idx].strftime('%H:%M')
            ax.set_xlabel('Error', fontsize=13)
            ax.set_title(f'Time: {time_str}', fontsize=15)
            ax.grid(True)
            ax.legend()

            # Set x-axis limits based on valid error data
            if np.any(valid_artist_mask):
                ax.set_xlim(left=0, right=max(np.max(error_hnn), np.max(error_artist[valid_artist_mask])) * 1.1)
            else:
               ax.set_xlim(left=0, right=np.max(error_hnn) * 1.1)
                
        axes[0].set_ylabel('Altitude [km]', fontsize=13)

        date_str = r_time[self.selected_indices[0]].strftime('%Y-%m-%d')
        fig.suptitle(f'Normalized Error Profiles\nDate: {date_str}', fontsize=20)
        plt.tight_layout()
        plt.show()

    #     
    # =============================================================================

    def plot_ionogram_measurements_and_errors(self):
        """
        Plot the ionogram images, the selected measurements, and the error profiles for the selected dates.
        This method creates an nx3 grid of subplots, where n is the number of selected dates.
        """
        if not self.selected_indices:
            print("No measurements selected. Please run select_measurements(n) or select_measurements_by_datetime(datetimes) first.")
            return
        
        
        r_time = from_array_to_datetime(self.X_EISCAT["r_time"])
        n = len(self.selected_indices)
        with sns.axes_style("dark"):
            fig, axes = plt.subplots(n, 3, figsize=(12, 4*n))
            spec = GridSpec(n, 3, figure=fig, wspace=0.5)  # Increase wspace for first column only
    
        axes = []  # To store axes for easy access
            
        if n == 1:
            axes = [axes]  # Ensure axes is iterable if there's only one subplot
        
        for i, idx in enumerate(self.selected_indices):
            # First column: Ionogram
            ax1 = fig.add_subplot(spec[i, 0])  # First column
            axes.append(ax1)    
            
            # Plot ionogram image
            ionogram_img = self.X_Ionogram["r_param"][idx]
            ionogram_img = np.asarray(ionogram_img)  # Ensure it's a NumPy array
            ionogram_img = ionogram_img.astype(np.int64)  # Ensure it has a valid numeric type
            
            num=str(i+1)
            
            axes[i][0].imshow(ionogram_img)
            axes[i][0].set_title(f'     Ionogram - {r_time[idx].strftime("%H:%M")}', fontsize=21)
            axes[i][0].axis('off')
            
            # Get the position of the title
            title_pos = axes[i][0].title.get_position()  # (x, y) in axis coordinates
            title_x, title_y = title_pos
            
            # Add the red string next to the title
            axes[i][0].text(title_x - 0.4, title_y + 0.03, num, color='red', weight='bold', fontsize=21, transform=axes[i][0].transAxes, ha='right')
            
            
            # Plot selected measurements using existing method
            self.plot_single_measurement(axes[i][1], idx)
            
            # Plot error profiles using existing method
            self.plot_single_error(axes[i][2], idx)
            
            
        date_str = r_time[self.selected_indices[0]].strftime('%Y-%m-%d')
        fig.suptitle(f'Date: {date_str}', fontsize=25, y=1.01)
        fig.subplots_adjust(wspace=-0.8, hspace=0.2)
        plt.tight_layout()
        plt.show()
        
    def plot_single_measurement(self, ax, idx):
        """
        Plot a single selected measurement on a given axis.
        """
        r_time = from_array_to_datetime(self.X_EISCAT["r_time"])
        r_h = self.X_EISCAT["r_h"].flatten()
        
        ne_eis = self.X_EISCAT["r_param"][:, idx]
        ne_eis_err = self.X_EISCAT["r_error"][:, idx]
        
        
        ax.plot(np.log10(ne_eis), r_h, label='EISCAT', linestyle='-')
        # ax.fill_betweenx(r_h, np.log10(ne_eis - ne_eis_err), np.log10(ne_eis + ne_eis_err), alpha=0.3)
        
        ax.plot(self.X_HNN["r_param"][:, idx], r_h, label='KIAN-Net', linestyle='-')
        if self.X_Artist is not None:
            ax.plot(np.log10(self.X_Artist["r_param"][:, idx]), r_h, label='Artist 4.5', linestyle='-')
            
        time_str = r_time[idx].strftime('%H:%M')
        
        # print(idx)
        
        if idx == self.selected_indices[0]:
            ax.set_title('Measurements', fontsize=20)
        
        if idx == self.selected_indices[-1]:
            ax.set_xlabel(r'$log_{10}(n_e)$ $[n\,m^{-3}]$', fontsize=15)
        
        # ax.set_xlabel(r'$log_{10}(n_e)$ $[n\,m^{-3}]$', fontsize=13)
        ax.set_ylabel(r'Altitude $[km]$', fontsize=13)
        # ax.set_title(f'Measurements - {time_str}', fontsize=20)
        # ax.set_xlim(xmin=9, xmax=12)
        ax.grid(True)
        ax.legend()
        
        
        
    def plot_single_error(self, ax, idx):
        """
        Plot a single error profile on a given axis.
        """
        r_time = from_array_to_datetime(self.X_EISCAT["r_time"])
        r_h = self.X_EISCAT["r_h"].flatten()
        error_hnn, error_artist, valid_artist_mask = self.calculate_errors(idx)
        
        ax.tick_params(labelleft=False)
        ax.plot(error_hnn, r_h, label='KIAN-Net', linestyle='-', color='C1')
        if np.any(valid_artist_mask):
            ax.plot(error_artist[valid_artist_mask], r_h[valid_artist_mask], label='Artist 4.5', linestyle='-', color='green')
            # Plot red line for NaN indices from the last valid value or first valid value
            nan_indices = np.where(~valid_artist_mask)[0]
            last_valid_index = np.max(np.where(valid_artist_mask)[0]) if np.any(valid_artist_mask) else None
            first_valid_index = np.min(np.where(valid_artist_mask)[0]) if np.any(valid_artist_mask) else None
            for nan_idx in nan_indices:
                if nan_idx < first_valid_index or nan_idx > last_valid_index:
                    # Draw a line from the closest valid value to the NaN index
                    closest_valid_index = first_valid_index if nan_idx < first_valid_index else last_valid_index
                    ax.plot([error_artist[closest_valid_index], error_artist[nan_idx]], [r_h[closest_valid_index], r_h[nan_idx]], 'r--', linewidth=2, label='Missing values' if nan_idx == nan_indices[0] else "")
        else:
            # Plot all error values if all are NaN
            ax.plot(error_artist, r_h,  'r-', linewidth=2, label='No Artist Data')
        
        
        # Set x-axis limits based on valid error data
        if np.any(valid_artist_mask):
            ax.set_xlim(left=-0.01, right=max(np.max(error_hnn), np.max(error_artist[valid_artist_mask])) * 1.1)
        else:
           ax.set_xlim(left=-0.01, right=np.max(error_hnn) * 1.1)
        
        time_str = r_time[idx].strftime('%H:%M')
        if idx == self.selected_indices[0]:
            ax.set_title('Error Profiles', fontsize=20)
        
        if idx == self.selected_indices[-1]:
            ax.set_xlabel(r"Error $[n\,m^{-3}]$", fontsize=15)
        
        ax.grid(True)
        ax.legend()
        
        
    # =============================================================================
    #                        Interactive Plot
    #                             (Start)
    

    def plot_compare_all_interactive(self):
        """
        Method for creating interactive plot. Here the user has the option to
        click on any M measurement on the colorplots to view the corresponding
        ionogram, electron densities and the errors.
        """
        
        r_time = from_array_to_datetime(self.X_EISCAT["r_time"])
        art_time = from_array_to_datetime(self.X_Artist["r_time"])
        r_h = self.X_EISCAT["r_h"].flatten()

        # Logarithmic scaling for electron density
        ne_eis = np.log10(self.X_EISCAT["r_param"])
        ne_hnn = self.X_HNN["r_param"]
        ne_art = np.log10(self.X_Artist["r_param"])
        
        date_str = r_time[0].strftime('%Y-%m-%d')

        # Create the figure and layout
        fig = plt.figure(figsize=(18, 10))
        gs = GridSpec(2, 4, height_ratios=[1, 1], width_ratios=[1, 1, 1, 0.05], hspace=0.5, wspace=0.2)
        
        
        ax0 = fig.add_subplot(gs[0, 0])
        ax1 = fig.add_subplot(gs[0, 1], sharey=ax0)
        ax2 = fig.add_subplot(gs[0, 2], sharey=ax0)
        cax = fig.add_subplot(gs[0, 3])
        
        

        fig.suptitle(f'Date: {date_str}', fontsize=20, y=0.95)
        
        # Plot EISCAT data
        ne_EISCAT = ax0.pcolormesh(r_time, r_h, ne_eis, shading='auto', cmap='turbo', vmin=10, vmax=12)
        ax0.set_title('EISCAT UHF')
        ax0.set_xlabel('Time [hh:mm]')
        ax0.set_ylabel('Altitude [km]')
        ax0.xaxis.set_major_formatter(DateFormatter('%H:%M'))
        
        # Plot HNN data
        ax1.pcolormesh(r_time, r_h, ne_hnn, shading='auto', cmap='turbo', vmin=10, vmax=12)
        ax1.set_title('KIAN-Net')
        ax1.set_xlabel('Time [hh:mm]')
        ax1.xaxis.set_major_formatter(DateFormatter('%H:%M'))
        ax1.tick_params(labelleft=False)
        
        # Plot Artist data
        ax2.pcolormesh(art_time, r_h, ne_art, shading='auto', cmap='turbo', vmin=10, vmax=12)
        ax2.set_title('Artist 4.5')
        ax2.set_xlabel('Time [hh:mm]')
        ax2.xaxis.set_major_formatter(DateFormatter('%H:%M'))
        ax2.tick_params(labelleft=False)
        
        # Rotate x-axis labels
        for ax in [ax0, ax1, ax2]:
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='center')
        
        
        
        
        # Add colorbar
        cbar = fig.colorbar(ne_EISCAT, cax=cax, orientation='vertical')
        cbar.set_label(r'$log_{10}(n_e)$ [n/m$^3$]')
        
        
        # Detail plot axes
        with sns.axes_style("dark"):
            ax_iono = fig.add_subplot(gs[1, 0])
            ax_detail = fig.add_subplot(gs[1, 1])
            ax_error = fig.add_subplot(gs[1, 2])
            
        
        # Function to update the detailed plot
        def update_detail_plot(time_idx):
            ax_detail.clear()
            ax_detail.plot(ne_eis[:, time_idx], r_h, label="EISCAT", color='C0')
            ax_detail.plot(ne_hnn[:, time_idx], r_h, label="HNN", color='C1')
            ax_detail.plot(ne_art[:, time_idx], r_h, label="Artist", color='C2')
            ax_detail.set_title(f"{r_time[time_idx].strftime('%H:%M:%S')}")
            ax_detail.set_xlabel(r'$log_{10}(n_e)$ [n/m$^3$]')
            ax_detail.set_ylabel("Altitude [km]")
            ax_detail.legend()
            ax_detail.grid()
            fig.canvas.draw_idle()
        
        
        def update_ionogram(time_idx):
            ionogram_img = self.X_Ionogram["r_param"][time_idx]
            ionogram_img = np.asarray(ionogram_img)  # Ensure it's a NumPy array
            ionogram_img = ionogram_img.astype(np.int64)  # Ensure it has a valid numeric type
            
            ax_iono.clear()
            ax_iono.imshow(ionogram_img)
            ax_iono.set_title(f"{r_time[time_idx].strftime('%H:%M:%S')}")
            fig.canvas.draw_idle()
        
        def update_error(time_idx):
            
            error_hnn, error_artist, valid_artist_mask = self.calculate_errors(time_idx)
            
            ax_error.clear()
            ax_error.plot(error_hnn, r_h, label='Error: EISCAT vs KIAN-Net', linestyle='-', color='C1')
            if np.any(valid_artist_mask):
                ax_error.plot(error_artist[valid_artist_mask], r_h[valid_artist_mask], label='Error: EISCAT vs Artist 4.5', linestyle='-', color='green')
                # Plot red line for NaN indices from the last valid value or first valid value
                nan_indices = np.where(~valid_artist_mask)[0]
                last_valid_index = np.max(np.where(valid_artist_mask)[0]) if np.any(valid_artist_mask) else None
                first_valid_index = np.min(np.where(valid_artist_mask)[0]) if np.any(valid_artist_mask) else None
                for nan_idx in nan_indices:
                    if nan_idx < first_valid_index or nan_idx > last_valid_index:
                        # Draw a line from the closest valid value to the NaN index
                        closest_valid_index = first_valid_index if nan_idx < first_valid_index else last_valid_index
                        ax_error.plot([error_artist[closest_valid_index], error_artist[nan_idx]], [r_h[closest_valid_index], r_h[nan_idx]], 'r--', linewidth=2, label='Missing values' if nan_idx == nan_indices[0] else "")
            else:
                # Plot all error values if all are NaN
                ax_error.plot(error_artist, r_h,  'r-', linewidth=2, label='No Artist Data')
            
            
            # Set x-axis limits based on valid error data
            if np.any(valid_artist_mask):
                ax_error.set_xlim(left=0, right=max(np.max(error_hnn), np.max(error_artist[valid_artist_mask])) * 1.1)
            else:
               ax_error.set_xlim(left=0, right=np.max(error_hnn) * 1.1)
            
            ax_error.set_title(f"{r_time[time_idx].strftime('%H:%M:%S')}")
            ax_error.set_xlabel("Error")
            ax_error.set_ylabel("Altitude [km]")
            ax_error.legend()
            ax_error.grid()
            fig.canvas.draw_idle()
    
        
        # Handle click events
        def on_click(event):
            if event.inaxes in [ax0, ax1, ax2]:
                # Convert xdata to datetime for comparison
                click_time = mdates.num2date(event.xdata).replace(tzinfo=None)
                time_idx = np.argmin([abs((t - click_time).total_seconds()) for t in r_time])
                update_ionogram(time_idx)
                update_detail_plot(time_idx)
                update_error(time_idx)
                
                
                # Clear any existing vertical lines
                for ax in [ax0, ax1, ax2]:
                    for line in ax.lines:
                        line.remove()
                
                # Add a red staple line to all three color plots
                for ax in [ax0, ax1, ax2]:
                    ax.axvline(event.xdata, color='red', linestyle='--', linewidth=2)
                        
        # Add interactivity
        fig.canvas.mpl_connect("button_press_event", on_click)
        Cursor(ax0, useblit=True, color='red', linewidth=1)
        Cursor(ax1, useblit=True, color='red', linewidth=1)
        Cursor(ax2, useblit=True, color='red', linewidth=1)
        
        plt.show()
    
    
    
    
    #                             (end)
    #                        Interactive Plot
    # =============================================================================
        
    
    
    # =============================================================================
    #                        Peak Ne Plots
    #                             (Start)
    
    
    def plot_peaks_all_measurements(self):
        # sns.set(style="dark", context=None, palette=None)
        r_time = from_array_to_datetime(self.X_EISCAT["r_time"])
        
        r_h = self.X_EISCAT["r_h"].flatten()
        
        ne_eis = np.log10(self.X_EISCAT["r_param"])
        ne_hnn = self.X_HNN["r_param"]
        ne_art = np.log10(self.X_Artist["r_param"])
        
        
        
        eis_param_peak = np.log10(self.X_EISCAT["r_param_peak"])
        hnn_param_peak = self.X_HNN["r_param_peak"]
        art_param_peak = np.log10(self.X_Artist["r_param_peak"])
        
        eis_h_peak = self.X_EISCAT["r_h_peak"]
        hnn_h_peak = self.X_HNN["r_h_peak"]
        art_h_peak = self.X_Artist["r_h_peak"]
        
        
        date_str = r_time[0].strftime('%Y-%m-%d')

        
        for m in range(ne_eis.shape[1]):
            fig, ax = plt.subplots()
            fig.suptitle(f'Date: {date_str}', fontsize=20, y=1.0)
            
            ax.plot(ne_eis[:, m], r_h, color="C0")
            ax.plot(ne_hnn[:, m], r_h, color="C1")
            # ax.plot(ne_art[:, m], r_h, color="C2")
            ax.scatter(eis_param_peak[0, m], eis_h_peak[0, m], color="C0", marker="o")
            ax.scatter(eis_param_peak[1, m], eis_h_peak[1, m], color="C0", marker="o")
            ax.scatter(hnn_param_peak[0, m], hnn_h_peak[0, m], color="C1", marker="X")
            ax.scatter(hnn_param_peak[1, m], hnn_h_peak[1, m], color="C1", marker="X")
            # ax.scatter(art_param_peak[0, m], art_h_peak[0, m], color="C2", marker="s")
            # ax.scatter(art_param_peak[1, m], art_h_peak[1, m], color="C2", marker="s")
            ax.set_xlim(xmin=9.5, xmax=12.1)
            ax.set_ylim(ymin=88, ymax=402)
            # ax[1].plot(ne_hnn[:, m], r_h, color="C0")
            # ax[1].scatter(hnn_param_peak[0, m], hnn_h_peak[0, m], color="C1")
            # ax[1].scatter(hnn_param_peak[1, m], hnn_h_peak[1, m], color="red")
            ax.grid()
            plt.show()
            
    # def plot_compare_all_peak_heights(self):
    #     # sns.set(style="dark", context=None, palette=None)
    #     r_time = from_array_to_datetime(self.X_EISCAT["r_time"])
    #     # art_time = from_array_to_datetime(self.X_Artist["r_time"])
        
        
        
    #     eis_param_peak = np.log10(self.X_EISCAT["r_param_peak"])
    #     hnn_param_peak = self.X_HNN["r_param_peak"]
    #     art_param_peak = np.log10(self.X_Artist["r_param_peak"])
        
    #     eis_h_peak = self.X_EISCAT["r_h_peak"]
    #     hnn_h_peak = self.X_HNN["r_h_peak"]
    #     art_h_peak = self.X_Artist["r_h_peak"]
        
        
    #     date_str = r_time[0].strftime('%Y-%m-%d')
        
    #     # Create a grid layout
    #     fig = plt.figure(figsize=(14, 6))
    #     gs = GridSpec(1, 2, width_ratios=[1, 1], wspace=0.05)
        
    #     # Shared y-axis setup
    #     ax0 = fig.add_subplot(gs[0])
    #     ax1 = fig.add_subplot(gs[1], sharey=ax0)
        
        
    #     fig.suptitle(f'Date: {date_str}', fontsize=20, y=1.0)
        
    #     # Min, Max = 9.5, 12.1
        
        
    #     ax0.scatter(r_time, eis_h_peak[0,:], color="C0", label="EISCAT UHF")
    #     ax0.scatter(r_time, hnn_h_peak[0,:], color="C1", label="KIAN-Net")
    #     ax0.scatter(r_time, art_h_peak[0,:], color="C2", label="Artist 4.5")
    #     ax0.set_title('E-region Peaks', fontsize=17)
    #     ax0.set_xlabel('Time [hh:mm]', fontsize=15)
    #     ax0.set_ylabel(r'$log_{10}(ne)$ [$n\,m^{-3}$]', fontsize=15)
    #     # ax0.set_ylim(ymin=Min, ymax=Max)
    #     ax0.xaxis.set_major_formatter(DateFormatter('%H:%M'))
    #     ax0.legend(fontsize=12)
    #     ax0.grid(True)
        
        
    #     # Plotting E-peaks
    #     ax1.scatter(r_time, eis_h_peak[1,:], color="C0", label="EISCAT UHF")
    #     ax1.scatter(r_time, hnn_h_peak[1,:], color="C1", label="KIAN-Nnet")
    #     ax1.scatter(r_time, art_h_peak[1,:], color="C2", label="Artist 4.5")
    #     ax1.set_title('F-region Peaks', fontsize=17)
    #     ax1.set_xlabel('Time', fontsize=15)
    #     # ax1.set_ylim(ymin=Min, ymax=Max)
    #     ax1.tick_params(labelleft=False)
    #     ax1.legend(fontsize=12)
    #     ax1.grid(True)
    #     ax1.xaxis.set_major_formatter(DateFormatter('%H:%M'))
        
    #     for ax in [ax0, ax1]:
    #         plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='center')
        
        
    #     plt.show()

    
    def plot_compare_all_peak_heights(self):
        # sns.set(style="dark", context=None, palette=None)
        # r_time = from_array_to_datetime(self.X_EISCAT["r_time"])
        # art_time = from_array_to_datetime(self.X_Artist["r_time"])
        r_time = from_array_to_datetime(self.X_EISCAT["r_time"])
        r_h = self.X_EISCAT["r_h"].flatten()
        
        ne_eis = np.log10(self.X_EISCAT["r_param"])
        # ne_kian = self.X_HNN["r_param"]
        
        
        # eis_param_peak = np.log10(self.X_EISCAT["r_param_peak"])
        # hnn_param_peak = self.X_HNN["r_param_peak"]
        # art_param_peak = np.log10(self.X_Artist["r_param_peak"])
        
        eis_h_peak = self.X_EISCAT["r_h_peak"]
        hnn_h_peak = self.X_HNN["r_h_peak"]
        art_h_peak = self.X_Artist["r_h_peak"]
        
        
        date_str = r_time[0].strftime('%Y-%m-%d')
        
        # Create a grid layout
        fig = plt.figure(figsize=(14, 7))
        gs = GridSpec(1, 2, width_ratios=[1, 1], wspace=0.05)
        
        # Shared y-axis setup
        ax0 = fig.add_subplot(gs[0])
        ax1 = fig.add_subplot(gs[1], sharey=ax0)
        
        
        fig.suptitle(f'Date: {date_str}', fontsize=20, y=1.0)
        
        # Min, Max = 9.5, 12.1
        
        ax0.pcolormesh(r_time, r_h, ne_eis, shading='auto', cmap='turbo', vmin=10, vmax=12)
        ax0.set_title('EISCAT UHF', fontsize=17)
        ax0.set_xlabel('Time [hh:mm]', fontsize=13)
        ax0.set_ylabel('Altitude [km]', fontsize=15)
        ax0.xaxis.set_major_formatter(DateFormatter('%H:%M'))
        # ax0.legend(fontsize=12)
        # ax0.grid(True)
        
        # Plotting E-peaks
        ax1.pcolormesh(r_time, r_h, ne_eis, shading='auto', cmap='grey', vmin=10, vmax=12, zorder=-1)
        ax1.plot(r_time, eis_h_peak[0,:], color="C0", label="EISCAT UHF", linestyle='-', linewidth=10, zorder=3)
        ax1.plot(r_time, hnn_h_peak[0,:], color="C1", label="KIAN-Net", linestyle='-', linewidth=6, zorder=3)
        ax1.plot(r_time, art_h_peak[0,:], color="C2", label="Artist 4.5", linestyle='-', linewidth=6, zorder=3)
        
        ax1.scatter(r_time, eis_h_peak[0,:], color="C0", s=100, zorder=2)
        ax1.scatter(r_time, hnn_h_peak[0,:], color="C1", s=60, zorder=2)
        ax1.scatter(r_time, art_h_peak[0,:], color="C2", s=60, zorder=2)
        
        ax1.plot(r_time, eis_h_peak[1,:], color="C0", linestyle='-', linewidth=10, zorder=3)
        ax1.plot(r_time, hnn_h_peak[1,:], color="C1", linestyle='-', linewidth=6, zorder=3)
        ax1.plot(r_time, art_h_peak[1,:], color="C2", linestyle='-', linewidth=6, zorder=3)
        
        ax1.scatter(r_time, eis_h_peak[1,:], color="C0", s=100, zorder=2)
        ax1.scatter(r_time, hnn_h_peak[1,:], color="C1", s=60, zorder=2)
        ax1.scatter(r_time, art_h_peak[1,:], color="C2", s=60, zorder=2)
        
        ax1.set_title('Peak Altitudes', fontsize=17)
        # ax1.set_xlabel('Peaks', fontsize=15)
        # # ax1.set_ylim(ymin=Min, ymax=Max)
        ax1.set_xlabel('Time [hh:mm]', fontsize=13)
        ax1.tick_params(labelleft=False)
        # ax1.legend(fontsize=12)
        ax1.xaxis.set_major_formatter(DateFormatter('%H:%M'))
        ax1.legend(fontsize=12)
        for ax in [ax0, ax1]:
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='center')
        
        
        plt.show()
    
    
    # # Function to compute R^2 score
    # def compute_r2_score(self, true_vals, pred_vals):
    #     return r2_score(true_vals, pred_vals)
    
    def compute_r2_score(self, y1, y2):
        """Calculate R2 score between two lines."""
        ss_tot = np.sum((y1 - np.mean(y1)) ** 2)
        ss_res = np.sum((y1 - y2) ** 2)
        return 1 - (ss_res / ss_tot)
    
    
    
    def plot_compare_all_peak_altitudes(self):
        sns.set(style="dark", context=None, palette=None)
    
        # Convert time data to datetime objects
        r_time = from_array_to_datetime(self.X_EISCAT["r_time"])
        art_time = from_array_to_datetime(self.X_Artist["r_time"])
        
        
        eis_h_peak = self.X_EISCAT["r_h_peak"]
        kian_h_peak = self.X_HNN["r_h_peak"]
        art_h_peak = self.X_Artist["r_h_peak"]
    
        # Format date for plot title
        date_str = r_time[0].strftime('%Y-%m-%d')
    
        # Create a grid layout
        fig = plt.figure(figsize=(14, 7))
        gs = GridSpec(1, 2, width_ratios=[1, 1], wspace=0.1)
    
        # Shared y-axis setup
        ax_e = fig.add_subplot(gs[0])
        ax_f = fig.add_subplot(gs[1])
    
        fig.suptitle(f'Date: {date_str}', fontsize=20, y=1.0)
        
        def plot_region(ax, region, Min, Max, title):
            # Extract valid data for the current region
            valid_kian = ~np.isnan(kian_h_peak[region, :]) & ~np.isnan(eis_h_peak[region, :])
            valid_artist = ~np.isnan(art_h_peak[region, :]) & ~np.isnan(eis_h_peak[region, :])
            
            # Determine min and max values for regression calculation based on valid data
            kian_x = kian_h_peak[region, valid_kian]
            kian_y = eis_h_peak[region, valid_kian]
            art_x = art_h_peak[region, valid_artist]
            art_y = eis_h_peak[region, valid_artist]
            
            
            
            # Plot diagonal reference line
            ax.plot([Min, Max], [Min, Max], color="C0", label="EISCAT vs EISCAT", linewidth=3, zorder=1)
            
            # Scatter plots for KIAN-Net and Artist 4.5
            ax.scatter(kian_x, kian_y, s=50, color="C1", label="KIAN-Net vs EISCAT", edgecolors="black")
            
            # Regression for KIAN-Net
            if valid_kian.any():
                slope_kian, intercept_kian, _, _, _ = linregress(kian_x, kian_y)
                if not np.isnan(slope_kian) and not np.isnan(intercept_kian):
                    
                    kian_reg_line = slope_kian * np.array([Min, Max]) + intercept_kian
                    diagonal_line = np.array([Min, Max])  # Diagonal reference
                    kian_r2_line = self.compute_r2_score(diagonal_line, kian_reg_line)
                    # print(f"KIAN-Net R^2 with diagonal ({title}): {kian_r2_line}")
        
                    ax.plot([Min, Max], kian_reg_line, color="C1", linestyle="--", linewidth=2,
                            label="Regression line", zorder=2)
                    
                    ax.text(Max-200, Min+10, s=rf"$\mathbf{{R^2={kian_r2_line:.3f}}}$", color="C1", fontsize=15)
                    
                    # # Calculate R^2 for regression line
                    # y_pred = slope * kian_x + intercept
                    # y_mean = np.mean(kian_y)
                    # ss_tot = np.sum((kian_y - y_mean) ** 2)
                    # ss_res = np.sum((kian_y - y_pred) ** 2)
                    # kian_r2 = 1 - (ss_res / ss_tot)
                    # print(f"KIAN-Net R^2 ({title}): {kian_r2}")
                    
                    # ax.plot([Min, Max], [slope * Min + intercept, slope * Max + intercept], 
                    #         color="C1", linestyle="--", linewidth=2, label=f" Reg $R^2=${kian_r2:.3}", zorder=2)
                    
                    
            ax.scatter(art_x, art_y, s=50, color="C2", label="Artist 4.5 vs EISCAT", edgecolors="black")
            # Regression for Artist
            if valid_artist.any():
                slope_artist, intercept_artist, _, _, _ = linregress(art_x, art_y)
                if not np.isnan(slope_artist) and not np.isnan(intercept_artist):
                    
                    artist_reg_line = slope_artist * np.array([Min, Max]) + intercept_artist
                    diagonal_line = np.array([Min, Max])
                    artist_r2_line = self.compute_r2_score(diagonal_line, artist_reg_line)
                    # print(f"Artist R^2 with diagonal ({title}): {artist_r2_line}")
                    
                    ax.plot([Min, Max], artist_reg_line, color="C2", linestyle="--", linewidth=2,
                            label="Regression line", zorder=2)
                    
                    ax.text(Max-1, Min+0.1, s=rf"$\mathbf{{R^{2}={artist_r2_line:.3f}}}$", color="C2", fontsize=15)
                    
                    # # Calculate R^2 for regression line
                    # y_pred = slope * art_x + intercept
                    # y_mean = np.mean(art_y)
                    # ss_tot = np.sum((art_y - y_mean) ** 2)
                    # ss_res = np.sum((art_y - y_pred) ** 2)
                    # artist_r2 = 1 - (ss_res / ss_tot)
                    # print(f"Artist R^2 ({title}): {artist_r2}")
                    
                    # ax.plot([Min, Max], [slope * Min + intercept, slope * Max + intercept], 
                    #         color="C2", linestyle="--", linewidth=2, label=f" Reg $R^2=${artist_r2:.3}", zorder=2)
                    
            # Configure plot aesthetics
            ax.set_title(f'{title} Peaks', fontsize=17)
            ax.set_xlabel('MODEL  $log_{10}\,(ne)$', fontsize=15)
            ax.set_xlim(xmin=Min, xmax=Max)
            ax.set_ylim(ymin=Min, ymax=Max)
            ax.legend(fontsize=11)
            ax.grid(True)
    
        # Plot E-region and F-region
        # plot_region(ax_e, region=0, Min=90, Max=400, title='E-region')
        plot_region(ax_f, region=1, Min=90, Max=400, title='F-region')
    
        # Adjust F-region plot labels
        ax_f.set_xlabel('MODEL  $log_{10}\,(ne)$', fontsize=15)
    
        plt.show()
    
    
    
    
    
    
    def plot_compare_all_peak_densities(self):
        sns.set(style="dark", context=None, palette=None)
    
        # Convert time data to datetime objects
        r_time = from_array_to_datetime(self.X_EISCAT["r_time"])
        art_time = from_array_to_datetime(self.X_Artist["r_time"])
    
        # Extract and process peak parameters
        eis_param_peak = np.log10(self.X_EISCAT["r_param_peak"])
        kian_param_peak = self.X_HNN["r_param_peak"]
        art_param_peak = np.log10(self.X_Artist["r_param_peak"])
    
    
        # Format date for plot title
        date_str = r_time[0].strftime('%Y-%m-%d')
    
        # Create a grid layout
        fig = plt.figure(figsize=(14, 7))
        gs = GridSpec(1, 2, width_ratios=[1, 1], wspace=0.1)
    
        # Shared y-axis setup
        ax_e = fig.add_subplot(gs[0])
        ax_f = fig.add_subplot(gs[1])
    
        fig.suptitle(f'Date: {date_str}', fontsize=20, y=1.0)
        
        def plot_region(ax, region, Min, Max, title):
            # Extract valid data for the current region
            valid_kian = ~np.isnan(kian_param_peak[region, :]) & ~np.isnan(eis_param_peak[region, :])
            valid_artist = ~np.isnan(art_param_peak[region, :]) & ~np.isnan(eis_param_peak[region, :])
            
            # Determine min and max values for regression calculation based on valid data
            kian_x = kian_param_peak[region, valid_kian]
            kian_y = eis_param_peak[region, valid_kian]
            art_x = art_param_peak[region, valid_artist]
            art_y = eis_param_peak[region, valid_artist]
            
            # data_min = min(np.min(kian_x), np.min(art_x), np.min(kian_y), np.min(art_y))
            # data_max = max(np.max(kian_x), np.max(art_x), np.max(kian_y), np.max(art_y))
            
            # Plot diagonal reference line
            ax.plot([Min, Max], [Min, Max], color="C0", label="EISCAT vs EISCAT", linewidth=3, zorder=1)
            
            # Scatter plots for KIAN-Net and Artist 4.5
            ax.scatter(kian_x, kian_y, s=50, color="C1", label="KIAN-Net vs EISCAT", edgecolors="black")
            
            # Regression for KIAN-Net
            if valid_kian.any():
                slope_kian, intercept_kian, _, _, _ = linregress(kian_x, kian_y)
                if not np.isnan(slope_kian) and not np.isnan(intercept_kian):
                    
                    kian_reg_line = slope_kian * np.array([Min, Max]) + intercept_kian
                    diagonal_line = np.array([Min, Max])  # Diagonal reference
                    kian_r2_line = self.compute_r2_score(diagonal_line, kian_reg_line)
                    # print(f"KIAN-Net R^2 with diagonal ({title}): {kian_r2_line}")
        
                    ax.plot([Min, Max], kian_reg_line, color="C1", linestyle="--", linewidth=2,
                            label="Regression line", zorder=2)
                    
                    ax.text(Max-1, Min+0.25, s=rf"$\mathbf{{R^2={kian_r2_line:.3f}}}$", color="C1", fontsize=15)
                    
                    # # Calculate R^2 for regression line
                    # y_pred = slope * kian_x + intercept
                    # y_mean = np.mean(kian_y)
                    # ss_tot = np.sum((kian_y - y_mean) ** 2)
                    # ss_res = np.sum((kian_y - y_pred) ** 2)
                    # kian_r2 = 1 - (ss_res / ss_tot)
                    # print(f"KIAN-Net R^2 ({title}): {kian_r2}")
                    
                    # ax.plot([Min, Max], [slope * Min + intercept, slope * Max + intercept], 
                    #         color="C1", linestyle="--", linewidth=2, label=f" Reg $R^2=${kian_r2:.3}", zorder=2)
                    
                    
            ax.scatter(art_x, art_y, s=50, color="C2", label="Artist 4.5 vs EISCAT", edgecolors="black")
            # Regression for Artist
            if valid_artist.any():
                slope_artist, intercept_artist, _, _, _ = linregress(art_x, art_y)
                if not np.isnan(slope_artist) and not np.isnan(intercept_artist):
                    
                    artist_reg_line = slope_artist * np.array([Min, Max]) + intercept_artist
                    diagonal_line = np.array([Min, Max])
                    artist_r2_line = self.compute_r2_score(diagonal_line, artist_reg_line)
                    # print(f"Artist R^2 with diagonal ({title}): {artist_r2_line}")
                    
                    ax.plot([Min, Max], artist_reg_line, color="C2", linestyle="--", linewidth=2,
                            label="Regression line", zorder=2)
                    
                    ax.text(Max-1, Min+0.1, s=rf"$\mathbf{{R^{2}={artist_r2_line:.3f}}}$", color="C2", fontsize=15)
                    
                    # # Calculate R^2 for regression line
                    # y_pred = slope * art_x + intercept
                    # y_mean = np.mean(art_y)
                    # ss_tot = np.sum((art_y - y_mean) ** 2)
                    # ss_res = np.sum((art_y - y_pred) ** 2)
                    # artist_r2 = 1 - (ss_res / ss_tot)
                    # print(f"Artist R^2 ({title}): {artist_r2}")
                    
                    # ax.plot([Min, Max], [slope * Min + intercept, slope * Max + intercept], 
                    #         color="C2", linestyle="--", linewidth=2, label=f" Reg $R^2=${artist_r2:.3}", zorder=2)
                    
            # Configure plot aesthetics
            ax.set_title(f'{title} Peaks', fontsize=17)
            # ax.set_xlabel('MODEL  $log_{10}\,(ne)$', fontsize=15)
            ax.set_xlim(xmin=Min, xmax=Max)
            ax.set_ylim(ymin=Min, ymax=Max)
            ax.legend(fontsize=11)
            ax.grid(True)
    
        # Plot E-region and F-region
        plot_region(ax_e, region=0, Min=9, Max=12, title='E-region')
        plot_region(ax_f, region=1, Min=9, Max=12, title='F-region')
    
        # Adjust region plot labels
        ax_e.set_xlabel(r'Prediction  $log_{10}\,(n_e)$', fontsize=15)
        ax_e.set_ylabel(r'EISCAT UHF  $log_{10}\,(n_e)$', fontsize=15)
        ax_f.set_xlabel(r'Prediction  $log_{10}\,(n_e)$', fontsize=15)
        
        plt.show()
        
    # def plot_compare_all_peak_densities(self):
    #     sns.set(style="dark", context=None, palette=None)
    
    #     # Convert time data to datetime objects
    #     r_time = from_array_to_datetime(self.X_EISCAT["r_time"])
    #     art_time = from_array_to_datetime(self.X_Artist["r_time"])
    
    #     # Extract and process peak parameters
    #     eis_param_peak = np.log10(self.X_EISCAT["r_param_peak"])
    #     kian_param_peak = self.X_HNN["r_param_peak"]
    #     art_param_peak = np.log10(self.X_Artist["r_param_peak"])
    
    #     # Format date for plot title
    #     date_str = r_time[0].strftime('%Y-%m-%d')
    
    #     # Create a grid layout
    #     fig = plt.figure(figsize=(14, 7))
    #     gs = GridSpec(1, 2, width_ratios=[1, 1], wspace=0.1)
    
    #     # Shared y-axis setup
    #     ax_e = fig.add_subplot(gs[0])
    #     ax_f = fig.add_subplot(gs[1])
    
    #     fig.suptitle(f'Date: {date_str}', fontsize=20, y=1.0)
    
    #     def plot_region(ax, region, Min, Max, title):
    #         # Extract valid data for the current region
    #         valid_kian = ~np.isnan(kian_param_peak[region, :]) & ~np.isnan(eis_param_peak[region, :])
    #         valid_artist = ~np.isnan(art_param_peak[region, :]) & ~np.isnan(eis_param_peak[region, :])
    
    #         # Get valid data points for KIAN and Artist
    #         kian_x = kian_param_peak[region, valid_kian]
    #         kian_y = eis_param_peak[region, valid_kian]
    #         art_x = art_param_peak[region, valid_artist]
    #         art_y = eis_param_peak[region, valid_artist]
    
    #         # Compute R^2 scores
    #         kian_r2 = self.compute_r2_score(kian_y, kian_x)
    #         artist_r2 = self.compute_r2_score(art_y, art_x)
    
    #         # Plot diagonal reference line
    #         ax.plot([Min, Max], [Min, Max], color="C0", label="EISCAT vs EISCAT", linewidth=3, zorder=1)
    
    #         # Scatter plots for KIAN-Net and Artist 4.5
    #         ax.scatter(kian_x, kian_y, s=50, color="C1", label=f"KIAN-Net $R^2={kian_r2:.3f}$", edgecolors="black")
    #         ax.scatter(art_x, art_y, s=50, color="C2", label=f"Artist 4.5 $R^2={artist_r2:.3f}$", edgecolors="black")
    
    #         # Regression for KIAN-Net
    #         if valid_kian.any():
    #             slope_kian, intercept_kian, _, _, _ = linregress(kian_x, kian_y)
    #             if not np.isnan(slope_kian) and not np.isnan(intercept_kian):
    #                 kian_reg_line = slope_kian * np.array([Min, Max]) + intercept_kian
    #                 ax.plot([Min, Max], kian_reg_line, color="C1", linestyle="--", linewidth=2, label="KIAN Regression")
    
    #         # Regression for Artist
    #         if valid_artist.any():
    #             slope_artist, intercept_artist, _, _, _ = linregress(art_x, art_y)
    #             if not np.isnan(slope_artist) and not np.isnan(intercept_artist):
    #                 artist_reg_line = slope_artist * np.array([Min, Max]) + intercept_artist
    #                 ax.plot([Min, Max], artist_reg_line, color="C2", linestyle="--", linewidth=2, label="Artist Regression")
    
    #         # Configure plot aesthetics
    #         ax.set_title(f'{title} Peaks', fontsize=17)
    #         ax.set_xlim(xmin=Min, xmax=Max)
    #         ax.set_ylim(ymin=Min, ymax=Max)
    #         ax.legend(fontsize=11)
    #         ax.grid(True)
    
    #     # Plot E-region and F-region
    #     plot_region(ax_e, region=0, Min=9, Max=12, title='E-region')
    #     plot_region(ax_f, region=1, Min=9, Max=12, title='F-region')
    
    #     # Adjust region plot labels
    #     ax_e.set_xlabel(r'Prediction  $log_{10}\,(n_e)$', fontsize=15)
    #     ax_e.set_ylabel(r'EISCAT UHF  $log_{10}\,(n_e)$', fontsize=15)
    #     ax_f.set_xlabel(r'Prediction  $log_{10}\,(n_e)$', fontsize=15)
    
    #     plt.show()


    #                             (end)
    #                         Peak Ne plots
    # =============================================================================










def plot_compare(ne_true, ne_pred, r_time):
    
    
    # Eiscat altitudes
    r_h = np.array([[91.5687711 ],[ 94.57444598],[ 97.57964223],[100.57010953],
           [103.57141624],[106.57728701],[110.08393175],[114.60422289],
           [120.1185208 ],[126.61221111],[134.1346149 ],[142.53945817],
           [152.05174717],[162.57986185],[174.09833378],[186.65837945],
           [200.15192581],[214.62769852],[230.12198695],[246.64398082],
           [264.11728204],[282.62750673],[302.15668686],[322.70723831],
           [344.19596481],[366.64409299],[390.113117]])
    

    date_str = r_time[0].strftime('%Y-%m-%d')


    # Creating the plots
    fig, ax = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
    fig.suptitle(f'Date: {date_str}', fontsize=20)
    fig.tight_layout()
    
    
    # Plotting original data
    ne_EISCAT = ax[0].pcolormesh(r_time, r_h.flatten(), ne_true.T, shading='auto', cmap='turbo', vmin=10, vmax=12)
    ax[0].set_title('EISCAT UHF', fontsize=17)
    ax[0].set_xlabel('Time [hours]', fontsize=13)
    ax[0].set_ylabel('Altitude [km]', fontsize=15)
    ax[0].xaxis.set_major_formatter(DateFormatter('%H:%M'))
    
    # Plotting predicted data
    ne_HNN = ax[1].pcolormesh(r_time, r_h.flatten(), ne_pred.T, shading='auto', cmap='turbo', vmin=10, vmax=12)
    ax[1].set_title('HNN', fontsize=17)
    ax[1].set_xlabel('Time [hours]', fontsize=13)
    ax[1].xaxis.set_major_formatter(DateFormatter('%H:%M'))
    
    
    # Add colorbar for the predicted data
    cbar = fig.colorbar(ne_HNN, ax=ax[1], orientation='vertical', fraction=0.048, pad=0.04)
    cbar.set_label(r'$log_{10}(n_e)$ [g/cm$^3$]', fontsize=17, labelpad=15)
    
    fig.autofmt_xdate()
    plt.show()




def plot_compare_r2(ne_true, ne_pred, r2_scores, r_time):
    
    
    # Eiscat altitudes
    r_h = np.array([[91.5687711 ],[ 94.57444598],[ 97.57964223],[100.57010953],
           [103.57141624],[106.57728701],[110.08393175],[114.60422289],
           [120.1185208 ],[126.61221111],[134.1346149 ],[142.53945817],
           [152.05174717],[162.57986185],[174.09833378],[186.65837945],
           [200.15192581],[214.62769852],[230.12198695],[246.64398082],
           [264.11728204],[282.62750673],[302.15668686],[322.70723831],
           [344.19596481],[366.64409299],[390.113117]])
    

    date_str = r_time[0].strftime('%Y-%m-%d')

    
    # Creating the plots
    fig, ax = plt.subplots(1, 3, figsize=(16, 6), gridspec_kw={'width_ratios': [1, 0.2, 1]}, sharey=True)
    fig.suptitle(f'Date: {date_str}', fontsize=20, y=1.02)
    fig.tight_layout()

    # Plotting original data
    ne_EISCAT = ax[0].pcolormesh(r_time, r_h, ne_true.T, shading='auto', cmap='turbo', vmin=10, vmax=12)
    ax[0].set_title('EISCAT UHF', fontsize=17)
    ax[0].set_xlabel('Time [hours]', fontsize=13)
    ax[0].set_ylabel('Altitude [km]', fontsize=15)
    ax[0].xaxis.set_major_formatter(DateFormatter('%H:%M'))

    # Plotting R2-scores as a line plot
    ax[1].plot(r2_scores, r_h, color='C0', label=r'$R^2$')
    ax[1].set_title(r'$R^2$ Scores', fontsize=17)
    ax[1].set_xlabel(r'$R^2$', fontsize=13)
    ax[1].grid()
    ax[1].legend()
    ax[1].set_xlim(xmin=-0.1, xmax=1.1)
    
    # Plotting predicted data
    ne_HNN = ax[2].pcolormesh(r_time, r_h, ne_pred.T, shading='auto', cmap='turbo', vmin=10, vmax=12)
    ax[2].set_title('HNN', fontsize=17)
    ax[2].set_xlabel('Time [hours]', fontsize=13)
    ax[2].xaxis.set_major_formatter(DateFormatter('%H:%M'))

    # Add colorbar for the predicted data
    cbar = fig.colorbar(ne_HNN, ax=ax[2], orientation='vertical', fraction=0.048, pad=0.04)
    cbar.set_label(r'$log_{10}(n_e)$ [g/cm$^3$]', fontsize=17, labelpad=15)

    fig.autofmt_xdate()
    plt.show()









def plot_compare_all(X_EISCAT, X_HNN, X_Artist):
    r_time = X_EISCAT["r_time"]
    art_time = X_Artist["r_time"]
    r_h = X_EISCAT["r_h"].flatten()
    
    ne_eis = np.log10(X_EISCAT["r_param"])
    ne_hnn = X_HNN["r_param"]
    ne_art = np.log10(X_Artist["r_param"])
    
    r_time = from_array_to_datetime(r_time)
    art_time = from_array_to_datetime(art_time)
    
    date_str = r_time[0].strftime('%Y-%m-%d')

    # Create a grid layout
    fig = plt.figure(figsize=(16, 8))
    gs = GridSpec(1, 4, width_ratios=[1, 1, 1, 0.05], wspace=0.1)

    # Shared y-axis setup
    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1], sharey=ax0)
    ax2 = fig.add_subplot(gs[2], sharey=ax0)
    cax = fig.add_subplot(gs[3])

    fig.suptitle(f'Date: {date_str}', fontsize=20, y=1.0)

    # x_limits = [r_time[0], r_time[-1]]

    # Plotting EISCAT
    ne_EISCAT = ax0.pcolormesh(r_time, r_h.flatten(), ne_eis, shading='auto', cmap='turbo', vmin=10, vmax=12)
    ax0.set_title('EISCAT UHF', fontsize=17)
    ax0.set_xlabel('Time [hours]', fontsize=13)
    ax0.set_ylabel('Altitude [km]', fontsize=15)
    ax0.xaxis.set_major_formatter(DateFormatter('%H:%M'))

    # Plotting DL model
    ax1.pcolormesh(r_time, r_h.flatten(), ne_hnn, shading='auto', cmap='turbo', vmin=10, vmax=12)
    ax1.set_title('DL model', fontsize=17)
    ax1.set_xlabel('Time [hours]', fontsize=13)
    ax1.xaxis.set_major_formatter(DateFormatter('%H:%M'))
    ax1.tick_params(labelleft=False)  # Suppress y-axis labels for this subplot

    # Plotting Artist 4.5
    ax2.pcolormesh(art_time, r_h.flatten(), ne_art, shading='auto', cmap='turbo', vmin=10, vmax=12)
    ax2.set_title('Artist 4.5', fontsize=17)
    ax2.set_xlabel('Time [hours]', fontsize=13)
    ax2.xaxis.set_major_formatter(DateFormatter('%H:%M'))
    # ax2.set_xlim(x_limits)
    ax2.tick_params(labelleft=False)  # Suppress y-axis labels for this subplot

    # Rotate x-axis labels
    for ax in [ax0, ax1, ax2]:
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Add colorbar
    cbar = fig.colorbar(ne_EISCAT, cax=cax, orientation='vertical')
    cbar.set_label(r'$log_{10}(n_e)$ [n/cm$^3$]', fontsize=17)
    
    plt.show()








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






