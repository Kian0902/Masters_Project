# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 14:31:51 2024

@author: Kian Sartipzadeh
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
from datetime import datetime
from matplotlib.gridspec import GridSpec
from eval_utils import from_array_to_datetime
import random

import seaborn as sns



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









def plot_compare_all(support, X_EISCAT, X_HNN, X_Artist):
    r_time = X_EISCAT["r_time"]
    art_time = X_Artist["r_time"]
    r_h = support["r_h"].flatten()
    
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




class RadarPlotter:
    def __init__(self, support, X_EISCAT, X_HNN, X_Artist, X_Ionogram):
        self.support = support
        self.X_EISCAT = X_EISCAT
        self.X_HNN = X_HNN
        self.X_Artist = X_Artist
        self.X_Ionogram = X_Ionogram
        self.selected_indices = []

    def plot_compare_all(self):
        r_time = from_array_to_datetime(self.X_EISCAT["r_time"])
        art_time = from_array_to_datetime(self.X_Artist["r_time"])
        r_h = self.support["r_h"].flatten()
        
        ne_eis = np.log10(self.X_EISCAT["r_param"])
        ne_hnn = self.X_HNN["r_param"]
        ne_art = np.log10(self.X_Artist["r_param"])
        
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
        
        # Plotting EISCAT
        ne_EISCAT = ax0.pcolormesh(r_time, r_h, ne_eis, shading='auto', cmap='turbo', vmin=10, vmax=12)
        ax0.set_title('EISCAT UHF', fontsize=17)
        ax0.set_xlabel('Time [hours]', fontsize=13)
        ax0.set_ylabel('Altitude [km]', fontsize=15)
        ax0.xaxis.set_major_formatter(DateFormatter('%H:%M'))
        
        # Plotting DL model
        ax1.pcolormesh(r_time, r_h, ne_hnn, shading='auto', cmap='turbo', vmin=10, vmax=12)
        ax1.set_title('DL model', fontsize=17)
        ax1.set_xlabel('Time [hours]', fontsize=13)
        ax1.xaxis.set_major_formatter(DateFormatter('%H:%M'))
        ax1.tick_params(labelleft=False)
        
        # Plotting Artist 4.5
        ax2.pcolormesh(art_time, r_h, ne_art, shading='auto', cmap='turbo', vmin=10, vmax=12)
        ax2.set_title('Artist 4.5', fontsize=17)
        ax2.set_xlabel('Time [hours]', fontsize=13)
        ax2.xaxis.set_major_formatter(DateFormatter('%H:%M'))
        ax2.tick_params(labelleft=False)
        
        # Rotate x-axis labels
        for ax in [ax0, ax1, ax2]:
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Highlight selected measurements
        for idx in self.selected_indices:
            for ax in [ax0, ax1, ax2]:
                ax.axvline(r_time[idx], color='red', linestyle='--', linewidth=3)
        
        # Add colorbar
        cbar = fig.colorbar(ne_EISCAT, cax=cax, orientation='vertical')
        cbar.set_label(r'$log_{10}(n_e)$ [n/cm$^3$]', fontsize=17)
        
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
        
        sns.set(style="dark", context=None, palette=None)
        
        r_time = from_array_to_datetime(self.X_EISCAT["r_time"])
        r_h = self.support["r_h"].flatten()
        n = len(self.selected_indices)
        
        fig, axes = plt.subplots(1, n, figsize=(5*n, 7), sharey=True)
        
        if n == 1:
            axes = [axes]  # Ensure axes is iterable if there's only one subplot
        
        for ax, idx in zip(axes, self.selected_indices):
            ax.plot(np.log10(self.X_EISCAT["r_param"][:, idx]), r_h, label='EISCAT', linestyle='-')
            ax.plot(self.X_HNN["r_param"][:, idx], r_h, label='DL Model', linestyle='-')
            ax.plot(np.log10(self.X_Artist["r_param"][:, idx]), r_h, label='Artist 4.5', linestyle='-')
            
            time_str = r_time[idx].strftime('%H:%M')
            ax.set_xlabel(r'$log_{10}(n_e)$ [$n/cm^3$]', fontsize=13)
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
        r_h = self.support["r_h"].flatten()
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
        
        sns.set(style="dark", context=None, palette=None)
        
        r_time = from_array_to_datetime(self.X_EISCAT["r_time"])
        n = len(self.selected_indices)
        
        fig, axes = plt.subplots(n, 3, figsize=(14, 5*n))
        
        if n == 1:
            axes = [axes]  # Ensure axes is iterable if there's only one subplot
        
        for i, idx in enumerate(self.selected_indices):
            # Plot ionogram image
            ionogram_img = self.X_Ionogram["r_param"][idx]
            ionogram_img = np.asarray(ionogram_img)  # Ensure it's a NumPy array
            ionogram_img = ionogram_img.astype(np.int64)  # Ensure it has a valid numeric type
            
                
            axes[i][0].imshow(ionogram_img)
            axes[i][0].set_title(f'Ionogram - Time: {r_time[idx].strftime("%H:%M")}', fontsize=15)
            axes[i][0].axis('off')
            
            # Plot selected measurements using existing method
            self.plot_single_measurement(axes[i][1], idx)
            
            # Plot error profiles using existing method
            self.plot_single_error(axes[i][2], idx)
            
            
        date_str = r_time[self.selected_indices[0]].strftime('%Y-%m-%d')
        fig.suptitle(f'Date: {date_str}', fontsize=20, y=1.01)
        plt.tight_layout()
        plt.show()
    
    def plot_single_measurement(self, ax, idx):
        """
        Plot a single selected measurement on a given axis.
        """
        r_time = from_array_to_datetime(self.X_EISCAT["r_time"])
        r_h = self.support["r_h"].flatten()
        
        ax.plot(np.log10(self.X_EISCAT["r_param"][:, idx]), r_h, label='EISCAT', linestyle='-')
        ax.plot(self.X_HNN["r_param"][:, idx], r_h, label='DL Model', linestyle='-')
        if self.X_Artist is not None:
            ax.plot(np.log10(self.X_Artist["r_param"][:, idx]), r_h, label='Artist 4.5', linestyle='-')
        
        time_str = r_time[idx].strftime('%H:%M')
        ax.set_xlabel(r'$log_{10}(n_e)$ [$n/cm^3$]', fontsize=13)
        ax.set_title(f'Measurements - Time: {time_str}', fontsize=15)
        ax.grid(True)
        ax.legend()

    def plot_single_error(self, ax, idx):
        """
        Plot a single error profile on a given axis.
        """
        r_time = from_array_to_datetime(self.X_EISCAT["r_time"])
        r_h = self.support["r_h"].flatten()
        error_hnn, error_artist, valid_artist_mask = self.calculate_errors(idx)
        
        # ax.plot(error_hnn, r_h, label='Error: EISCAT vs DL Model', linestyle='-', color='C1')
        # if np.any(valid_artist_mask):
        #     ax.plot(error_artist[valid_artist_mask], r_h[valid_artist_mask], label='Error: EISCAT vs Artist 4.5', linestyle='-', color='green')
        # else:
        #     ax.plot(error_artist, r_h, 'r-', linewidth=2, label='No Artist Data')
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
        
        
        # Set x-axis limits based on valid error data
        if np.any(valid_artist_mask):
            ax.set_xlim(left=0, right=max(np.max(error_hnn), np.max(error_artist[valid_artist_mask])) * 1.1)
        else:
           ax.set_xlim(left=0, right=np.max(error_hnn) * 1.1)
        
        
        time_str = r_time[idx].strftime('%H:%M')
        ax.set_xlabel('Error', fontsize=13)
        ax.set_title(f'Error Profiles - Time: {time_str}', fontsize=15)
        ax.grid(True)
        ax.legend()


    # def plot_ionogram_measurements_and_errors(self):
    #     """
    #     Plot the ionogram images, the selected measurements, and the error profiles for the selected dates.
    #     This method creates an nx3 grid of subplots, where n is the number of selected dates.
    #     """
    #     if not self.selected_indices:
    #         print("No measurements selected. Please run select_measurements(n) or select_measurements_by_datetime(datetimes) first.")
    #         return
        
    #     sns.set(style="dark", context=None, palette=None)
        
    #     r_time = from_array_to_datetime(self.X_EISCAT["r_time"])
    #     n = len(self.selected_indices)
        
    #     fig, axes = plt.subplots(n, 3, figsize=(15, 5 * n))
        
    #     if n == 1:
    #         axes = [axes]  # Ensure axes is iterable if there's only one subplot
        
    #     for i, idx in enumerate(self.selected_indices):
    #         # Plot ionogram image
    #         ionogram_img = self.X_ion["r_param"][idx]
    #         axes[i][0].imshow(ionogram_img)
    #         axes[i][0].set_title(f'Ionogram - Time: {r_time[idx].strftime("%H:%M")}', fontsize=15)
    #         axes[i][0].axis('off')
            
    #         # Plot selected measurements
    #         self.plot_single_measurement(axes[i][1], idx)
            
    #         # Plot error profiles
    #         self.plot_single_error(axes[i][2], idx)
        
    #     plt.tight_layout()
    #     plt.show()







# def plot_compare_all(support, X_EISCAT, X_HNN, X_Artist):
    
    
#     r_time   = X_EISCAT["r_time"]
#     art_time = X_Artist["r_time"]
    
    
#     r_h = support["r_h"].flatten()
    
#     ne_eis = np.log10(X_EISCAT["r_param"])
#     ne_hnn = X_HNN["r_param"]
#     ne_art = np.log10(X_Artist["r_param"])
    
    
    
#     r_time   = from_array_to_datetime(r_time)
#     art_time = from_array_to_datetime(art_time)
    
    
#     date_str = r_time[0].strftime('%Y-%m-%d')


#     # Creating the plots
#     fig, ax = plt.subplots(1, 3, figsize=(14, 8), sharey=True)
#     fig.tight_layout()
#     fig.suptitle(f'Date: {date_str}', fontsize=20, y=1.08)
    
    
#     x_limits = [r_time[0], r_time[-1]]
    
#     # Plotting original data
#     ne_EISCAT = ax[0].pcolormesh(r_time, r_h.flatten(), ne_eis, shading='auto', cmap='turbo', vmin=10, vmax=12)
#     ax[0].set_title('EISCAT UHF', fontsize=17)
#     ax[0].set_xlabel('Time [hours]', fontsize=13)
#     ax[0].set_ylabel('Altitude [km]', fontsize=15)
#     ax[0].xaxis.set_major_formatter(DateFormatter('%H:%M'))
    
    
#     # Plotting original data
#     ax[1].pcolormesh(r_time, r_h.flatten(), ne_hnn, shading='auto', cmap='turbo', vmin=10, vmax=12)
#     ax[1].set_title('DL model', fontsize=17)
#     ax[1].set_xlabel('Time [hours]', fontsize=13)
#     ax[1].xaxis.set_major_formatter(DateFormatter('%H:%M'))
    
    
#     # Plotting original data
#     ax[2].pcolormesh(art_time, r_h.flatten(), ne_art, shading='auto', cmap='turbo', vmin=10, vmax=12)
#     ax[2].set_title('Artist 4.5', fontsize=17)
#     ax[2].set_xlabel('Time [hours]', fontsize=13)
#     ax[2].xaxis.set_major_formatter(DateFormatter('%H:%M'))
#     ax[2].set_xlim(x_limits)
    
    
    
#     # Add colorbar for the original data
#     cbar = fig.colorbar(ne_EISCAT, ax=ax[2], orientation='vertical', fraction=0.1, pad=0.04)
#     cbar.set_label(r'$log_{10}(n_e)$ [n/cm$^3$]', fontsize=17)
    
#     plt.show()
















# def plot_compare_all(ne_true, ne_pred, ne_art, r_time, art_time):
    
    
    
    
    
#     # Eiscat altitudes
#     r_h = np.array([[91.5687711 ],[ 94.57444598],[ 97.57964223],[100.57010953],
#             [103.57141624],[106.57728701],[110.08393175],[114.60422289],
#             [120.1185208 ],[126.61221111],[134.1346149 ],[142.53945817],
#             [152.05174717],[162.57986185],[174.09833378],[186.65837945],
#             [200.15192581],[214.62769852],[230.12198695],[246.64398082],
#             [264.11728204],[282.62750673],[302.15668686],[322.70723831],
#             [344.19596481],[366.64409299],[390.113117]])
    
#     # Artist altitudes
#     art_h = np.arange(80, 485, 5)
    
#     date_str = r_time[0].strftime('%Y-%m-%d')


#     # Creating the plots
#     fig, ax = plt.subplots(1, 3, figsize=(14, 8), sharey=True)
#     fig.tight_layout()
#     fig.suptitle(f'Date: {date_str}', fontsize=15)
    
    
    
#     x_limits = [r_time[0], r_time[-1]]
#     print(x_limits)
    
#     # Plotting original data
#     ne_EISCAT = ax[0].pcolormesh(r_time, r_h.flatten(), ne_true.T, shading='auto', cmap='turbo', vmin=10, vmax=12)
#     ax[0].set_title('EISCAT UHF', fontsize=17)
#     ax[0].set_xlabel('Time [hours]')
#     ax[0].set_ylabel('Altitude [km]')
#     ax[0].xaxis.set_major_formatter(DateFormatter('%H:%M'))
    
    
#     # Plotting original data
#     ax[1].pcolormesh(r_time, r_h.flatten(), ne_pred.T, shading='auto', cmap='turbo', vmin=10, vmax=12)
#     ax[1].set_title('Ours', fontsize=17)
#     ax[1].set_xlabel('Time [hours]')
#     ax[1].xaxis.set_major_formatter(DateFormatter('%H:%M'))
    
    
#     # Plotting original data
#     ax[2].pcolormesh(art_time, art_h, ne_art.T, shading='auto', cmap='turbo', vmin=10, vmax=12)
#     ax[2].set_title('Artist 4.5', fontsize=17)
#     ax[2].set_xlabel('Time [hours]')
#     ax[2].xaxis.set_major_formatter(DateFormatter('%H:%M'))
#     ax[2].set_xlim(x_limits)
    
    
    
#     # Add colorbar for the original data
#     cbar = fig.colorbar(ne_EISCAT, ax=ax[2], orientation='vertical', fraction=0.03, pad=0.04, aspect=44, shrink=3)
#     cbar.set_label(r'$log_{10}(n_e)$ [g/cm$^3$]', fontsize=17)
    
#     fig.autofmt_xdate()
#     plt.show()



# (15,) <class 'numpy.ndarray'>
# 2019-03-07 20:15:00 <class 'datetime.datetime'>
# 2019-03-07
# [datetime.datetime(2019, 3, 7, 20, 15), datetime.datetime(2019, 3, 7, 23, 59)]


# (133,) <class 'numpy.ndarray'>
# 2019-03-07 20:15:00 <class 'datetime.datetime'>
# 2019-03-07
# [datetime.datetime(2019, 3, 7, 20, 15), datetime.datetime(2021, 1, 7, 0, 0)]





