# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 14:31:51 2024

@author: Kian Sartipzadeh
"""

import numpy as np
import matplotlib.pyplot as plt
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







class RadarInteractive:
    def __init__(self, X_EISCAT, X_KIAN, X_Artist, X_IRI, X_Ionogram, X_GEO):
        self.X_EISCAT = X_EISCAT
        self.X_KIAN = X_KIAN
        self.X_Artist = X_Artist
        self.X_IRI = X_IRI
        self.X_Ionogram = X_Ionogram
        self.X_GEO = X_GEO
        self.selected_indices = []
    
    
    
    
    
    # =============================================================================
    #                        Plotting Normalized Errors
    
        
    def error_function(self, X_eiscat, X):
        error = np.abs(X_eiscat - X)/X_eiscat
        return error
    
    
    def calculate_errors(self, idx):
        eiscat_param = np.log10(self.X_EISCAT["r_param"][:, idx])
        hnn_param = self.X_KIAN["r_param"][:, idx]
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
    


        
    # =============================================================================
    #                        Interactive Plot
    #                             (Start)
    

    def plot_interactive(self):
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
        ne_kian = self.X_KIAN["r_param"]
        ne_art = np.log10(self.X_Artist["r_param"])
        ne_iri = np.log10(self.X_IRI["r_param"])
        
        
        date_str = r_time[0].strftime('%Y-%m-%d')

        # Create the figure and layout
        fig = plt.figure(figsize=(18, 10))
        gs = GridSpec(2, 4, width_ratios=[1, 1, 0.05, 1], wspace=0.3)
        # gs = GridSpec(2, 2, height_ratios=[1, 1], width_ratios=[1, 1, 0.05], hspace=0.5, wspace=0.2)
        
        ax0 = fig.add_subplot(gs[0, 0])
        ax1 = fig.add_subplot(gs[0, 1], sharey=ax0)
        cax0 = fig.add_subplot(gs[0, 2])
        
        ax2 = fig.add_subplot(gs[1, 0])
        ax3 = fig.add_subplot(gs[1, 1], sharey=ax2)
        cax1 = fig.add_subplot(gs[1, 2])
        
        

        fig.suptitle(f'Date: {date_str}', fontsize=20, y=0.95)
        
        # Plot EISCAT data
        ne_EISCAT = ax0.pcolormesh(r_time, r_h, ne_eis, shading='auto', cmap='turbo', vmin=10, vmax=12)
        ax0.set_title('EISCAT UHF', fontsize=17)
        ax0.set_ylabel('Altitude [km]')
        ax0.xaxis.set_major_formatter(DateFormatter('%H:%M'))
        ax0.tick_params(labelbottom=False)
        
        # Plot HNN data
        ax1.pcolormesh(r_time, r_h, ne_kian, shading='auto', cmap='turbo', vmin=10, vmax=12)
        ax1.set_title('KIAN-Net', fontsize=17)
        ax1.xaxis.set_major_formatter(DateFormatter('%H:%M'))
        ax1.tick_params(labelleft=False, labelbottom=False)
        
        # Plot Artist data
        ax2.pcolormesh(art_time, r_h, ne_art, shading='auto', cmap='turbo', vmin=10, vmax=12)
        ax2.set_title('Artist 4.5', fontsize=17)
        ax2.set_xlabel('Time [hh:mm]')
        ax2.set_ylabel('Altitude [km]')
        ax2.xaxis.set_major_formatter(DateFormatter('%H:%M'))
        
        
        # Plot EISCAT data
        ax3.pcolormesh(r_time, r_h, ne_iri, shading='auto', cmap='turbo', vmin=10, vmax=12)
        ax3.set_title('IRI', fontsize=17)
        ax3.set_xlabel('Time [hh:mm]')
        ax3.xaxis.set_major_formatter(DateFormatter('%H:%M'))
        ax3.tick_params(labelleft=False)
        
        # Rotate x-axis labels
        for ax in [ax0, ax1, ax2, ax3]:
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='center')
        
        
        
        
        # Add colorbar
        cbar0 = fig.colorbar(ne_EISCAT, cax=cax0, orientation='vertical')
        cbar0.set_label(r'$log_{10}(n_e)$ [n/m$^3$]')
        
        cbar1 = fig.colorbar(ne_EISCAT, cax=cax1, orientation='vertical')
        cbar1.set_label(r'$log_{10}(n_e)$ [n/m$^3$]')
        
        # Detail plot axes
        with sns.axes_style("dark"):
            ax_iono = fig.add_subplot(gs[0, 3])
            ax_geo  = fig.add_subplot(gs[1, 3])
                
            
            # ax_detail = fig.add_subplot(gs[1, 1])
            # ax_error = fig.add_subplot(gs[1, 2])
            
        
        # # Function to update the detailed plot
        # def update_detail_plot(time_idx):
        #     ax_detail.clear()
        #     ax_detail.plot(ne_eis[:, time_idx], r_h, label="EISCAT", color='C0')
        #     ax_detail.plot(ne_hnn[:, time_idx], r_h, label="HNN", color='C1')
        #     ax_detail.plot(ne_art[:, time_idx], r_h, label="Artist", color='C2')
        #     ax_detail.set_title(f"{r_time[time_idx].strftime('%H:%M:%S')}")
        #     ax_detail.set_xlabel(r'$log_{10}(n_e)$ [n/m$^3$]')
        #     ax_detail.set_ylabel("Altitude [km]")
        #     ax_detail.legend()
        #     ax_detail.grid()
        #     fig.canvas.draw_idle()
        
        
        def update_geophys(time_idx):
            X_geo = self.X_GEO["r_param"]
            feature_labels=[
                'DoY/366', 'ToD/1440', 'SZ/44', 'Kp', 'R', 'Dst', 'ap', 'AE', 'AL', 'AU', 
                'PC_pot', 'F10_7', 'Ly_alp', 'Bx', 'By', 'Bz', 'dBx', 'dBy', 'dBz'
            ]
            with sns.axes_style("dark"):
                ax_geo.clear()
                ax_geo.bar(feature_labels, X_geo[:, time_idx], edgecolor='black')
                ax_geo.set_ylim(-1.05, 1.05)
                ax_geo.grid(True)
                plt.setp(ax_geo.xaxis.get_majorticklabels(), rotation=45, ha='center')
                fig.canvas.draw_idle()
        
        
        def update_ionogram(time_idx):
            ionogram_img = self.X_Ionogram["r_param"][time_idx]
            ionogram_img = np.asarray(ionogram_img)  # Ensure it's a NumPy array
            ionogram_img = ionogram_img.astype(np.int64)  # Ensure it has a valid numeric type
            
            ax_iono.clear()
            ax_iono.imshow(ionogram_img)
            ax_iono.set_title(f"{r_time[time_idx].strftime('%H:%M:%S')}")
            fig.canvas.draw_idle()
            
            
            
        
        # def update_error(time_idx):
            
        #     error_hnn, error_artist, valid_artist_mask = self.calculate_errors(time_idx)
            
        #     ax_error.clear()
        #     ax_error.plot(error_hnn, r_h, label='Error: EISCAT vs KIAN-Net', linestyle='-', color='C1')
        #     if np.any(valid_artist_mask):
        #         ax_error.plot(error_artist[valid_artist_mask], r_h[valid_artist_mask], label='Error: EISCAT vs Artist 4.5', linestyle='-', color='green')
        #         # Plot red line for NaN indices from the last valid value or first valid value
        #         nan_indices = np.where(~valid_artist_mask)[0]
        #         last_valid_index = np.max(np.where(valid_artist_mask)[0]) if np.any(valid_artist_mask) else None
        #         first_valid_index = np.min(np.where(valid_artist_mask)[0]) if np.any(valid_artist_mask) else None
        #         for nan_idx in nan_indices:
        #             if nan_idx < first_valid_index or nan_idx > last_valid_index:
        #                 # Draw a line from the closest valid value to the NaN index
        #                 closest_valid_index = first_valid_index if nan_idx < first_valid_index else last_valid_index
        #                 ax_error.plot([error_artist[closest_valid_index], error_artist[nan_idx]], [r_h[closest_valid_index], r_h[nan_idx]], 'r--', linewidth=2, label='Missing values' if nan_idx == nan_indices[0] else "")
        #     else:
        #         # Plot all error values if all are NaN
        #         ax_error.plot(error_artist, r_h,  'r-', linewidth=2, label='No Artist Data')
            
            
        #     # Set x-axis limits based on valid error data
        #     if np.any(valid_artist_mask):
        #         ax_error.set_xlim(left=0, right=max(np.max(error_hnn), np.max(error_artist[valid_artist_mask])) * 1.1)
        #     else:
        #        ax_error.set_xlim(left=0, right=np.max(error_hnn) * 1.1)
            
        #     ax_error.set_title(f"{r_time[time_idx].strftime('%H:%M:%S')}")
        #     ax_error.set_xlabel("Error")
        #     ax_error.set_ylabel("Altitude [km]")
        #     ax_error.legend()
        #     ax_error.grid()
        #     fig.canvas.draw_idle()
    
        
        # Handle click events
        def on_click(event):
            if event.inaxes in [ax0, ax1, ax2, ax3]:
                # Convert xdata to datetime for comparison
                click_time = mdates.num2date(event.xdata).replace(tzinfo=None)
                time_idx = np.argmin([abs((t - click_time).total_seconds()) for t in r_time])
                update_geophys(time_idx)
                update_ionogram(time_idx)
                # update_detail_plot(time_idx)
                # update_error(time_idx)
                
                
                # Clear any existing vertical lines
                for ax in [ax0, ax1, ax2, ax3]:
                    for line in ax.lines:
                        line.remove()
                
                # Add a red staple line to all three color plots
                for ax in [ax0, ax1, ax2, ax3]:
                    ax.axvline(event.xdata, color='red', linestyle='--', linewidth=2)
                        
        # Add interactivity
        fig.canvas.mpl_connect("button_press_event", on_click)
        Cursor(ax0, useblit=True, color='red', linewidth=1)
        Cursor(ax1, useblit=True, color='red', linewidth=1)
        Cursor(ax2, useblit=True, color='red', linewidth=1)
        Cursor(ax3, useblit=True, color='red', linewidth=1)
        plt.show()
    
    
    
    
    #                             (end)
    #                        Interactive Plot
    # =============================================================================
        
    
    

