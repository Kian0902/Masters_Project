# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 14:31:51 2024

@author: Kian Sartipzadeh
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from matplotlib.dates import DateFormatter
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
from matplotlib.widgets import Cursor
from matplotlib.patches import Patch
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
    
    
    

    def custom_camp(self, color1, color2, name='custom_cmap'):
        """
        Create a diverging colormap with four specified colors diverging from white.

        Parameters:
            color1: str or tuple
                The first color (e.g., 'blue' or (0.1, 0.2, 0.8)).
            color2: str or tuple
                The second color (e.g., 'cyan' or (0.1, 0.8, 0.8)).
            name: str
                Name for the custom colormap.

        Returns:
            matplotlib.colors.LinearSegmentedColormap: Custom colormap.
        """
        colors = [color1, 'white', color2]
        cmap = LinearSegmentedColormap.from_list(name, colors)
        return cmap
    
    
    # =============================================================================
    #                        Plotting Normalized Errors
    
        
    def error_function(self, X_eiscat, X):
        error = np.abs(X_eiscat - X)/X_eiscat
        return error
    
    
    def calculate_errors(self, idx):
        eiscat_param = np.log10(self.X_EISCAT["r_param"][:, idx])
        kian_param = self.X_KIAN["r_param"][:, idx]
        artist_param = np.log10(self.X_Artist["r_param"][:, idx])
        iri_param = np.log10(self.X_IRI["r_param"][:, idx])
        
        
        # Check if all values are NaN
        if np.all(np.isnan(artist_param)):
            valid_artist_mask = np.full(artist_param.shape, False)
            artist_param = np.zeros_like(artist_param)
        else:
            valid_artist_mask = ~np.isnan(artist_param)
            artist_param = np.nan_to_num(artist_param, nan=0)
        
        error_kian = self.error_function(eiscat_param, kian_param)
        error_artist = self.error_function(eiscat_param, artist_param)
        error_iri = self.error_function(eiscat_param, iri_param)
        
        return error_kian, error_artist, error_iri, valid_artist_mask
        
    
    

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
        fig = plt.figure(figsize=(24, 10))
        gs = GridSpec(3, 7, width_ratios=[1, 1, 0.05, 0.5, 1, 0.1, 1], height_ratios=[1, 0.01, 1], wspace=0.1)
        
        # First row
        ax0 = fig.add_subplot(gs[0, 0])
        ax1 = fig.add_subplot(gs[0, 1], sharey=ax0)
        cax0 = fig.add_subplot(gs[0, 2])
        # ax_space0 = fig.add_subplot(gs[0, 3])
        
        
        # Second row
        ax_ver_space = fig.add_subplot(gs[1, :])
        
        
        # Third row
        ax2 = fig.add_subplot(gs[2, 0])
        ax3 = fig.add_subplot(gs[2, 1], sharey=ax2)
        cax1 = fig.add_subplot(gs[2, 2])
        ax_space1 = fig.add_subplot(gs[:, 3])
        
        
        # All rows
        ax_space2 = fig.add_subplot(gs[:, 5])
        
        
        

        fig.suptitle(f'Date: {date_str}', fontsize=20, y=0.95)
        
        # Plot EISCAT data
        ne_EISCAT = ax0.pcolormesh(r_time, r_h, ne_eis, shading='auto', cmap='turbo', vmin=10, vmax=12)
        ax0.set_title('EISCAT UHF', fontsize=17)
        ax0.set_ylabel('Altitude [km]')
        ax0.xaxis.set_major_formatter(DateFormatter('%H:%M'))
        # ax0.tick_params(labelbottom=False)
        
        # Plot Kian-Net data
        ax1.pcolormesh(r_time, r_h, ne_kian, shading='auto', cmap='turbo', vmin=10, vmax=12)
        ax1.set_title('KIAN-Net', fontsize=17)
        ax1.xaxis.set_major_formatter(DateFormatter('%H:%M'))
        ax1.tick_params(labelleft=False)
        
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
        
        for ax in [ax_space1, ax_space2, ax_ver_space]:
            ax.set_axis_off()
        
        
        # Add colorbar
        cbar0 = fig.colorbar(ne_EISCAT, cax=cax0, orientation='vertical')
        cbar0.set_label(r'$log_{10}(n_e)$ [n/m$^3$]')
        
        cbar1 = fig.colorbar(ne_EISCAT, cax=cax1, orientation='vertical')
        cbar1.set_label(r'$log_{10}(n_e)$ [n/m$^3$]')
        
        # Detail plot axes
        with sns.axes_style("dark"):
            ax_iono = fig.add_subplot(gs[2, 4])
            ax_geo  = fig.add_subplot(gs[2, 6])
            ax_profile = fig.add_subplot(gs[0, 4])
            ax_error = fig.add_subplot(gs[0, 6])
            
            
        
        # Function to update the single Ne profiles plot
        def update_profile(time_idx):
            
            
            
            # Calculate R² scores
            r2_kian = r2_score(ne_eis[:, time_idx], ne_kian[:, time_idx])
            r2_iri = r2_score(ne_eis[:, time_idx], ne_iri[:, time_idx])
            
            if ne_art[:, time_idx] is not None:
                valid_artist = ~np.isnan(ne_eis[:, time_idx]) & ~np.isnan(ne_art[:, time_idx])
                r2_art = (r2_score(ne_eis[:, time_idx][valid_artist], ne_art[:, time_idx][valid_artist])
                             if np.any(valid_artist) else None)
            else:
                r2_art = None
            
            # Add R² scores to the plot as text annotations with color coding
            annotation_texts = []
            if r2_kian is not None:
                annotation_texts.append((f"R²= {r2_kian:.3f}", "C1"))  # Orange for HNN
            if r2_art is not None:
                annotation_texts.append((f"R²= {r2_art:.3f}", "C2"))  # Green for Artist 4.5
            if r2_iri is not None:
                annotation_texts.append((f"R²= {r2_iri:.3f}", "C3"))

            
            ax_profile.clear()
            # Choose the best location for the annotations
            for i, (text, color) in enumerate(annotation_texts):
                ax_profile.text(0.56, 87 - i * 4.5, text, transform=ax.transAxes,
                        fontsize=11, color=color, weight='bold',
                        bbox=None)
                
            ax_profile.plot(ne_eis[:, time_idx], r_h, label="EISCAT", color='C0', linewidth=2)
            ax_profile.plot(ne_kian[:, time_idx], r_h, label="KIAN-Net", color='C1', linestyle="--", linewidth=2)
            ax_profile.plot(ne_art[:, time_idx], r_h, label="Artist", color='C2', linestyle="-.", linewidth=2)
            ax_profile.plot(ne_iri[:, time_idx], r_h, label="IRI", color='C3', linestyle=":", linewidth=2)
            ax_profile.set_title(f"Electron Density Profile  {r_time[time_idx].strftime('%H:%M')}")
            ax_profile.set_xlabel(r'$log_{10}(n_e)$ [n/m$^3$]')
            ax_profile.set_ylabel("Altitude [km]")
            ax_profile.grid(True, color='white')
            ax_profile.legend(fontsize = 9, loc="center left")
            fig.canvas.draw_idle()
        
        
        
        def update_geophys(time_idx):
            X_geo = self.X_GEO["r_param"]
            feature_labels = [
                'DoY/366', 'ToD/1440', 'SZ/44', 'Kp', 'R', 'Dst', 'ap', 'AE', 'AL', 'AU', 
                'PC_pot', 'F10_7', 'Ly_alp', 'Bx', 'By', 'Bz', 'dBx', 'dBy', 'dBz'
            ]
            ax_geo.clear()
            ax_geo.set_title(f"Geophysical State Parameters")
            ax_geo.barh(feature_labels, X_geo[:, time_idx], edgecolor='black')  # Changed to barh
            ax_geo.set_xlim(-1.05, 1.05)  # Adjust limits for horizontal orientation
            ax_geo.grid(True, color='white')
            # ax_geo.yaxis.set_label_position("right")
            ax_geo.yaxis.tick_right()
            ax_geo.set_xlabel("Normalized (Z-score)") 
            fig.canvas.draw_idle()
            
        
        
        
        def update_ionogram(time_idx):
            ionogram_img = self.X_Ionogram["r_param"][time_idx]
            ionogram_img = np.asarray(ionogram_img)  # Ensure it's a NumPy array
            ionogram_img = ionogram_img.astype(np.int64)  # Ensure it has a valid numeric type
            
            n=9
            Frange = np.linspace(1, 9, 81)
            Zrange = np.linspace(80, 480, 81)
            
            ax_iono.clear()
            ax_iono.imshow(ionogram_img, origin='upper')
            x_ticks = np.linspace(0, ionogram_img.shape[1] - 1, n)
            y_ticks = np.linspace(0, ionogram_img.shape[0] - 1, n)
            
            x_tick_labels = np.linspace(Frange.min(), Frange.max(), n)
            y_tick_labels = np.linspace(Zrange.max(), Zrange.min(), n)
            
            ax_iono.set_xticks(x_ticks)
            ax_iono.set_xticklabels([f"{x:.0f}" for x in x_tick_labels])
            ax_iono.set_yticks(y_ticks)
            ax_iono.set_yticklabels([f"{y:.0f}" for y in y_tick_labels])
            
            ax_iono.set_xlabel("Freq [MHz]")
            ax_iono.set_ylabel("Virtual Altitude [km]")
            ax_iono.set_title(f"Ionogram   {r_time[time_idx].strftime('%H:%M')}")
            
            
            green_patch = Patch(color='green', label='X-mode')
            red_patch = Patch(color='red', label='O-mode')
            
            ax_iono.legend(handles=[red_patch, green_patch], loc='upper right', title="Modes", frameon=True)
            
            
            
            fig.canvas.draw_idle()
            
            
            
        
        def update_error(time_idx):
            
            error_kian, error_artist, error_iri, valid_artist_mask = self.calculate_errors(time_idx)
            
            ax_error.clear()
            ax_error.plot(error_kian, r_h, label='KIAN-Net', linestyle='-', color='C1')
            ax_error.plot(error_iri, r_h, label='IRI', linestyle='-', color='C3')
            if np.any(valid_artist_mask):
                ax_error.plot(error_artist[valid_artist_mask], r_h[valid_artist_mask], label='Artist 4.5', linestyle='-', color='green')
            
            ax_error.set_title(f"Relative Error")
            ax_error.legend(fontsize = 9)
            ax_error.grid(True, color='white')
            ax_error.tick_params(labelleft=False)
            fig.canvas.draw_idle()
    
        
        # Handle click events
        def on_click(event):
            if event.inaxes in [ax0, ax1, ax2, ax3]:
                # Convert xdata to datetime for comparison
                click_time = mdates.num2date(event.xdata).replace(tzinfo=None)
                time_idx = np.argmin([abs((t - click_time).total_seconds()) for t in r_time])
                update_geophys(time_idx)
                update_ionogram(time_idx)
                update_profile(time_idx)
                update_error(time_idx)
                
                
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
        
    

    
    def plot_proximity(self):
        r_time = from_array_to_datetime(self.X_EISCAT["r_time"])
        r_h = self.X_EISCAT["r_h"].flatten()

        # Logarithmic scaling for electron density
        ne_eis = np.log10(self.X_EISCAT["r_param"])
        ne_kian = self.X_KIAN["r_param"]
        ne_art = np.log10(self.X_Artist["r_param"])
        ne_iri = np.log10(self.X_IRI["r_param"])
        
        ne_art = np.nan_to_num(ne_art, nan=0)
        
        
        # # Calculate absolute differences
        diff_kian = self.error_function(ne_eis, ne_kian)
        diff_art = self.error_function(ne_eis, ne_art)
        diff_iri = self.error_function(ne_eis, ne_iri)
        
        diff_mag_iri = diff_iri - diff_kian
        diff_mag_art = diff_art - diff_kian
        
        
        # Identify the closest model at each point
        closest_model = np.argmin(np.stack([diff_art, diff_kian, diff_iri], axis=0), axis=0)
        
        
        
        # Define colors for the models
        model_colors = ['C0', 'C1', 'C2']  # art, kian, iri
        
        # Create a figure
        plt.figure(figsize=(14, 8))
        
        # Heatmap to show the closest model
        # plt.imshow(closest_model, aspect='auto', origin="lower", cmap=ListedColormap(model_colors))
        
        
        cmap_iri = self.custom_camp("blue", "red")
        
        plt.pcolormesh(r_time, r_h, diff_mag_art, cmap=cmap_iri, vmin=-0.11, vmax=0.11)
        plt.colorbar(label='Closest Model')
        plt.title('Closest Prediction Model to Ground Truth (ne_eis)')
        plt.xlabel('Time (day)')
        plt.ylabel('Altitude (km)')
        plt.show()
        
        
        cmap_art = self.custom_camp("green", "red")
        
        plt.pcolormesh(r_time, r_h, diff_mag_iri, cmap=cmap_art, vmin=-0.11, vmax=0.11)
        plt.colorbar(label='Closest Model')
        plt.title('Closest Prediction Model to Ground Truth (ne_eis)')
        plt.xlabel('Time (day)')
        plt.ylabel('Altitude (km)')
        plt.show()
        
        
        # date_str = r_time[0].strftime('%Y-%m-%d')
        
        # # Create a grid layout
        # fig = plt.figure(figsize=(12, 10))
        # gs = GridSpec(1, 2, width_ratios=[1, 0.05], wspace=0.1)
        
        # # Shared y-axis setup
        # ax0 = fig.add_subplot(gs[0])
        # cax = fig.add_subplot(gs[1])
        
        # fig.suptitle(f'Date: {date_str}', fontsize=20, y=1.03)
        
        # # Plot the comparison with RGB proximity gradient
        # mesh = ax0.pcolormesh(r_time, r_h, ne_eis, shading='auto', cmap=cmap)
        # ax0.set_title(f'Model Proximity to EISCAT', fontsize=17)
        # ax0.set_xlabel('Time [hours]', fontsize=13)
        # ax0.xaxis.set_major_formatter(DateFormatter('%H:%M'))
        # ax0.tick_params(labelleft=False)
        
        # # Rotate x-axis labels
        # for ax in [ax0]:
        #     plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='center')
        
        # # Add colorbar legend
        # legend_elements = [
        #     Patch(facecolor='red', edgecolor='black', label='KIAN-Net'),
        #     Patch(facecolor='blue', edgecolor='black', label='Artist 4.5'),
        #     Patch(facecolor='green', edgecolor='black', label='IRI')
        # ]
        # cax.legend(handles=legend_elements, loc='center', title='Closest Model', fontsize=13)
        # cax.axis('off')
        
        # plt.show()
