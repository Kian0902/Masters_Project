"""
Created on Mon Jan 27 16:23:19 2025

@author: Kian Sartipzadeh
"""


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, Normalize, LogNorm
from matplotlib.dates import DateFormatter
from matplotlib.patches import Patch
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
from matplotlib.widgets import Cursor
from datetime import datetime
from scipy.stats import linregress
from paper_utils import from_array_to_datetime, merge_nested_dict, merge_nested_pred_dict, merge_nested_peak_dict, inspect_dict

import seaborn as sns
# For all plots: sns.set(style="dark", context=None, palette=None)
# For single plot: with sns.axes_style("dark"):








class PaperPlotter:
    def __init__(self, X_EIS, X_KIAN, X_ION, X_GEO, X_ART, X_ECH):
        self.X_EIS = X_EIS
        self.X_KIAN = X_KIAN
        self.X_ION = X_ION
        self.X_GEO = X_GEO
        self.X_ART = X_ART
        self.X_ECH = X_ECH
        # self.selected_indices = []
    
    
    def relative_error(self, X1, X2):
        err =  (X2 - X1)/X1
        return err
    
    
    def plot_compare_all(self):
        
        # ___________ Getting Data ___________
        
        # Merging all global keys
        x_eis = merge_nested_dict(self.X_EIS)['All']
        x_kian = merge_nested_pred_dict(self.X_KIAN)['All']
        x_ion = merge_nested_pred_dict(self.X_ION)['All']
        x_geo = merge_nested_pred_dict(self.X_GEO)['All']
        x_art = merge_nested_dict(self.X_ART)['All']
        x_ech = merge_nested_dict(self.X_ECH)['All']
        
        r_time = from_array_to_datetime(x_eis['r_time'])
        r_h = x_eis['r_h'].flatten()
        
        # Ne-profiles
        ne_eis  = x_eis["r_param"]
        ne_kian = 10**x_kian["r_param"]
        ne_ion  = 10**x_ion["r_param"]
        ne_geo  = 10**x_geo["r_param"]
        ne_art  = x_art["r_param"]
        ne_ech  = x_ech["r_param"]
        
        date_str = r_time[0].strftime('%b-%Y')
        
        
        
        # ___________ Defining axes ___________
        fig = plt.figure(figsize=(12, 12))
        fig.suptitle(f'Comparrison Between Prediction Models and Ground Truth\nDate: {date_str}', fontsize=17, y=0.97)
        
        gs = GridSpec(6, 2, width_ratios=[1, 0.015], wspace=0.1, hspace=0.35)
        ax0 = fig.add_subplot(gs[0, 0])
        ax1 = fig.add_subplot(gs[1, 0], sharex=ax0)
        ax2 = fig.add_subplot(gs[2, 0], sharex=ax0)
        ax3 = fig.add_subplot(gs[3, 0], sharex=ax0)
        ax4 = fig.add_subplot(gs[4, 0], sharex=ax0)
        ax5 = fig.add_subplot(gs[5, 0], sharex=ax0)
        cax = fig.add_subplot(gs[:, 1])
        
        
        
        #  ___________ Creating Plots  ___________
        MIN, MAX = 1e10, 1e12
        
        ne = ax0.pcolormesh(r_time, r_h, ne_eis, shading='gouraud', cmap='turbo', norm=LogNorm(vmin=MIN, vmax=MAX))
        ax1.pcolormesh(r_time, r_h, ne_kian, shading='gouraud', cmap='turbo', norm=LogNorm(vmin=MIN, vmax=MAX))
        ax2.pcolormesh(r_time, r_h, ne_ion, shading='gouraud', cmap='turbo', norm=LogNorm(vmin=MIN, vmax=MAX))
        ax3.pcolormesh(r_time, r_h, ne_geo, shading='gouraud', cmap='turbo', norm=LogNorm(vmin=MIN, vmax=MAX))
        ax4.pcolormesh(r_time, r_h, ne_art, shading='gouraud', cmap='turbo', norm=LogNorm(vmin=MIN, vmax=MAX))
        ax5.pcolormesh(r_time, r_h, ne_ech, shading='gouraud', cmap='turbo', norm=LogNorm(vmin=MIN, vmax=MAX))
        
        # Set titles and labels
        ax0.set_title('(a) EISCAT UHF', fontsize=16)
        ax1.set_title('(b) KIAN-Net', fontsize=16)
        ax2.set_title('(c) Iono-CNN', fontsize=16)
        ax3.set_title('(d) Geo-DMLP', fontsize=16)
        ax4.set_title('(e) Artist 5.0', fontsize=16)
        ax5.set_title('(f) E-Chaim', fontsize=16)
        
        # y-labels
        fig.supylabel('Altitude [km]', x=0.075)
        # ax0.set_ylabel('Altitude [km]', fontsize=13)
        # ax1.set_ylabel('Altitude [km]', fontsize=13)
        # ax2.set_ylabel('Altitude [km]', fontsize=13)
        # ax3.set_ylabel('Altitude [km]', fontsize=13)
        # ax4.set_ylabel('Altitude [km]', fontsize=13)
        # ax5.set_ylabel('Altitude [km]', fontsize=13)
        
        # x-label
        ax5.set_xlabel('UT [dd hh:mm]', fontsize=13)
        
        # Ticks
        ax0.tick_params(labelbottom=False)  # Hide x-ticks on EISCAT plot
        ax1.tick_params(labelbottom=False)
        ax2.tick_params(labelbottom=False)
        ax3.tick_params(labelbottom=False)
        ax4.tick_params(labelbottom=False)
        ax5.xaxis.set_major_formatter(DateFormatter('%d %H:%M'))
        
        plt.setp(ax5.xaxis.get_majorticklabels(), rotation=45, ha='center')
        
        # Add colorbar
        cbar = fig.colorbar(ne, cax=cax, orientation='vertical')
        cbar.set_label('$log_{10}$ $N_e$  (m$^{-3}$)', fontsize=13)
        
        plt.show()
    
    
    def plot_peaks(self):
        
        # ___________ Getting Data ___________
        
        # Merging all global keys
        x_eis = merge_nested_peak_dict(self.X_EIS)['All']
        x_kian = merge_nested_peak_dict(self.X_KIAN)['All']
        x_ion = merge_nested_peak_dict(self.X_ION)['All']
        x_geo = merge_nested_peak_dict(self.X_GEO)['All']
        x_art = merge_nested_peak_dict(self.X_ART)['All']
        x_ech = merge_nested_peak_dict(self.X_ECH)['All']
        
        r_time = from_array_to_datetime(x_eis['r_time'])
        r_h = x_eis['r_h'].flatten()
        
        # Ne-profiles
        ne_eis  = x_eis["r_param"]
        ne_kian = 10**x_kian["r_param"]
        ne_ion  = 10**x_ion["r_param"]
        ne_geo  = 10**x_geo["r_param"]
        ne_art  = x_art["r_param"]
        ne_ech  = x_ech["r_param"]
        
        
        peak_eis = x_eis['r_h_peak']
        peak_kian = x_kian['r_h_peak']
        peak_ion = x_ion['r_h_peak']
        peak_geo = x_geo['r_h_peak']
        peak_art = x_art['r_h_peak']
        peak_ech = x_ech['r_h_peak']
        
        
        date_str = r_time[0].strftime('%b-%Y')
        
        # print(peak_kian[0,:].shape)
        
        # ___________ Defining axes ___________
        fig = plt.figure(figsize=(12, 12))
        fig.suptitle(f'Comparrison Between Prediction Models and Ground Truth\nDate: {date_str}', fontsize=17, y=0.97)
        
        gs = GridSpec(6, 2, width_ratios=[1, 0.015], wspace=0.1, hspace=0.35)
        ax0 = fig.add_subplot(gs[0, 0])
        ax1 = fig.add_subplot(gs[1, 0], sharex=ax0)
        ax2 = fig.add_subplot(gs[2, 0], sharex=ax0)
        ax3 = fig.add_subplot(gs[3, 0], sharex=ax0)
        ax4 = fig.add_subplot(gs[4, 0], sharex=ax0)
        ax5 = fig.add_subplot(gs[5, 0], sharex=ax0)
        cax = fig.add_subplot(gs[:, 1])
        
        
        
        #  ___________ Creating Plots  ___________
        MIN, MAX = 1e10, 1e12
        
        
        ne = ax0.pcolormesh(r_time, r_h, ne_eis, shading='gouraud', cmap='turbo', norm=LogNorm(vmin=MIN, vmax=MAX))
        ax0.plot(r_time, peak_eis[0, :], color="magenta")
        ax0.plot(r_time, peak_eis[1, :], color="magenta")
        
        ax1.pcolormesh(r_time, r_h, ne_kian, shading='gouraud', cmap='turbo', norm=LogNorm(vmin=MIN, vmax=MAX))
        ax1.plot(r_time, peak_kian[0, :], color="magenta")
        ax1.plot(r_time, peak_kian[1, :], color="magenta")
        
        ax2.pcolormesh(r_time, r_h, ne_ion, shading='gouraud', cmap='turbo', norm=LogNorm(vmin=MIN, vmax=MAX))
        ax2.plot(r_time, peak_ion[0, :], color="magenta")
        ax2.plot(r_time, peak_ion[1, :], color="magenta")
        
        ax3.pcolormesh(r_time, r_h, ne_geo, shading='gouraud', cmap='turbo', norm=LogNorm(vmin=MIN, vmax=MAX))
        ax3.plot(r_time, peak_geo[0, :], color="magenta")
        ax3.plot(r_time, peak_geo[1, :], color="magenta")
        
        ax4.pcolormesh(r_time, r_h, ne_art, shading='gouraud', cmap='turbo', norm=LogNorm(vmin=MIN, vmax=MAX))
        ax4.plot(r_time, peak_art[0, :], color="magenta")
        ax4.plot(r_time, peak_art[1, :], color="magenta")
        
        ax5.pcolormesh(r_time, r_h, ne_ech, shading='gouraud', cmap='turbo', norm=LogNorm(vmin=MIN, vmax=MAX))
        ax5.plot(r_time, peak_ech[0, :], color="magenta")
        ax5.plot(r_time, peak_ech[1, :], color="magenta")
        
        
        # Set titles and labels
        ax0.set_title('(a) EISCAT UHF', fontsize=16)
        ax1.set_title('(b) KIAN-Net', fontsize=16)
        ax2.set_title('(c) Iono-CNN', fontsize=16)
        ax3.set_title('(d) Geo-DMLP', fontsize=16)
        ax4.set_title('(e) Artist 5.0', fontsize=16)
        ax5.set_title('(f) E-Chaim', fontsize=16)
        
        # y-labels
        fig.supylabel('Altitude [km]', x=0.075)
        # ax0.set_ylabel('Altitude [km]', fontsize=13)
        # ax1.set_ylabel('Altitude [km]', fontsize=13)
        # ax2.set_ylabel('Altitude [km]', fontsize=13)
        # ax3.set_ylabel('Altitude [km]', fontsize=13)
        # ax4.set_ylabel('Altitude [km]', fontsize=13)
        # ax5.set_ylabel('Altitude [km]', fontsize=13)
        
        # x-label
        ax5.set_xlabel('UT [dd hh:mm]', fontsize=13)
        
        # Ticks
        ax0.tick_params(labelbottom=False)  # Hide x-ticks on EISCAT plot
        ax1.tick_params(labelbottom=False)
        ax2.tick_params(labelbottom=False)
        ax3.tick_params(labelbottom=False)
        ax4.tick_params(labelbottom=False)
        ax5.xaxis.set_major_formatter(DateFormatter('%d %H:%M'))
        
        plt.setp(ax5.xaxis.get_majorticklabels(), rotation=45, ha='center')
        
        # Add colorbar
        cbar = fig.colorbar(ne, cax=cax, orientation='vertical')
        cbar.set_label('$log_{10}$ $N_e$  (m$^{-3}$)', fontsize=13)
        
        plt.show()
        
        
        
        
    
    
    def compute_r2_score(self, y1, y2):
        """Calculate R2 score between two lines."""
        ss_tot = np.sum((y1 - np.mean(y1)) ** 2)
        ss_res = np.sum((y1 - y2) ** 2)
        return 1 - (ss_res / ss_tot)
    
    
    
    def plot_compare_all_peak_altitudes(self):
        sns.set(style="dark", context=None, palette=None)
    
        
        
        
        # Merging all global keys
        X_eis = merge_nested_peak_dict(self.X_EIS)['All']
        X_kian = merge_nested_peak_dict(self.X_KIAN)['All']
        # x_ion = merge_nested_peak_dict(self.X_ION)['All']
        # x_geo = merge_nested_peak_dict(self.X_GEO)['All']
        X_art = merge_nested_peak_dict(self.X_ART)['All']
        # x_ech = merge_nested_peak_dict(self.X_ECH)['All']
        
        
        eis_h_peak = X_eis["r_h_peak"]
        kian_h_peak = X_kian["r_h_peak"]
        art_h_peak = X_art["r_h_peak"]
        
        
        # Convert time data to datetime objects
        r_time = from_array_to_datetime(X_eis["r_time"])
        art_time = from_array_to_datetime(X_art["r_time"])
        
        
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
                    
                    ax.text(Max-25, Min+10, s=rf"$\mathbf{{R^2={kian_r2_line:.3f}}}$", color="C1", fontsize=15)
                    
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
                    
                    ax.text(Max-5, Min+0.1, s=rf"$\mathbf{{R^{2}={artist_r2_line:.3f}}}$", color="C2", fontsize=15)
                    
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
        plot_region(ax_e, region=0, Min=90, Max=170, title='E-region')
        plot_region(ax_f, region=1, Min=90, Max=400, title='F-region')
    
        # Adjust F-region plot labels
        ax_f.set_xlabel('MODEL  $log_{10}\,(ne)$', fontsize=15)
    
        plt.show()
    
    
    
    
    
    
    def plot_compare_all_peak_densities(self):
        sns.set(style="dark", context=None, palette=None)
    
        # Merging all global keys
        X_eis = merge_nested_peak_dict(self.X_EIS)['All']
        X_kian = merge_nested_peak_dict(self.X_KIAN)['All']
        X_ion = merge_nested_peak_dict(self.X_ION)['All']
        X_geo = merge_nested_peak_dict(self.X_GEO)['All']
        X_art = merge_nested_peak_dict(self.X_ART)['All']
        X_ech = merge_nested_peak_dict(self.X_ECH)['All']
        
        
        eis_param_peak = np.log10(X_eis["r_param_peak"])
        kian_param_peak = X_kian["r_param_peak"]
        ion_param_peak = X_ion["r_param_peak"]
        geo_param_peak = X_geo["r_param_peak"]
        art_param_peak = np.log10(X_art["r_param_peak"])
        ech_param_peak = np.log10(X_ech["r_param_peak"])
        
        # print(art_param_peak)
        
        
        # Convert time data to datetime objects
        r_time = from_array_to_datetime(X_eis["r_time"])
        art_time = from_array_to_datetime(X_art["r_time"])
        
        # Format date for plot title
        date_str = r_time[0].strftime('%b-%Y')
        
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
            valid_ion = ~np.isnan(ion_param_peak[region, :]) & ~np.isnan(eis_param_peak[region, :])
            valid_geo = ~np.isnan(geo_param_peak[region, :]) & ~np.isnan(eis_param_peak[region, :])
            valid_artist = ~np.isnan(art_param_peak[region, :]) & ~np.isnan(eis_param_peak[region, :])
            valid_ech = ~np.isnan(ech_param_peak[region, :]) & ~np.isnan(eis_param_peak[region, :])
            
            # Determine min and max values for regression calculation based on valid data
            kian_x = kian_param_peak[region, valid_kian]
            kian_y = eis_param_peak[region, valid_kian]
            ion_x = ion_param_peak[region, valid_ion]
            ion_y = eis_param_peak[region, valid_ion]
            geo_x = geo_param_peak[region, valid_geo]
            geo_y = eis_param_peak[region, valid_geo]
            art_x = art_param_peak[region, valid_artist]
            art_y = eis_param_peak[region, valid_artist]
            ech_x = ech_param_peak[region, valid_ech]
            ech_y = eis_param_peak[region, valid_ech]
            
            # data_min = min(np.min(kian_x), np.min(art_x), np.min(kian_y), np.min(art_y))
            # data_max = max(np.max(kian_x), np.max(art_x), np.max(kian_y), np.max(art_y))
            
            # Plot diagonal reference line
            ax.plot([Min, Max], [Min, Max], color="C0", label="EISCAT", linewidth=3, zorder=1)
            
            # Scatter plots for KIAN-Net and Artist 4.5
            ax.scatter(kian_x, kian_y, s=40, color="C1", label="KIAN-Net", edgecolors="black")
            
            # Regression for KIAN-Net
            if valid_kian.any():
                slope_kian, intercept_kian, _, _, _ = linregress(kian_x, kian_y)
                if not np.isnan(slope_kian) and not np.isnan(intercept_kian):
                    
                    kian_reg_line = slope_kian * np.array([Min, Max]) + intercept_kian
                    diagonal_line = np.array([Min, Max])  # Diagonal reference
                    kian_r2_line = self.compute_r2_score(diagonal_line, kian_reg_line)
                    # print(f"KIAN-Net R^2 with diagonal ({title}): {kian_r2_line}")
        
                    ax.plot([Min, Max], kian_reg_line, color="C1", linestyle="--", linewidth=2,
                            label=" ", zorder=2)
                    
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
                    
                    
            ax.scatter(art_x, art_y, s=40, color="C2", label="Artist 5.0", edgecolors="black")
            # Regression for Artist
            if valid_artist.any():
                slope_artist, intercept_artist, _, _, _ = linregress(art_x, art_y)
                if not np.isnan(slope_artist) and not np.isnan(intercept_artist):
                    
                    artist_reg_line = slope_artist * np.array([Min, Max]) + intercept_artist
                    diagonal_line = np.array([Min, Max])
                    artist_r2_line = self.compute_r2_score(diagonal_line, artist_reg_line)
                    # print(f"Artist R^2 with diagonal ({title}): {artist_r2_line}")
                    
                    ax.plot([Min, Max], artist_reg_line, color="C2", linestyle="--", linewidth=2,
                            label=" ", zorder=2)
                    
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
                    
                    
                    
            ax.scatter(ech_x, ech_y, s=40, color="C3", label="E-CHAIM", edgecolors="black")
            # Regression for Artist
            if valid_ech.any():
                slope_ech, intercept_ech, _, _, _ = linregress(ech_x, ech_y)
                if not np.isnan(slope_ech) and not np.isnan(intercept_ech):
                    
                    ech_reg_line = slope_ech * np.array([Min, Max]) + intercept_ech
                    diagonal_line = np.array([Min, Max])
                    ech_r2_line = self.compute_r2_score(diagonal_line, ech_reg_line)
                    # print(f"Artist R^2 with diagonal ({title}): {artist_r2_line}")
                    
                    ax.plot([Min, Max], ech_reg_line, color="C3", linestyle="--", linewidth=2,
                            label=" ", zorder=2)
                    
                    ax.text(Max-1, Min+0.4, s=rf"$\mathbf{{R^{2}={ech_r2_line:.3f}}}$", color="C3", fontsize=15)
                    
                    # # Calculate R^2 for regression line
                    # y_pred = slope * art_x + intercept
                    # y_mean = np.mean(art_y)
                    # ss_tot = np.sum((art_y - y_mean) ** 2)
                    # ss_res = np.sum((art_y - y_pred) ** 2)
                    # artist_r2 = 1 - (ss_res / ss_tot)
                    # print(f"Artist R^2 ({title}): {artist_r2}")
                    
                    # ax.plot([Min, Max], [slope * Min + intercept, slope * Max + intercept], 
                    #         color="C2", linestyle="--", linewidth=2, label=f" Reg $R^2=${artist_r2:.3}", zorder=2)
            
            
            ax.scatter(geo_x, geo_y, s=40, color="C4", label="Geo-DMLP", edgecolors="black")
            # Regression for geo
            if valid_geo.any():
                slope_geo, intercept_geo, _, _, _ = linregress(geo_x, geo_y)
                if not np.isnan(slope_geo) and not np.isnan(intercept_geo):
                    
                    geo_reg_line = slope_geo * np.array([Min, Max]) + intercept_geo
                    diagonal_line = np.array([Min, Max])
                    geo_r2_line = self.compute_r2_score(diagonal_line, geo_reg_line)
                    # print(f"Artist R^2 with diagonal ({title}): {artist_r2_line}")
                    
                    ax.plot([Min, Max], geo_reg_line, color="C4", linestyle="--", linewidth=2,
                            label=" ", zorder=2)
                    
                    ax.text(Max-1, Min+0.55, s=rf"$\mathbf{{R^{2}={geo_r2_line:.3f}}}$", color="C4", fontsize=15)
                    
                    # # Calculate R^2 for regression line
                    # y_pred = slope * art_x + intercept
                    # y_mean = np.mean(art_y)
                    # ss_tot = np.sum((art_y - y_mean) ** 2)
                    # ss_res = np.sum((art_y - y_pred) ** 2)
                    # artist_r2 = 1 - (ss_res / ss_tot)
                    # print(f"Artist R^2 ({title}): {artist_r2}")
                    
                    # ax.plot([Min, Max], [slope * Min + intercept, slope * Max + intercept], 
                    #         color="C2", linestyle="--", linewidth=2, label=f" Reg $R^2=${artist_r2:.3}", zorder=2)
            
            
            ax.scatter(ion_x, ion_y, s=40, color="C5", label="Iono-CNN", edgecolors="black")
            # Regression for ion
            if valid_ion.any():
                slope_ion, intercept_ion, _, _, _ = linregress(ion_x, ion_y)
                if not np.isnan(slope_ion) and not np.isnan(intercept_ion):
                    
                    ion_reg_line = slope_ion * np.array([Min, Max]) + intercept_ion
                    diagonal_line = np.array([Min, Max])
                    ion_r2_line = self.compute_r2_score(diagonal_line, ion_reg_line)
                    # print(f"Artist R^2 with diagonal ({title}): {artist_r2_line}")
                    
                    ax.plot([Min, Max], ion_reg_line, color="C5", linestyle="--", linewidth=2,
                            label=" ", zorder=2)
                    
                    ax.text(Max-1, Min+0.70, s=rf"$\mathbf{{R^{2}={ion_r2_line:.3f}}}$", color="C5", fontsize=15)
                    
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
            ax_f.legend(fontsize=11)
            ax.grid(True)
    
        # Plot E-region and F-region
        plot_region(ax_e, region=0, Min=9, Max=12, title='E-region')
        plot_region(ax_f, region=1, Min=9, Max=12, title='F-region')
    
        # Adjust region plot labels
        ax_e.set_xlabel(r'Prediction  $log_{10}\,(n_e)$', fontsize=15)
        ax_e.set_ylabel(r'EISCAT UHF  $log_{10}\,(n_e)$', fontsize=15)
        ax_f.set_xlabel(r'Prediction  $log_{10}\,(n_e)$', fontsize=15)
        
        plt.show()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
#     def plot_vertical_compare_ne(self):
#         """
#         Function for comparing EISCAT UHF to KIAN-Net, Artist 4.5, and IRI in a 4x1 vertical grid.
#         """
#         r_time = from_array_to_datetime(self.X_EISCAT["r_time"])
#         r_h = self.X_EISCAT["r_h"].flatten()
        
#         # Linear scaling for electron density
#         ne_eis = self.X_EISCAT["r_param"]
#         ne_kian = 10**self.X_KIAN["r_param"]
#         ne_art = self.X_Artist["r_param"]
#         ne_iri = self.X_IRI["r_param"]
        
#         date_str = r_time[0].strftime('%Y-%m-%d')
        
#         # Figure
#         fig = plt.figure(figsize=(6, 10))
#         fig.suptitle(f'Date: {date_str}', fontsize=17, y=0.98)
        
#         gs = GridSpec(4, 2, width_ratios=[1, 0.05], height_ratios=[1, 1, 1, 1], hspace=0.3, wspace=0.05)
        
#         # Axes for plots
#         ax0 = fig.add_subplot(gs[0, 0])   # EISCAT UHF
#         ax1 = fig.add_subplot(gs[1, 0], sharex=ax0, sharey=ax0)  # KIAN-Net
#         ax2 = fig.add_subplot(gs[2, 0], sharex=ax0, sharey=ax0)  # Artist 4.5
#         ax3 = fig.add_subplot(gs[3, 0], sharex=ax0, sharey=ax0)  # IRI
#         cax = fig.add_subplot(gs[:, 1])   # Shared Colorbar
        
#         # Global settings
#         MIN, MAX = 1e10, 1e12
#         subtit_size, xlab_size, ylab_size, cbar_size = 17, 13, 13, 13
        
#         # Plot EISCAT UHF
#         ne_EISCAT = ax0.pcolormesh(r_time, r_h, ne_eis, shading='auto', cmap='turbo', norm=LogNorm(vmin=MIN, vmax=MAX))
#         ax0.set_title('EISCAT UHF', fontsize=subtit_size)
#         ax0.set_ylabel('Altitude [km]', fontsize=ylab_size)
#         ax0.xaxis.set_major_formatter(DateFormatter('%H:%M'))
#         ax0.tick_params(labelbottom=False)
        
#         # Plot KIAN-Net
#         ax1.pcolormesh(r_time, r_h, ne_kian, shading='auto', cmap='turbo', norm=LogNorm(vmin=MIN, vmax=MAX))
#         ax1.set_title('KIAN-Net', fontsize=subtit_size)
#         ax1.set_ylabel('Altitude [km]', fontsize=ylab_size)
#         ax1.xaxis.set_major_formatter(DateFormatter('%H:%M'))
#         ax1.tick_params(labelbottom=False)
        
#         # Plot Artist 4.5
#         ax2.pcolormesh(r_time, r_h, ne_art, shading='auto', cmap='turbo', norm=LogNorm(vmin=MIN, vmax=MAX))
#         ax2.set_title('Artist 4.5', fontsize=subtit_size)
#         ax2.set_ylabel('Altitude [km]', fontsize=ylab_size)
#         ax2.xaxis.set_major_formatter(DateFormatter('%H:%M'))
#         ax2.tick_params(labelbottom=False)
        
#         # Plot IRI
#         ax3.pcolormesh(r_time, r_h, ne_iri, shading='auto', cmap='turbo', norm=LogNorm(vmin=MIN, vmax=MAX))
#         ax3.set_title('IRI', fontsize=subtit_size)
#         ax3.set_xlabel('Time [UT]', fontsize=xlab_size)
#         ax3.set_ylabel('Altitude [km]', fontsize=ylab_size)
#         ax3.xaxis.set_major_formatter(DateFormatter('%H:%M'))
        
#         # Shared colorbar
#         cbar = fig.colorbar(ne_EISCAT, cax=cax, orientation='vertical')
#         cbar.set_label('$n_e$ [n/m$^3$]', fontsize=cbar_size)
        
#         # Rotate x-axis labels
#         for ax in [ax0, ax1, ax2, ax3]:
#             plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='center')
        
#         plt.show()

    
    
    
#     def plot_compare_error(self):
#         """
#         Function for comparing relative errors between EISCAT UHF, KIAN-Net, Artist 4.5 and IRI.
#         """
#         r_time = from_array_to_datetime(self.X_EISCAT["r_time"])
#         r_h = self.X_EISCAT["r_h"].flatten()
        
#         # Linear scaling for electron density
#         ne_eis = self.X_EISCAT["r_param"]
#         ne_kian = 10**self.X_KIAN["r_param"]
#         ne_art = self.X_Artist["r_param"]
#         ne_iri = self.X_IRI["r_param"]
        
        
#         err_kian = self.relative_error(ne_eis, ne_kian)
#         err_art = self.relative_error(ne_eis, ne_art)
#         err_iri = self.relative_error(ne_eis, ne_iri)
        
        
#         # date_str = r_time[0].strftime('%Y-%m-%d')

#         # Figure 
#         fig = plt.figure(figsize=(12, 5))
#         gs = GridSpec(1, 4, width_ratios=[1, 1, 1, 0.05], wspace=0.1)
        
        
#         # ___________ Defining axes ___________
#         # 1st row
#         ax00 = fig.add_subplot(gs[0, 0])                    # KIAN-Net
#         ax01 = fig.add_subplot(gs[0, 1], sharey=ax00)       # Artist 4.5
#         ax02 = fig.add_subplot(gs[0, 2], sharey=ax00)       # IRI
#         cax03 = fig.add_subplot(gs[0, 3])                   # Colorbar
        
        
#         #  ___________ Creating Plots  ___________
#         MIN, MAX = -1, 1
#         subtit_size, xlab_size, ylab_size, cbar_size = 17, 13, 13, 13
        
#         # Error KIAN-Net
#         error_kian = ax00.pcolormesh(r_time, r_h, err_kian, shading='auto', cmap='bwr', vmin=MIN, vmax=MAX)
#         ax00.set_title('KIAN-Net vs EISCAT', fontsize=subtit_size)
#         ax00.set_xlabel('Time [UT]', fontsize=xlab_size)
#         ax00.set_ylabel('Altitude [km]', fontsize=ylab_size)
#         ax00.xaxis.set_major_formatter(DateFormatter('%H:%M'))
        
#         # Error Artist 4.5
#         err_art_nan = np.zeros(err_art.shape)  # For grey color
#         ax01.pcolormesh(r_time, r_h, err_art_nan, shading='auto', cmap='gray', vmin=MIN, vmax=MAX, zorder=0)
#         ax01.pcolormesh(r_time, r_h, err_art, shading='auto', cmap='bwr', vmin=MIN, vmax=MAX, zorder=2)
#         ax01.set_title('Artist 4.5 vs EISCAT', fontsize=subtit_size)
#         ax01.set_xlabel('Time [UT]', fontsize=xlab_size)
#         ax01.xaxis.set_major_formatter(DateFormatter('%H:%M'))
#         ax01.tick_params(labelleft=False)
        
        
#         # Error IRI
#         ax02.pcolormesh(r_time, r_h, err_iri, shading='auto', cmap='bwr', vmin=MIN, vmax=MAX)
#         ax02.set_title('IRI vs EISCAT', fontsize=subtit_size)
#         ax02.set_xlabel('Time [UT]', fontsize=xlab_size)
#         ax02.xaxis.set_major_formatter(DateFormatter('%H:%M'))
#         ax02.tick_params(labelleft=False)
        
#         # Colorbar
#         cbar03 = fig.colorbar(error_kian, cax=cax03, orientation='vertical')
#         cbar03.set_label('Relative Error', fontsize=cbar_size)
        
    
#         # Rotate x-axis labels
#         for ax in [ax00, ax01, ax02]:
#             plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='center')
        

#         plt.show()


    
# class AdvancedPaperPlotter:
#     def __init__(self, X_EISCAT, X_KIAN, X_Artist, X_IRI, X_Ionogram, X_GEO):
#         self.X_EISCAT = X_EISCAT
#         self.X_KIAN = X_KIAN
#         self.X_Artist = X_Artist
#         self.X_IRI = X_IRI
#         self.X_Ionogram = X_Ionogram
#         self.X_GEO = X_GEO
#         # self.selected_days = []
    
#     def plot_vertical_compare_3days_ne(self, selected_days):
#         """
#         Function for comparing EISCAT UHF to KIAN-Net, Artist 4.5, and IRI over 3 selected days
#         in a 4x3 grid where each column represents a day, with a color bar per row on the far right.
#         """
        
#         fig = plt.figure(figsize=(12, 15))
#         fig.suptitle('Comparison Over 3 Days', fontsize=20, y=0.96)
        
#         gs = GridSpec(4, 4, width_ratios=[1, 1, 1, 0.05], height_ratios=[1, 1, 1, 1], hspace=0.3, wspace=0.2)
        
#         # Global settings
#         MIN, MAX = 1e10, 1e12
#         subtit_size, xlab_size, ylab_size, cbar_size = 17, 13, 13, 13
        
#         axes = []
#         model_pcolormeshes = [None] * 4  # To store pcolormesh objects for each row
        
#         for i, date in enumerate(selected_days):
#             r_time = from_array_to_datetime(self.X_EISCAT[date]["r_time"])
#             r_h = self.X_EISCAT[date]["r_h"].flatten()
            
#             ne_eis = self.X_EISCAT[date]["r_param"]
#             ne_kian = 10**self.X_KIAN[date]["r_param"]
#             ne_art = self.X_Artist[date]["r_param"]
#             ne_iri = self.X_IRI[date]["r_param"]
            
#             # Axes for plots
#             ax0 = fig.add_subplot(gs[0, i])   # EISCAT UHF
#             ax1 = fig.add_subplot(gs[1, i], sharex=ax0, sharey=ax0)  # KIAN-Net
#             ax2 = fig.add_subplot(gs[2, i], sharex=ax0, sharey=ax0)  # Artist 4.5
#             ax3 = fig.add_subplot(gs[3, i], sharex=ax0, sharey=ax0)  # IRI
            
#             axes.extend([ax0, ax1, ax2, ax3])
            
#             # Plot EISCAT UHF
#             ne_EISCAT = ax0.pcolormesh(r_time, r_h, ne_eis, shading='auto', cmap='turbo', norm=LogNorm(vmin=MIN, vmax=MAX))
#             ax0.set_title(f'{date}\nEISCAT UHF', fontsize=subtit_size, fontweight='bold')
#             ax0.tick_params(labelbottom=False)
#             if i == 0:
#                 model_pcolormeshes[0] = ne_EISCAT  # Save for colorbar
            
#             # Plot KIAN-Net
#             ne_kian_plot = ax1.pcolormesh(r_time, r_h, ne_kian, shading='auto', cmap='turbo', norm=LogNorm(vmin=MIN, vmax=MAX))
#             ax1.set_title('KIAN-Net', fontsize=subtit_size, fontweight='bold')
#             ax1.tick_params(labelbottom=False)
#             if i == 0:
#                 model_pcolormeshes[1] = ne_kian_plot
            
#             # Plot Artist 4.5
#             ne_art_plot = ax2.pcolormesh(r_time, r_h, ne_art, shading='auto', cmap='turbo', norm=LogNorm(vmin=MIN, vmax=MAX))
#             ax2.set_title('Artist 4.5', fontsize=subtit_size, fontweight='bold')
#             ax2.tick_params(labelbottom=False)
#             if i == 0:
#                 model_pcolormeshes[2] = ne_art_plot
            
#             # Plot IRI
#             ne_iri_plot = ax3.pcolormesh(r_time, r_h, ne_iri, shading='auto', cmap='turbo', norm=LogNorm(vmin=MIN, vmax=MAX))
#             ax3.set_title('IRI', fontsize=subtit_size, fontweight='bold')
#             ax3.set_xlabel('Time [UT]', fontsize=xlab_size)
#             ax3.xaxis.set_major_formatter(DateFormatter('%H:%M'))
#             if i == 0:
#                 model_pcolormeshes[3] = ne_iri_plot
            
#             # Set y-labels for the first column only
#             if i == 0:
#                 ax0.set_ylabel('Altitude [km]', fontsize=ylab_size)
#                 ax1.set_ylabel('Altitude [km]', fontsize=ylab_size)
#                 ax2.set_ylabel('Altitude [km]', fontsize=ylab_size)
#                 ax3.set_ylabel('Altitude [km]', fontsize=ylab_size)
#             else:
#                 ax0.tick_params(labelleft=False)
#                 ax1.tick_params(labelleft=False)
#                 ax2.tick_params(labelleft=False)
#                 ax3.tick_params(labelleft=False)
        
#         # Add colorbars for each row
#         for row in range(4):
#             cax = fig.add_subplot(gs[row, 3])
#             cbar = fig.colorbar(model_pcolormeshes[row], cax=cax, orientation='vertical')
#             cbar.set_label('$n_e$ [n/m$^3$]', fontsize=cbar_size)
        
#         # Rotate x-axis labels
#         for ax in axes:
#             plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='center')
        
#         plt.show()
    
