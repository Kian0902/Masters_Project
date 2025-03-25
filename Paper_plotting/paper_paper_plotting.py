"""
Created on Mon Jan 27 16:23:19 2025

@author: Kian Sartipzadeh
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.dates import DateFormatter
from matplotlib.patches import Patch
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
from matplotlib.widgets import Cursor
from datetime import datetime
from scipy.stats import linregress, gaussian_kde, pearsonr
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
    
    
    def plot_compare_all_peak_densities(self, show_marginal_kde=False):
        
        # Merge data dictionaries for all regions
        X_eis = merge_nested_peak_dict(self.X_EIS)['All']
        X_kian = merge_nested_peak_dict(self.X_KIAN)['All']
        X_ion = merge_nested_peak_dict(self.X_ION)['All']
        X_geo = merge_nested_peak_dict(self.X_GEO)['All']
        X_art = merge_nested_peak_dict(self.X_ART)['All']
        X_ech = merge_nested_peak_dict(self.X_ECH)['All']
    
        # Prepare peak density data (all in log10(ne))
        eis_param_peak = np.log10(X_eis["r_param_peak"])
        kian_param_peak = X_kian["r_param_peak"]
        ion_param_peak = X_ion["r_param_peak"]
        geo_param_peak = X_geo["r_param_peak"]
        art_param_peak = np.log10(X_art["r_param_peak"])
        ech_param_peak = np.log10(X_ech["r_param_peak"])
    
        # Define models with names, data, and colors
        models = [
            {"name": "KIAN-Net", "data": kian_param_peak, "color": "C1"},
            {"name": "Artist 5.0", "data": art_param_peak, "color": "C2"},
            {"name": "E-CHAIM", "data": ech_param_peak, "color": "C3"},
            {"name": "Geo-DMLP", "data": geo_param_peak, "color": "C4"},
            {"name": "Iono-CNN", "data": ion_param_peak, "color": "C5"},
        ]
    
        # Convert time to datetime for title
        r_time = from_array_to_datetime(X_eis["r_time"])
        # date_str = r_time[0].strftime('%b-%Y')
        
        
        fig = plt.figure(figsize=(14, 7))
    
        # Set up GridSpec based on whether marginal KDEs are requested
        if show_marginal_kde:
            gs = GridSpec(2, 5, height_ratios=[0.2, 1], width_ratios=[1, 0.2, 0.1, 1, 0.2], hspace=0, wspace=0)
            with sns.axes_style("dark"):
                ax_e = fig.add_subplot(gs[1, 0])
                ax_f = fig.add_subplot(gs[1, 3])
                
            ax_e_top = fig.add_subplot(gs[0, 0], sharex=ax_e)
            ax_f_top = fig.add_subplot(gs[0, 3], sharex=ax_f)
            ax_e_top.set_axis_off()
            ax_f_top.set_axis_off()
            
            # Spacing between e/f-reg plots
            ax_spacing = fig.add_subplot(gs[:, 3])
            ax_spacing.set_axis_off()
            
            
            ax_e_right = fig.add_subplot(gs[1, 1], sharey=ax_e)
            ax_f_right = fig.add_subplot(gs[1, 4], sharey=ax_f)
            ax_e_right.set_axis_off()
            ax_f_right.set_axis_off()
        else:
            gs = GridSpec(1, 2, width_ratios=[1, 1], wspace=0.1)
            ax_e = fig.add_subplot(gs[0, 0])
            ax_f = fig.add_subplot(gs[0, 1])
    
        fig.suptitle('Peak Elecron densities\n(EISCAT UHF vs Model)', fontsize=20, y=0.95)
    
        def plot_region(ax, region, Min, Max, title, ax_top=None, ax_right=None):
            # Plot the diagonal line (perfect agreement)
            diagonal_line = np.array([Min, Max])
            ax.plot([Min, Max], diagonal_line, color="C0", label="EISCAT", 
                    linewidth=3, zorder=1)
    
            # Loop over models to plot scatter, regression lines, metrics, and optional KDEs
            for idx, model in enumerate(models):
                # Extract valid data for the region
                model_data = model["data"][region, :]
                valid = ~np.isnan(model_data) & ~np.isnan(eis_param_peak[region, :])
                x = model_data[valid]  # Predicted values (model)
                y = eis_param_peak[region, valid]  # Observed values (EISCAT)
    
                # Plot scatter points
                ax.scatter(x, y, s=40, color=model["color"], label=model["name"], 
                           edgecolors="black")
                
                # Subplots titles
                ax.text(0.3, 0.9, s=f'{title} Peaks', color='black',
                        fontsize=15, transform=ax.transAxes)
                
                # Check if there are any valid predictions
                if valid.any():
                    
                    r_corr, _ = pearsonr(x, y)
                    r2 = self.compute_r2_score(y, x)
                    rmse = np.sqrt(np.mean((y - x) ** 2))
                    
                    # Annotate R² and RMSE with model name
                    ax.text(0.6, 0.2 - (0.04 * idx), 
                            s=rf"$\mathbf{{({r2:.2f},\ {rmse:.2f},\ {r_corr:.2f})}}$", 
                            color=model["color"], fontsize=10, transform=ax.transAxes)
                    
                    # Plot marginal KDEs if axes are provided
                    if ax_top is not None and ax_right is not None:
                        # KDE for x (predicted densities)
                        kde_x = gaussian_kde(x)
                        x_grid = np.linspace(Min, Max, 100)
                        kde_x_values = kde_x(x_grid)
                        ax_top.plot(x_grid, kde_x_values, color=model["color"])
    
                        # KDE for y (observed densities)
                        kde_y = gaussian_kde(y)
                        y_grid = np.linspace(Min, Max, 100)
                        kde_y_values = kde_y(y_grid)
                        ax_right.plot(kde_y_values, y_grid, color=model["color"])
                else:
                    # If no valid data, display "N/A"
                    ax.text(0.6, 0.2 - (0.04 * idx), 
                            s=rf"$\mathbf{{(N/A,\ N/A,\ N/A)}}$", 
                            color=model["color"], fontsize=10, transform=ax.transAxes)
                
                if idx == 4:
                    ax.text(0.6, 0.2 - (0.04 * (-1)), 
                            s=rf"$\mathbf{{(R^{2},\ RMSE, \, r)}}$", 
                            color="black", fontsize=10, transform=ax.transAxes)
                    
            # Set plot aesthetics
            ax.set_xlim(Min, Max)
            ax.set_ylim(Min, Max)
            ax.grid(True)
            
            # Configure marginal axes if they exist
            if ax_top is not None and ax_right is not None:
                # Set limits to start at 0, with automatic upper bounds
                ax_top.set_ylim([0, None])
                ax_right.set_xlim([0, None])
                
                # Add density labels
                ax_top.set_ylabel("Density", fontsize=10)
                ax_right.set_xlabel("Density", fontsize=10)
                
                # Hide ticks that overlap with main plot
                ax_top.tick_params(labelbottom=False)
                ax_right.tick_params(labelleft=False)
                
                # Remove unnecessary spines for a cleaner look
                ax_top.spines['bottom'].set_visible(False)
                ax_top.spines['top'].set_visible(False)
                ax_top.spines['right'].set_visible(False)
                ax_right.spines['left'].set_visible(False)
                ax_right.spines['top'].set_visible(False)
                ax_right.spines['right'].set_visible(False)
    
        # Plot E-region and F-region with appropriate parameters
        if show_marginal_kde:
            plot_region(ax_e, region=0, Min=8.5, Max=12.5, title='(a) E-region', ax_top=ax_e_top, ax_right=ax_e_right)
            plot_region(ax_f, region=1, Min=9.5, Max=12.5, title='(b) F-region', ax_top=ax_f_top, ax_right=ax_f_right)
        else:
            plot_region(ax_e, region=0, Min=8.5, Max=12, title='E-region')
            plot_region(ax_f, region=1, Min=9.5, Max=12.5, title='F-region')
    
        # Set axis labels
        ax_e.set_xlabel(r'Model  $log_{10}\,(n_e)$', fontsize=15)
        ax_e.set_ylabel(r'EISCAT UHF  $log_{10}\,(n_e)$', fontsize=15)
        ax_f.set_xlabel(r'Model  $log_{10}\,(n_e)$', fontsize=15)
    
        # Add legend to F-region plot with white background
        ax_e.legend(fontsize=9, facecolor='white', loc='upper left')
        ax_f.legend(fontsize=9, facecolor='white', loc='upper left')
    
        plt.show()
    

    
    def plot_compare_all_peak_altitudes(self):
        # Merge data dictionaries for all regions
        X_eis = merge_nested_peak_dict(self.X_EIS)['All']
        X_kian = merge_nested_peak_dict(self.X_KIAN)['All']
        X_ion = merge_nested_peak_dict(self.X_ION)['All']
        X_geo = merge_nested_peak_dict(self.X_GEO)['All']
        X_art = merge_nested_peak_dict(self.X_ART)['All']
        X_ech = merge_nested_peak_dict(self.X_ECH)['All']
        
        # Prepare peak height data
        eis_h_peak = X_eis["r_h_peak"]
        kian_h_peak = X_kian["r_h_peak"]
        ion_h_peak = X_ion["r_h_peak"]
        geo_h_peak = X_geo["r_h_peak"]
        art_h_peak = X_art["r_h_peak"]
        ech_h_peak = X_ech["r_h_peak"]
        
        # Define models with names, data, and colors
        models = [
            {"name": "KIAN-Net", "data": kian_h_peak, "color": "C1"},
            {"name": "Iono-CNN", "data": ion_h_peak, "color": "C5"},
            {"name": "Geo-DMLP", "data": geo_h_peak, "color": "C4"},
            {"name": "Artist 5.0", "data": art_h_peak, "color": "C2"},
            {"name": "E-CHAIM", "data": ech_h_peak, "color": "C3"},
        ]
        
        # Set up the figure with two subplots
        fig = plt.figure(figsize=(15, 7))
        gs = GridSpec(1, 2, width_ratios=[1, 1], wspace=0.2)
        fig.suptitle('Peak Elecron Density Altitudes', fontsize=20, y=1)
        with sns.axes_style("dark"):
            ax_violin = fig.add_subplot(gs[0, 0])
            ax_height = fig.add_subplot(gs[0, 1])
           
        # --- Original Violin Plot (Unchanged) ---
        for region in [0, 1]:
            # Collect data for all models and observed EISCAT data
            data_list = []
            
            # Add observed EISCAT heights as a reference
            eis_data = eis_h_peak[region, :]
            valid_eis = ~np.isnan(eis_data)
            df_obs = pd.DataFrame({'model': 'Observed', 'height': eis_data[valid_eis]})
            data_list.append(df_obs)
         
            # Add predicted heights for each model
            for model in models:
                model_data = model["data"][region, :]
                valid = ~np.isnan(model_data)
                df = pd.DataFrame({'model': model["name"], 'height': model_data[valid]})
                data_list.append(df)
                    
            # Combine all data into a single DataFrame
            all_data = pd.concat(data_list)
            
            # Define color palette: 'Observed' in gray, models with their specified colors
            palette = {'Observed': 'gray', **{m["name"]: m["color"] for m in models}}
            
            # Create a violin plot
            sns.violinplot(
                x='model',
                y='height',
                data=all_data,
                ax=ax_violin,
                hue='model',
                palette=palette,
                order=['Observed'] + [m["name"] for m in models],
                # inner='quartile',
                legend=False
            )
            
            # Customize the plot
            ax_violin.set_title('(a) Distribution Violin Plot', fontsize=17)
            ax_violin.set_xlabel('Model', fontsize=15)
            ax_violin.set_ylabel('Altitudes [km]', fontsize=15)
            ax_violin.set_ylim(80, 350)
            ax_violin.grid(True)
        
        # --- New RMSE Bar Plot for ax_height ---
        rmse_data = []
        for model in models:
            for region, region_name in [(0, 'E'), (1, 'F')]:
                eis_data = eis_h_peak[region, :]
                model_data = model["data"][region, :]
                valid = ~np.isnan(eis_data) & ~np.isnan(model_data)
                if valid.any():
                    rmse = np.sqrt(np.mean((eis_data[valid] - model_data[valid])**2))
                    rmse_data.append({'Model': model["name"], 'Region': region_name, 'RMSE': rmse})
        rmse_df = pd.DataFrame(rmse_data)
        sns.barplot(
            x='Model', y='RMSE', hue='Region', data=rmse_df, ax=ax_height,
            palette=['C0', 'C1'], edgecolor='k'
        )
        
        
        # Add RMSE values above each bar
        for p in ax_height.patches:
            height = p.get_height()
            if not np.isnan(height) and height > 0.01:  # Only label non-NaN bars
                ax_height.text(
                    p.get_x() + p.get_width() / 2.,  # Center of the bar
                    height + 0.5,  # Slightly above the bar
                    f'{height:.2f}',  # Format to 2 decimal places
                    ha='center', va='bottom',
                    fontsize=11
                )
        ax_height.set_title('(b) RMSE Bar Plot', fontsize=17)
        ax_height.set_xlabel('Model', fontsize=15)
        ax_height.set_ylabel('RMSE [km]', fontsize=15)
        ax_height.legend(title='Region')
        ax_height.yaxis.grid(True)  # Add horizontal grid lines
        ax_height.xaxis.grid(False)  # Ensure no vertical grid lines
        
        plt.setp(ax_violin.xaxis.get_majorticklabels(), rotation=45, ha='center')
        plt.setp(ax_height.xaxis.get_majorticklabels(), rotation=45, ha='center')
        
        # Adjust layout and display
        plt.tight_layout()
        plt.show()
        
    # def plot_compare_all_peak_altitudes(self):
    #     # Merge data dictionaries for all regions
    #     X_eis = merge_nested_peak_dict(self.X_EIS)['All']
    #     X_kian = merge_nested_peak_dict(self.X_KIAN)['All']
    #     X_ion = merge_nested_peak_dict(self.X_ION)['All']
    #     X_geo = merge_nested_peak_dict(self.X_GEO)['All']
    #     X_art = merge_nested_peak_dict(self.X_ART)['All']
    #     X_ech = merge_nested_peak_dict(self.X_ECH)['All']
        
    #     # Prepare peak height data
    #     eis_h_peak = X_eis["r_h_peak"]
    #     kian_h_peak = X_kian["r_h_peak"]
    #     ion_h_peak = X_ion["r_h_peak"]
    #     geo_h_peak = X_geo["r_h_peak"]
    #     art_h_peak = X_art["r_h_peak"]
    #     ech_h_peak = X_ech["r_h_peak"]
        
    #     # Define models with names, data, and colors
    #     models = [
    #         {"name": "KIAN-Net", "data": kian_h_peak, "color": "C1"},
    #         {"name": "Geo-DMLP", "data": geo_h_peak, "color": "C4"},
    #         {"name": "Iono-CNN", "data": ion_h_peak, "color": "C5"},
    #         {"name": "Artist 5.0", "data": art_h_peak, "color": "C2"},
    #         {"name": "E-CHAIM", "data": ech_h_peak, "color": "C3"},
           
    #         ]
    
    #     # Set up the figure with two subplots
    #     fig = plt.figure(figsize=(14, 7))
    #     gs = GridSpec(1, 2, width_ratios=[1, 1])
    #     with sns.axes_style("dark"):
    #         ax_violin = fig.add_subplot(gs[0, 0])
    #         ax_height = fig.add_subplot(gs[0, 1], sharey=ax_violin)
       
       

    #     for region in [0, 1]:
    #         # Collect data for all models and observed EISCAT data
    #         data_list = []
            
    #         # Add observed EISCAT heights as a reference
    #         eis_data = eis_h_peak[region, :]
    #         valid_eis = ~np.isnan(eis_data)
    #         df_obs = pd.DataFrame({'model': 'Observed', 'height': eis_data[valid_eis]})
    #         data_list.append(df_obs)
     
    #         # Add predicted heights for each model
    #         for model in models:
    #             model_data = model["data"][region, :]
    #             valid = ~np.isnan(model_data)
    #             df = pd.DataFrame({'model': model["name"], 'height': model_data[valid]})
    #             data_list.append(df)
                
    #         # Combine all data into a single DataFrame
    #         all_data = pd.concat(data_list)
            
    #         # Define color palette: 'Observed' in gray, models with their specified colors
    #         palette = {'Observed': 'gray', **{m["name"]: m["color"] for m in models}}
            
    #         # Create a violin plot
    #         sns.violinplot(
    #             x='model',
    #             y='height',
    #             data=all_data,
    #             ax=ax_violin,
    #             hue='model',
    #             palette=palette,
    #             order=['Observed'] + [m["name"] for m in models],
    #             # inner='quartile',
    #             legend=False
    #         )
            
    #         # Customize the plot
    #         ax_violin.set_title('(a) Violin Plot', fontsize=15)
    #         ax_violin.set_xlabel('Model', fontsize=13)
    #         ax_violin.set_ylabel('Altitudes (km)', fontsize=13)
    #         ax_violin.set_ylim(80, 350)
    #         ax_violin.grid(True)
                
            
            
    #     # Adjust layout and display
    #     plt.tight_layout()
    #     plt.show()
        
        
    
