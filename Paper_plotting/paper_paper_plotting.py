"""
Created on Mon Jan 27 16:23:19 2025

@author: Kian Sartipzadeh
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import trapezoid as trapz
from matplotlib.colors import LogNorm, ListedColormap, to_rgba
from matplotlib.dates import DateFormatter
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
from matplotlib.widgets import Cursor
from datetime import datetime
from scipy.stats import linregress, gaussian_kde, pearsonr, spearmanr
from paper_utils import from_array_to_datetime, merge_nested_dict, merge_nested_pred_dict, merge_nested_peak_dict, merge_nested_peak_pred_dict, merge_nested_peak_dict, get_altitude_r2_score_nans, get_measurements_r2_score_nans, inspect_dict
from sklearn.metrics import r2_score, mean_absolute_percentage_error
from sklearn import preprocessing as pre
import seaborn as sns
# For all plots: sns.set(style="dark", context=None, palette=None)
# For single plot: with sns.axes_style("dark"):

import dcor


class PaperPlotter:
    def __init__(self, X_EIS, X_KIAN, X_ION, X_GEO, X_ART, X_ECH):
        self.X_EIS = X_EIS
        self.X_KIAN = X_KIAN
        self.X_ION = X_ION
        self.X_GEO = X_GEO
        self.X_ART = X_ART
        self.X_ECH = X_ECH
        # self.selected_indices = []

        
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
    
    def relative_error(self, X_true, X_pred):
        err =  (X_pred - X_true)/X_true
        return err
    
    def absolute_relative_error(self, X_true, X_pred):
        err =  abs(X_pred - X_true)/X_true
        return err
    
    
    def get_best_model(self, stacked_error):
        # Identify points where all models have NaN
        all_nan = np.all(np.isnan(stacked_error), axis=0)
        err_stack_inf = np.where(np.isnan(stacked_error), 100, stacked_error)
        best_model = np.argmin(err_stack_inf, axis=0)
        best_model_masked = np.ma.array(best_model, mask=all_nan)
        return best_model_masked
    
    
    def plot_compare_error_all(self, model_colors=None):
        # Default colors if none are provided
        if model_colors is None:
            model_colors = ['C1', 'C4', 'C5', 'C2', 'C3']
    
        # Check that exactly 5 colors are provided
        if len(model_colors) != 5:
            raise ValueError("You must provide exactly 5 colors, one for each model.")
        
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
        ne_eis  = np.log10(x_eis["r_param"])
        ne_kian = x_kian["r_param"]
        ne_ion  = x_ion["r_param"]
        ne_geo  = x_geo["r_param"]
        ne_art  = np.log10(x_art["r_param"])
        ne_ech  = np.log10(x_ech["r_param"])
        
        # Fill nans with 0
        nan_ind = np.isnan(ne_art)
        ne_art = np.where(nan_ind == True, 0, ne_art)
        
        date_str = r_time[0].strftime('%b-%Y')
        
        err_kian = self.relative_error(ne_eis, ne_kian)
        err_ion = self.relative_error(ne_eis, ne_ion)
        err_geo = self.relative_error(ne_eis, ne_geo)
        err_art = self.relative_error(ne_eis, ne_art)
        err_ech = self.relative_error(ne_eis, ne_ech)
        
        # Calculate absolute logarithmic errors
        abs_err_kian = self.absolute_relative_error(ne_eis, ne_kian)
        abs_err_ion = self.absolute_relative_error(ne_eis, ne_ion)
        abs_err_geo = self.absolute_relative_error(ne_eis, ne_geo)
        abs_err_art = self.absolute_relative_error(ne_eis, ne_art)
        abs_err_ech = self.absolute_relative_error(ne_eis, ne_ech)
    
        # Stack errors into a (5, N, M) array
        err_stack = np.stack([abs_err_kian, abs_err_ion, abs_err_geo, abs_err_art, abs_err_ech], axis=0)
        
        best_model_masked = self.get_best_model(err_stack)
        best_model_reversed = 4 - best_model_masked
        
        # Create a custom colormap with your colors
        CMAP1 = ListedColormap(model_colors[::-1])
        
        # ___________ Defining axes ___________
        fig = plt.figure(figsize=(12, 12))
        fig.suptitle(f'Comparison Between Prediction Models and Ground Truth\nDate: {date_str}', fontsize=17, y=0.97)
        
        gs = GridSpec(6, 2, width_ratios=[1, 0.015], wspace=0.1, hspace=0.35)
        ax0 = fig.add_subplot(gs[0, 0])
        ax1 = fig.add_subplot(gs[1, 0], sharex=ax0)
        ax2 = fig.add_subplot(gs[2, 0], sharex=ax0)
        ax3 = fig.add_subplot(gs[3, 0], sharex=ax0)
        ax4 = fig.add_subplot(gs[4, 0], sharex=ax0)
        ax5 = fig.add_subplot(gs[5, 0], sharex=ax0)
        cax1 = fig.add_subplot(gs[0, 1])
        cax2 = fig.add_subplot(gs[1:, 1])
        
        # ___________ Creating Plots ___________
        CMAP, MIN, MAX = "RdBu_r", -0.15, 0.15
        
        # Use best_model_reversed instead of best_model_masked
        err1 = ax0.pcolormesh(r_time, r_h, best_model_reversed, cmap=CMAP1, shading='gouraud')
        err2 = ax1.pcolormesh(r_time, r_h, err_kian, shading='gouraud', cmap=CMAP, vmin=MIN, vmax=MAX)
        ax2.pcolormesh(r_time, r_h, err_ion, shading='gouraud', cmap=CMAP, vmin=MIN, vmax=MAX)
        ax3.pcolormesh(r_time, r_h, err_geo, shading='gouraud', cmap=CMAP, vmin=MIN, vmax=MAX)
        ax4.pcolormesh(r_time, r_h, err_art, shading='gouraud', cmap=CMAP, vmin=MIN, vmax=MAX)
        ax5.pcolormesh(r_time, r_h, err_ech, shading='gouraud', cmap=CMAP, vmin=MIN, vmax=MAX)
        
        # Set titles and labels
        ax0.set_title('(a) Best Model (Lowest Relative Absolute Error)', fontsize=16)
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
        cbar1 = fig.colorbar(err1, cax=cax1, ticks=np.arange(5))
        cbar1.ax.set_yticklabels(['E-Chaim', 'Artist 5.0', 'Geo-DMLP', 'Iono-CNN', 'KIAN-Net'])
        # cbar1.set_label('Best Model', fontsize=13)
        
        cbar2 = fig.colorbar(err2, cax=cax2, orientation='vertical')
        cbar2.set_label('Relative Error log10(m$^{-3}$)', fontsize=13)
        
        plt.show()
        
        
    
        
        
    def plot_best_model(self, model_colors=None):
    
        # Default colors if none are provided
        if model_colors is None:
            model_colors = ['C1', 'C4', 'C5', 'C2', 'C3']
    
        # Check that exactly 5 colors are provided
        if len(model_colors) != 5:
            raise ValueError("You must provide exactly 5 colors, one for each model.")
    
        # Merge data from all models
        x_eis = merge_nested_dict(self.X_EIS)['All']
        x_kian = merge_nested_pred_dict(self.X_KIAN)['All']
        x_ion = merge_nested_pred_dict(self.X_ION)['All']
        x_geo = merge_nested_pred_dict(self.X_GEO)['All']
        x_art = merge_nested_dict(self.X_ART)['All']
        x_ech = merge_nested_dict(self.X_ECH)['All']
    
        # Extract time and altitude
        r_time = from_array_to_datetime(x_eis['r_time'])
        r_h = x_eis['r_h'].flatten()
    
        # Compute electron densities in log10 space
        ne_eis = np.log10(x_eis["r_param"])  # EISCAT UHF
        ne_kian = x_kian["r_param"]          # KIAN-Net (already in log10)
        ne_ion = x_ion["r_param"]            # Iono-CNN (already in log10)
        ne_geo = x_geo["r_param"]            # Geo-DMLP (already in log10)
        ne_art = np.log10(x_art["r_param"])  # Artist 5.0
        ne_ech = np.log10(x_ech["r_param"])  # E-Chaim
    
        # Calculate absolute logarithmic errors
        err_kian = self.absolute_relative_error(ne_eis, ne_kian)
        err_ion = self.absolute_relative_error(ne_eis, ne_ion)
        err_geo = self.absolute_relative_error(ne_eis, ne_geo)
        err_art = self.absolute_relative_error(ne_eis, ne_art)
        err_ech = self.absolute_relative_error(ne_eis, ne_ech)
    
        # Stack errors into a (5, N, M) array
        err_stack = np.stack([err_kian, err_ion, err_geo, err_art, err_ech], axis=0)
        
        best_model_masked = self.get_best_model(err_stack)
        
        # Create a custom colormap with your colors
        cmap = ListedColormap(model_colors)
    
        # Create the plot
        fig, ax = plt.subplots(figsize=(14, 6))
        pcm = ax.pcolormesh(r_time, r_h, best_model_masked, cmap=cmap, shading='gouraud')
    
        # Add a color bar with model names
        cbar = fig.colorbar(pcm, ax=ax, ticks=np.arange(5))
        cbar.ax.set_yticklabels(['KIAN-Net', 'Iono-CNN', 'Geo-DMLP', 'Artist 5.0', 'E-Chaim'])
        cbar.set_label('Best Model')
    
        # Set labels and title
        ax.set_xlabel('UT [dd hh:mm]')
        ax.set_ylabel('Altitude [km]')
        date_str = r_time[0].strftime('%b-%Y')
        ax.set_title(f'Best Prediction Model Based on Smallest |log10(Ne_pred) - log10(Ne_true)|\nDate: {date_str}')
    
        # Format the x-axis with datetime
        ax.xaxis.set_major_formatter(DateFormatter('%d %H:%M'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='center')
    
        plt.tight_layout()
        plt.show()
    
    
    
    def plot_altitude_metrics(self):
        # ___________ Getting Data ___________
        x_eis = merge_nested_dict(self.X_EIS)['All']
        x_kian = merge_nested_pred_dict(self.X_KIAN)['All']
        x_ion = merge_nested_pred_dict(self.X_ION)['All']
        x_geo = merge_nested_pred_dict(self.X_GEO)['All']
        x_art = merge_nested_dict(self.X_ART)['All']
        x_ech = merge_nested_dict(self.X_ECH)['All']
        
        r_h = x_eis['r_h'].flatten()  # Altitude points (e.g., 27 altitudes)
        ne_eis = np.log10(x_eis["r_param"])  # True values in log10 scale
        ne_kian = x_kian["r_param"]          # Predicted values
        ne_ion = x_ion["r_param"]
        ne_geo = x_geo["r_param"]
        ne_art = np.log10(x_art["r_param"])
        ne_ech = np.log10(x_ech["r_param"])
        
        # Define models and their predictions
        models = {
            'KIAN-Net': ne_kian,
            'Iono-CNN': ne_ion,
            'Geo-DMLP': ne_geo,
            'Artist 5.0': ne_art,
            'E-Chaim': ne_ech
        }
        
        # Define plotting properties for each model
        plot_props = {
            'KIAN-Net': {'color': 'C1', 'lw': 3, 'zorder': 5, 'ls': '-'},
            'Iono-CNN': {'color': 'C4', 'lw': 3.5, 'zorder': 4, 'ls': (2, (4.5, 1))},
            'Geo-DMLP': {'color': 'C5', 'lw': 4, 'zorder': 1, 'ls': (0, (1, 1))},
            'Artist 5.0': {'color': 'C2', 'lw': 3, 'zorder': 3, 'ls': (1, (3, 1, 1, 1))},
            'E-Chaim': {'color': 'C3', 'lw': 2, 'zorder': 2, 'ls': '-'}
        }
        
        # Identify valid measurements where all EISCAT UHF altitudes are non-NaN
        valid_measurements = [m for m in range(ne_eis.shape[1]) if np.all(~np.isnan(ne_eis[:, m]))]
        if not valid_measurements:
            print("No measurements with fully valid EISCAT UHF data found.")
            return
        
        # Initialize dictionaries to store metrics for each model and altitude
        metrics = {
            'R2': {model: np.full(len(r_h), np.nan) for model in models},
            'RMSE': {model: np.full(len(r_h), np.nan) for model in models},
            'Pearson': {model: np.full(len(r_h), np.nan) for model in models},
            'Distance Correlation': {model: np.full(len(r_h), np.nan) for model in models}
        }
        
        # --- Reused Metric Calculation Functions ---
        def normalize(X):
            return pre.MinMaxScaler().fit_transform(X.reshape(-1, 1)).flatten()
        
        def calculate_r2_tot(ne_true_alt, ne_pred_alt):
            mask = ~np.isnan(ne_pred_alt)
            if np.sum(mask) < 2:  # Need at least 2 points for R²
                return -1
            true_valid = normalize(ne_true_alt[mask])
            pred_valid = normalize(ne_pred_alt[mask])
            return r2_score(true_valid, pred_valid)
        
        def calculate_rmse_tot(ne_true_alt, ne_pred_alt):
            mask = ~np.isnan(ne_pred_alt)
            if np.sum(mask) < 1:  # Need at least 1 point for RMSE
                return 2
            true_valid = normalize(ne_true_alt[mask])
            pred_valid = normalize(ne_pred_alt[mask])
            mse = np.mean((true_valid - pred_valid) ** 2)
            return np.sqrt(mse)
        
        def calculate_pearson_r(ne_true_alt, ne_pred_alt):
            mask = ~np.isnan(ne_pred_alt)
            if np.sum(mask) < 2:  # Need at least 2 points for correlation
                return 0
            true_valid = normalize(ne_true_alt[mask])
            pred_valid = normalize(ne_pred_alt[mask])
            r, _ = pearsonr(true_valid, pred_valid)
            return r
        
        def calculate_distance_cor(ne_true_alt, ne_pred_alt):
            mask = ~np.isnan(ne_pred_alt)
            if np.sum(mask) < 2:  # Need at least 2 points for correlation
                return 0
            true_valid = normalize(ne_true_alt[mask].astype(np.float64))
            pred_valid = normalize(ne_pred_alt[mask].astype(np.float64))
            return dcor.distance_correlation(true_valid, pred_valid)
        
        # --- Compute Metrics Per Altitude ---
        def compute_metrics_for_altitude(alt_idx):
            true_values = ne_eis[alt_idx, valid_measurements]  # True values at valid timestamps
            for model_name, pred_values in models.items():
                pred = pred_values[alt_idx, valid_measurements]  # Predicted values at valid timestamps
                metrics['R2'][model_name][alt_idx] = calculate_r2_tot(true_values, pred)
                metrics['RMSE'][model_name][alt_idx] = calculate_rmse_tot(true_values, pred)
                metrics['Pearson'][model_name][alt_idx] = calculate_pearson_r(true_values, pred)
                metrics['Distance Correlation'][model_name][alt_idx] = calculate_distance_cor(true_values, pred)
        
        # Loop over each altitude and compute metrics
        for alt_idx in range(len(r_h)):
            compute_metrics_for_altitude(alt_idx)
        
        # --- Plotting ---
        with sns.axes_style("dark"):
            fig, axes = plt.subplots(2, 2, figsize=(8, 9), sharey=True)
        axes = axes.flatten()
        metric_names = ['R2', 'RMSE', 'Pearson', 'Distance Correlation']
        
        for i, metric_name in enumerate(metric_names):
            ax = axes[i]
            for model_name in models:
                if model_name == "E-Chaim":
                    m, ms = "o", 5
                else:
                    m, ms = None, None
                props = plot_props[model_name]
                ax.plot(metrics[metric_name][model_name], r_h, label=model_name,
                        color=props['color'], lw=props['lw'], zorder=props['zorder'], ls=props['ls'], marker=m, markersize=ms)
            ax.set_title(metric_name, fontsize=15)
            ax.set_xlabel(metric_name)
            if i % 2 == 0:  # Left column
                ax.set_ylabel('Altitude [km]')
            ax.legend()
            ax.grid(True)
        
        plt.tight_layout()
        plt.show()
        
        
                
    def plot_kde_altitude(self, altitude, return_metrics=False, show_plot=False, print_metrics=False):
        """
        Plot KDE comparison of electron density models at specified altitude and compute
        Bhattacharyya coefficient, JS-Divergence, and Wasserstein distance for each model vs EISCAT.
        Metrics are calculated only over the valid EISCAT PDF range.
        """
        # Merge data from all models and EISCAT
        x_eis = merge_nested_dict(self.X_EIS)['All']
        x_kian = merge_nested_pred_dict(self.X_KIAN)['All']
        x_ion = merge_nested_pred_dict(self.X_ION)['All']
        x_geo = merge_nested_pred_dict(self.X_GEO)['All']
        x_art = merge_nested_dict(self.X_ART)['All']
        x_ech = merge_nested_dict(self.X_ECH)['All']
        
        # Get altitude array and find the closest altitude index
        r_h = x_eis['r_h'].flatten()
        alt_idx = np.argmin(np.abs(r_h - altitude))
        closest_alt = r_h[alt_idx]
        
        # Extract EISCAT data and identify valid time indices
        ne_eis = x_eis['r_param'][alt_idx, :]
        data_eis = np.log10(ne_eis)
        valid_mask = np.isfinite(data_eis)
        
        if not np.any(valid_mask):
            print(f"No valid EISCAT data at {closest_alt} km. KDE not possible.")
            return
        
        # Prepare datasets using only valid time indices
        datasets = []
        datasets.append(('EISCAT UHF', data_eis[valid_mask]))
        
        # Define model processing parameters
        model_params = [
            ('KIAN-Net', x_kian, True, False),
            ('Artist 5.0', x_art, False, True),
            ('E-Chaim', x_ech, False, False),
            ('Iono-CNN', x_ion, True, False),
            ('Geo-DMLP', x_geo, True, False),
        ]
        
        for name, data, is_log, is_artist in model_params:
            raw = data['r_param'][alt_idx, :][valid_mask]
            if is_artist:
                raw = np.where(np.isnan(raw), 1e9, raw)
                processed = np.log10(raw)
            else:
                processed = raw if is_log else np.log10(raw)
            datasets.append((name, processed))
        
        # Predefined visualization styles
        style_config = {
            'EISCAT UHF': {'color': 'C0', 'lw': 3, 'zorder': 6, 'ls': '-'},
            'KIAN-Net': {'color': 'C1', 'lw': 3, 'zorder': 5, 'ls': '-'},
            'Artist 5.0': {'color': 'C2', 'lw': 3, 'zorder': 3, 'ls': '-'},
            'E-Chaim': {'color': 'C3', 'lw': 3, 'zorder': 2, 'ls': '-'},
            'Iono-CNN': {'color': 'C4', 'lw': 3, 'zorder': 4, 'ls': '-'},
            'Geo-DMLP': {'color': 'C5', 'lw': 3, 'zorder': 1, 'ls': '-'},
        }
    
        # Create figure and axis
        if show_plot:
            fig = plt.figure(figsize=(12, 7))
            with sns.axes_style("dark"):
                ax = fig.add_subplot(GridSpec(1, 1)[0])
    
        # Determine x-axis grid based ONLY on EISCAT data
        eis_data = datasets[0][1]
        clean_eis = eis_data[np.isfinite(eis_data)]
        x_min = np.min(clean_eis) - 0.5  # Extend slightly for KDE continuity
        x_max = np.max(clean_eis) + 0.5
        x_grid_global = np.linspace(x_min, x_max, 1000)
        dx = (x_max - x_min) / (len(x_grid_global) - 1)
    
        # Calculate and plot KDEs, storing results for metrics
        kde_results = []
        for label, data in datasets:
            clean = data[np.isfinite(data)]
            if len(clean) < 2:
                print(f"Skipping {label} - insufficient data")
                kde_results.append((label, None))
                continue
            
            kde = gaussian_kde(clean, bw_method=0.2)
            y = kde(x_grid_global)  # Evaluate on EISCAT's grid
            kde_results.append((label, y))
            
            # Plotting
            style = style_config[label]
            if show_plot:
                ax.plot(x_grid_global, y, label=label, 
                        color=style['color'],
                        linewidth=style['lw'],
                        linestyle=style['ls'],
                        zorder=style['zorder'],
                        alpha=0.9)
        
        # Configure final plot appearance
        if show_plot:
            ax.set_xlim(x_min, x_max)
            ax.set_xlabel(r'$\log_{10}(N_e)$ [m$^{-3}$]', fontsize=12)
            ax.set_ylabel('Probability Density', fontsize=12)
            ax.set_title(f'Electron Density Distribution at {closest_alt:.1f} km', 
                        fontsize=14, pad=15)
            ax.grid(True, alpha=1)
            ax.legend(frameon=True, framealpha=0.9, 
                     facecolor='white', loc='upper left',
                     fontsize=10, handlelength=2.5)
            plt.show()
    
        # Compute metrics between each model and EISCAT
        eis_label, eis_kde = kde_results[0]
        if eis_kde is None:
            print("Error: EISCAT KDE not available for metrics calculation.")
            return
    
        metrics_dict = {}  # Initialize dictionary for metric storage
        epsilon = 1e-10  # To avoid division by zero in JS divergence
        for model_label, model_kde in kde_results[1:]:
            if model_kde is None:
                print(f"Skipping metrics for {model_label} (insufficient data)")
                continue
            
            # Bhattacharyya Coefficient
            bc = np.sum(np.sqrt(eis_kde * model_kde)) * dx
            
            # JS Divergence (symmetrized measure)
            m = 0.5 * (eis_kde + model_kde)
            kl_p = np.sum(eis_kde * np.log((eis_kde + epsilon) / (m + epsilon))) * dx
            kl_q = np.sum(model_kde * np.log((model_kde + epsilon) / (m + epsilon))) * dx
            js = 0.5 * (kl_p + kl_q)
            
            # Wasserstein Distance
            cdf_eis = np.cumsum(eis_kde) * dx
            cdf_model = np.cumsum(model_kde) * dx
            wd = np.sum(np.abs(cdf_eis - cdf_model)) * dx
            
            # Store as numpy array in dictionary
            metrics_dict[model_label] = np.array([bc, js, wd])
        
        if print_metrics:
            print("\nModel Comparison Metrics at {:.1f} km (EISCAT Range Only):".format(closest_alt))
            print("{:<12} {:<20} {:<20} {:<20}".format("Model", "Bhattacharyya", "JS-Divergence", "Wasserstein"))
        
        # Preserve original model order
        model_order = [param[0] for param in model_params]
        for model in model_order:
            if model in metrics_dict:
                bc, js, wd = metrics_dict[model]
                if print_metrics:
                    print("{:<12} {:<20.4f} {:<20.4f} {:<20.4f}".format(model, bc, js, wd))
        
        if return_metrics:
            return metrics_dict  # Return dictionary instead of list
    
    
    def plot_metrics_vs_altitude(self, show_altitude_plots=False):
        """
        Calculate metrics for all altitudes and plot them against altitude.
        Creates three plots (Bhattacharyya, JS-Divergence, Wasserstein) with
        altitude on y-axis and metric values on x-axis.
        """
        # Get all available altitudes from EISCAT data
        x_eis = merge_nested_dict(self.X_EIS)['All']
        r_h = x_eis['r_h'].flatten()
        unique_alts = np.unique(r_h[np.isfinite(r_h)])
        
        # Initialize metric storage
        metric_names = ['Bhattacharyya', 'JS-Divergence', 'Wasserstein']
        model_names = ['KIAN-Net', 'Artist 5.0', 'E-Chaim', 'Iono-CNN', 'Geo-DMLP']
        metrics_cube = {name: {m: [] for m in metric_names} for name in model_names}
        valid_alts = []
    
        # Collect metrics for all altitudes
        for alt in unique_alts:
            # Calculate metrics for this altitude
            metrics = self.plot_kde_altitude(alt, return_metrics=True, show_plot=show_altitude_plots)
            
            if not metrics:
                continue  # Skip if no valid metrics
                
            valid_alts.append(alt)
            for model in model_names:
                if model in metrics:
                    bc, js, wd = metrics[model]
                    metrics_cube[model]['Bhattacharyya'].append(bc)
                    metrics_cube[model]['JS-Divergence'].append(js)
                    metrics_cube[model]['Wasserstein'].append(wd)
                else:
                    # Handle missing models with NaN
                    for m in metric_names:
                        metrics_cube[model][m].append(np.nan)
    
        # Convert to numpy arrays
        for model in model_names:
            for m in metric_names:
                metrics_cube[model][m] = np.array(metrics_cube[model][m])
        
        # Create figure with 3 subplots
        with sns.axes_style("dark"):
            fig, axes = plt.subplots(1, 3, figsize=(10, 8))
        colors = ['C1', 'C2', 'C3', 'C4', 'C5']
        
        for i, metric in enumerate(metric_names):
            ax = axes[i]
            for j, model in enumerate(model_names):
                ax.plot(metrics_cube[model][metric], valid_alts, 
                        color=colors[j],
                        linewidth=2.5,
                        label=model,
                        alpha=0.8)
            
            ax.set_title(metric)
            ax.set_xlabel(metric)
            ax.grid(True)
            ax.set_ylim([min(valid_alts), max(valid_alts)])
        
        # Shared y-axis configuration
        axes[0].set_ylabel('Altitude [km]')
        axes[1].legend(loc='upper right', bbox_to_anchor=(1.0, 1.0))
        plt.suptitle('Model Performance vs Altitude', y=1.02)
        plt.tight_layout()
        
        return metrics_cube
    
    
    
    # def plot_best_model(self):
    #     # Import necessary module within the method
    #     from matplotlib.colors import ListedColormap
        
    #     # Merge dictionaries to get 'All' data
    #     x_eis = merge_nested_dict(self.X_EIS)['All']
    #     x_kian = merge_nested_pred_dict(self.X_KIAN)['All']
    #     x_ion = merge_nested_pred_dict(self.X_ION)['All']
    #     x_geo = merge_nested_pred_dict(self.X_GEO)['All']
    #     x_art = merge_nested_dict(self.X_ART)['All']
    #     x_ech = merge_nested_dict(self.X_ECH)['All']
        
    #     # Extract time and altitude
    #     r_time = from_array_to_datetime(x_eis['r_time'])
    #     r_h = x_eis['r_h'].flatten()
        
    #     # Compute electron densities in log10 space
    #     ne_eis = np.log10(x_eis["r_param"])  # EISCAT UHF
    #     ne_kian = x_kian["r_param"]          # KIAN-Net (already log10)
    #     ne_ion = x_ion["r_param"]            # Iono-CNN (already log10)
    #     ne_geo = x_geo["r_param"]            # Geo-DMLP (already log10)
    #     ne_art = np.log10(x_art["r_param"])  # Artist 5.0
    #     ne_ech = np.log10(x_ech["r_param"])  # E-Chaim
        
    #     # Calculate absolute logarithmic errors
    #     err_kian = self.absolute_relative_error(ne_eis, ne_kian)
    #     err_ion = self.absolute_relative_error(ne_eis, ne_ion)
    #     err_geo = self.absolute_relative_error(ne_eis, ne_geo)
    #     err_art = self.absolute_relative_error(ne_eis, ne_art)
    #     err_ech = self.absolute_relative_error(ne_eis, ne_ech)
        
    #     # Stack errors into a (5, N, M) array
    #     err_stack = np.stack([err_kian, err_ion, err_geo, err_art, err_ech], axis=0)
        
    #     # Identify points where all models have NaN
    #     all_nan = np.all(np.isnan(err_stack), axis=0)
        
    #     # Replace NaN with infinity for comparison
    #     err_stack_inf = np.where(np.isnan(err_stack), np.inf, err_stack)
        
    #     # Find the index of the model with the smallest error (0: KIAN, 1: ION, etc.)
    #     best_model = np.argmin(err_stack_inf, axis=0)
        
    #     # Create a masked array, masking points where all models are NaN
    #     best_model_masked = np.ma.array(best_model, mask=all_nan)
        
    #     # Define a colormap with 5 distinct colors
    #     cmap = ListedColormap(plt.cm.tab10(np.arange(5)))
        
    #     # Create the plot
    #     fig, ax = plt.subplots(figsize=(10, 8))
    #     pcm = ax.pcolormesh(r_time, r_h, best_model_masked, cmap=cmap, shading='auto')
        
    #     # Add a color bar
    #     cbar = fig.colorbar(pcm, ax=ax, ticks=np.arange(5))
    #     cbar.ax.set_yticklabels(['KIAN-Net', 'Iono-CNN', 'Geo-DMLP', 'Artist 5.0', 'E-Chaim'])
    #     cbar.set_label('Best Model')
        
    #     # Set labels and title
    #     ax.set_xlabel('UT [dd hh:mm]')
    #     ax.set_ylabel('Altitude [km]')
    #     date_str = r_time[0].strftime('%b-%Y')
    #     ax.set_title(f'Best Prediction Model Based on Smallest |log10(Ne_pred) - log10(Ne_true)|\nDate: {date_str}')
        
    #     # Format the x-axis with datetime
    #     ax.xaxis.set_major_formatter(DateFormatter('%d %H:%M'))
    #     plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='center')
        
    #     plt.tight_layout()
    #     plt.show()
    
    def plot_compare_all_error(self):
        
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
        
        # inspect_dict(x_eis)
        
        # Ne-profiles
        ne_eis  = np.log10(x_eis["r_param"])
        ne_kian = x_kian["r_param"]
        ne_ion  = x_ion["r_param"]
        ne_geo  = x_geo["r_param"]
        ne_art  = np.log10(x_art["r_param"])
        ne_ech  = np.log10(x_ech["r_param"])
        
        date_str = r_time[0].strftime('%b-%Y')
        
        # Identify valid measurements where all EISCAT UHF values are non-NaN
        valid_measurements = [m for m in range(ne_eis.shape[1]) if np.all(~np.isnan(ne_eis[:, m]))]
        if not valid_measurements:
            print("No measurements with fully valid EISCAT UHF data found.")
            return
        
        
        def normalize(X):
            return pre.MinMaxScaler().fit_transform(X.reshape(-1, 1)).flatten()
        
        # Helper functions for score calculations
        def calculate_r2_tot(ne_true_m, ne_pred_m):
            mask = ~np.isnan(ne_pred_m)
            if np.sum(mask) < 2:  # Need at least 2 points for R²
                return -1
            true_valid = normalize(ne_true_m[mask])
            pred_valid = normalize(ne_pred_m[mask])
            return r2_score(true_valid, pred_valid)
    
        def calculate_rmse_tot(ne_true_m, ne_pred_m):
            mask = ~np.isnan(ne_pred_m)
            if np.sum(mask) < 1:  # Need at least 1 point for RMSE
                return 2
            true_valid = normalize(ne_true_m[mask])
            pred_valid = normalize(ne_pred_m[mask])
            mse = np.mean((true_valid - pred_valid) ** 2)
            return np.sqrt(mse)
        
        def calculate_relative_error_tot(ne_true_m, ne_pred_m):
            mask = ~np.isnan(ne_pred_m)
            if np.sum(mask) < 1:  # Need at least 1 point for RMSE
                return 0
            
            # print(ne_true_m)
            
            true_valid = ne_true_m[mask]
            pred_valid = ne_pred_m[mask]
            
            eps= 0.1
            # err =  abs(pred_valid - true_valid)/(true_valid + eps)
            err = mean_absolute_percentage_error(true_valid, pred_valid)
            
            # print(err.shape)
            return np.mean(err)
        
        def calculate_pearson_r(ne_true_m, ne_pred_m):
            mask = ~np.isnan(ne_pred_m)
            if np.sum(mask) < 2:  # Need at least 2 points for correlation
                return 0
            true_valid = normalize(ne_true_m[mask])
            pred_valid = normalize(ne_pred_m[mask])
            r, _ = pearsonr(true_valid, pred_valid)
            return r
        
        
        def calculate_distance_cor(ne_true_m, ne_pred_m):
            mask = ~np.isnan(ne_pred_m)
            if np.sum(mask) < 2:  # Need at least 2 points for correlation
                return 0
            true_valid = normalize(ne_true_m[mask].astype(np.float64))
            pred_valid = normalize(ne_pred_m[mask].astype(np.float64))
            return dcor.distance_correlation(true_valid, pred_valid)
        
        
        def calculate_scores(ne_true, ne_pred, valid_measurements, score_func):
            metric_m = np.array([score_func(ne_true[:, m], ne_pred[:, m]) for m in valid_measurements])
            metric_tot = np.mean(metric_m)
            
            print(f"{score_func.__name__} result: {metric_tot:.2f}")
            return metric_tot
        
        # Define score functions
        a = calculate_r2_tot
        b = calculate_rmse_tot
        c = calculate_relative_error_tot
        d = calculate_pearson_r
        e = calculate_distance_cor
        
        # Calculate scores for valid measurements
        print("KIAN-Net Metrics:")
        r2_kian = calculate_scores(ne_eis, ne_kian, valid_measurements, a)
        rmse_kian = calculate_scores(ne_eis, ne_kian, valid_measurements, b)
        rel_err_kian = calculate_scores(ne_eis, ne_kian, valid_measurements, c)
        r_kian = calculate_scores(ne_eis, ne_kian, valid_measurements, d)
        dcor_kian = calculate_scores(ne_eis, ne_kian, valid_measurements, e)

        
        print("Artist Metrics:")
        r2_art  = calculate_scores(ne_eis, ne_art, valid_measurements, a)
        rmse_art  = calculate_scores(ne_eis, ne_art, valid_measurements, b)
        rel_err_art = calculate_scores(ne_eis, ne_art, valid_measurements, c)
        r_art  = calculate_scores(ne_eis, ne_art, valid_measurements, d)
        dcor_art = calculate_scores(ne_eis, ne_art, valid_measurements, e)

        
        print("E-CHAIM Metrics:")
        r2_ech  = calculate_scores(ne_eis, ne_ech, valid_measurements, a)
        rmse_ech  = calculate_scores(ne_eis, ne_ech, valid_measurements, b)
        rel_err_ech = calculate_scores(ne_eis, ne_ech, valid_measurements, c)
        r_ech  = calculate_scores(ne_eis, ne_ech, valid_measurements, d)
        dcor_ech = calculate_scores(ne_eis, ne_ech, valid_measurements, e)
        
        print("Iono-CNN Metrics:")
        r2_ion  = calculate_scores(ne_eis, ne_ion, valid_measurements, a)
        rmse_ion  = calculate_scores(ne_eis, ne_ion, valid_measurements, b)
        rel_err_ion = calculate_scores(ne_eis, ne_ion, valid_measurements, c)
        r_ion  = calculate_scores(ne_eis, ne_ion, valid_measurements, d)
        dcor_ion = calculate_scores(ne_eis, ne_ion, valid_measurements, e)
        
        print("Geo-DMLP Metrics:")
        r2_geo  = calculate_scores(ne_eis, ne_geo, valid_measurements, a)
        rmse_geo  = calculate_scores(ne_eis, ne_geo, valid_measurements, b)
        rel_err_geo = calculate_scores(ne_eis, ne_geo, valid_measurements, c)
        r_geo  = calculate_scores(ne_eis, ne_geo, valid_measurements, d)
        dcor_geo = calculate_scores(ne_eis, ne_geo, valid_measurements, e)
    
    # def plot_compare_all_error(self):
        
    #     # ___________ Getting Data ___________
        
    #     # Merging all global keys
    #     x_eis = merge_nested_dict(self.X_EIS)['All']
    #     x_kian = merge_nested_pred_dict(self.X_KIAN)['All']
    #     x_ion = merge_nested_pred_dict(self.X_ION)['All']
    #     x_geo = merge_nested_pred_dict(self.X_GEO)['All']
    #     x_art = merge_nested_dict(self.X_ART)['All']
    #     x_ech = merge_nested_dict(self.X_ECH)['All']
        
    #     r_time = from_array_to_datetime(x_eis['r_time'])
    #     r_h = x_eis['r_h'].flatten()
        
    #     # Ne-profiles
    #     ne_eis  = np.log10(x_eis["r_param"])
    #     ne_kian = x_kian["r_param"]
    #     ne_ion  = x_ion["r_param"]
    #     ne_geo  = x_geo["r_param"]
    #     ne_art  = np.log10(x_art["r_param"])
    #     ne_ech  = np.log10(x_ech["r_param"])
        
    #     date_str = r_time[0].strftime('%b-%Y')
        
        
        
    #     # Identify valid measurements where all EISCAT UHF values are non-NaN
    #     valid_measurements = [m for m in range(ne_eis.shape[1]) if np.all(~np.isnan(ne_eis[:, m]))]
    #     if not valid_measurements:
    #         print("No measurements with fully valid EISCAT UHF data found.")
    #         return
        
    #     from sklearn import preprocessing as pre
    #     def normalize(X):
            
    #         return pre.MinMaxScaler().fit_transform(X.reshape(-1, 1))
        
        
        
    #     # Helper functions for score calculations
    #     def calculate_r2_tot(ne_true_m, ne_pred_m):
    #         mask = ~np.isnan(ne_pred_m)
    #         if np.sum(mask) < 2:  # Need at least 2 points for R²
    #             return -1
            
    #         # print(ne_true_m.shape, ne_pred_m.shape)
    #         true_valid = normalize(ne_true_m[mask])
    #         pred_valid = normalize(ne_pred_m[mask])
    #         return r2_score(true_valid.flatten(), pred_valid.flatten())
    
    #     def calculate_rmse_tot(ne_true_m, ne_pred_m):
    #         mask = ~np.isnan(ne_pred_m)
    #         if np.sum(mask) < 1:  # Need at least 1 point for RMSE
    #             return 2
    #         true_valid = normalize(ne_true_m[mask])
    #         pred_valid = normalize(ne_pred_m[mask])
    #         mse = np.mean((true_valid.flatten() - pred_valid.flatten()) ** 2)
    #         return np.sqrt(mse)
        
        
    #     def calculate_scores(ne_true, ne_pred, valid_measurements, score_func):
            
    #         metric_m = np.array([score_func(ne_true[:, m], ne_pred[:, m]) for m in valid_measurements])
    #         metric_tot = np.mean(metric_m)
    #         print(metric_tot)
    #         return metric_tot
        
    #     # Calculate scores for valid measurements
    #     a = calculate_r2_tot
    #     b = calculate_rmse_tot
        
        
        
        
        
    #     r2_kian = calculate_scores(ne_eis, ne_kian, valid_measurements, a)
    #     # r2_art  = calculate_scores(ne_eis, ne_art, valid_measurements, a)
    #     # r2_ech  = calculate_scores(ne_eis, ne_ech, valid_measurements, a)
    #     # r2_ion  = calculate_scores(ne_eis, ne_ion, valid_measurements, a)
    #     # r2_geo  = calculate_scores(ne_eis, ne_geo, valid_measurements, a)
    
    #     rmse_kian = calculate_scores(ne_eis, ne_kian, valid_measurements, b)
    #     # rmse_art  = calculate_scores(ne_eis, ne_art, valid_measurements, b)
    #     # rmse_ech  = calculate_scores(ne_eis, ne_ech, valid_measurements, b)
    #     # rmse_ion  = calculate_scores(ne_eis, ne_ion, valid_measurements, b)
    #     # rmse_geo  = calculate_scores(ne_eis, ne_geo, valid_measurements, b)
        
        
        
        # # ___________ Defining axes ___________
        # fig = plt.figure(figsize=(12, 12))
        # fig.suptitle(f'Comparrison Between Prediction Models and Ground Truth\nDate: {date_str}', fontsize=17, y=0.97)
        
        # gs = GridSpec(6, 2, width_ratios=[1, 0.015], wspace=0.1, hspace=0.35)
        # ax0 = fig.add_subplot(gs[0, 0])
        # ax1 = fig.add_subplot(gs[1, 0], sharex=ax0)
        # ax2 = fig.add_subplot(gs[2, 0], sharex=ax0)
        # ax3 = fig.add_subplot(gs[3, 0], sharex=ax0)
        # ax4 = fig.add_subplot(gs[4, 0], sharex=ax0)
        # ax5 = fig.add_subplot(gs[5, 0], sharex=ax0)
        # cax = fig.add_subplot(gs[:, 1])
        
        
        
        # #  ___________ Creating Plots  ___________
        # MIN, MAX = 1e10, 1e12
        
        # ne = ax0.pcolormesh(r_time, r_h, ne_eis, shading='gouraud', cmap='turbo', norm=LogNorm(vmin=MIN, vmax=MAX))
        # ax1.pcolormesh(r_time, r_h, ne_kian, shading='gouraud', cmap='turbo', norm=LogNorm(vmin=MIN, vmax=MAX))
        # ax2.pcolormesh(r_time, r_h, ne_ion, shading='gouraud', cmap='turbo', norm=LogNorm(vmin=MIN, vmax=MAX))
        # ax3.pcolormesh(r_time, r_h, ne_geo, shading='gouraud', cmap='turbo', norm=LogNorm(vmin=MIN, vmax=MAX))
        # ax4.pcolormesh(r_time, r_h, ne_art, shading='gouraud', cmap='turbo', norm=LogNorm(vmin=MIN, vmax=MAX))
        # ax5.pcolormesh(r_time, r_h, ne_ech, shading='gouraud', cmap='turbo', norm=LogNorm(vmin=MIN, vmax=MAX))
        
        # # Set titles and labels
        # ax0.set_title('(a) EISCAT UHF', fontsize=16)
        # ax1.set_title('(b) KIAN-Net', fontsize=16)
        # ax2.set_title('(c) Iono-CNN', fontsize=16)
        # ax3.set_title('(d) Geo-DMLP', fontsize=16)
        # ax4.set_title('(e) Artist 5.0', fontsize=16)
        # ax5.set_title('(f) E-Chaim', fontsize=16)
        
        # # y-labels
        # fig.supylabel('Altitude [km]', x=0.075)
        
        # # x-label
        # ax5.set_xlabel('UT [dd hh:mm]', fontsize=13)
        
        # # Ticks
        # ax0.tick_params(labelbottom=False)  # Hide x-ticks on EISCAT plot
        # ax1.tick_params(labelbottom=False)
        # ax2.tick_params(labelbottom=False)
        # ax3.tick_params(labelbottom=False)
        # ax4.tick_params(labelbottom=False)
        # ax5.xaxis.set_major_formatter(DateFormatter('%d %H:%M'))
        
        # plt.setp(ax5.xaxis.get_majorticklabels(), rotation=45, ha='center')
        
        # # Add colorbar
        # cbar = fig.colorbar(ne, cax=cax, orientation='vertical')
        # cbar.set_label('$log_{10}$ $N_e$  (m$^{-3}$)', fontsize=13)
        
        # plt.show()
    
    
    
    
    
    

    def plot_r2_rmse_dist(self, overlay_pdf=False):
        """
        Plots histograms and PDFs of R² scores and RMSE for KIAN-Net, Artist 5.0, E-Chaim,
        Iono-CNN, and Geo-DMLP models compared to EISCAT UHF data.
    
        Two plotting styles are available:
        
          - Overlay (if overlay_pdf=True): Uses a 1x2 grid where each subplot overlays the histogram
            and its PDF. In this style custom legend handles are built so that the histogram legend
            patch shows its facecolor (with alpha) and an edge colored as the PDF, and the PDF line
            appears as its own legend entry.
          - Separate (if overlay_pdf=False): Uses a 2x2 grid with histograms on the top row and PDFs
            on the bottom row.
            
        Only measurements with no NaNs in the EISCAT UHF data are considered.
        """
        # Merge dictionaries for EISCAT and selected models
        x_eis = merge_nested_dict(self.X_EIS)['All']
        x_kian = merge_nested_pred_dict(self.X_KIAN)['All']
        x_ion = merge_nested_pred_dict(self.X_ION)['All']
        x_geo = merge_nested_pred_dict(self.X_GEO)['All']
        x_art = merge_nested_dict(self.X_ART)['All']
        x_ech = merge_nested_dict(self.X_ECH)['All']
    
        # Extract electron density profiles (Ne)
        ne_eis = np.log10(x_eis['r_param'])   # EISCAT provides Ne directly (log10)
        ne_kian = x_kian['r_param']             # KIAN already provides log10(Ne)
        ne_ion = x_ion['r_param']
        ne_geo = x_geo['r_param']
        ne_art = np.log10(x_art['r_param'])
        ne_ech = np.log10(x_ech['r_param'])
    
        # Identify valid measurements where all EISCAT UHF values are non-NaN
        valid_measurements = [m for m in range(ne_eis.shape[1]) if np.all(~np.isnan(ne_eis[:, m]))]
        if not valid_measurements:
            print("No measurements with fully valid EISCAT UHF data found.")
            return
    
        # Helper functions for score calculations
        def calculate_r2_for_m(ne_true_m, ne_pred_m):
            mask = ~np.isnan(ne_pred_m)
            if np.sum(mask) < 2:  # Need at least 2 points for R²
                return -1
            true_valid = ne_true_m[mask]
            pred_valid = ne_pred_m[mask]
            return r2_score(true_valid, pred_valid)
    
        def calculate_rmse_for_m(ne_true_m, ne_pred_m):
            mask = ~np.isnan(ne_pred_m)
            if np.sum(mask) < 1:  # Need at least 1 point for RMSE
                return 2
            true_valid = ne_true_m[mask]
            pred_valid = ne_pred_m[mask]
            mse = np.mean((true_valid - pred_valid) ** 2)
            return np.sqrt(mse)
    
        def calculate_scores(ne_true, ne_pred, valid_measurements, score_func):
            return np.array([score_func(ne_true[:, m], ne_pred[:, m]) for m in valid_measurements])
    
        # Calculate scores for valid measurements
        r2_kian = calculate_scores(ne_eis, ne_kian, valid_measurements, calculate_r2_for_m)
        r2_art  = calculate_scores(ne_eis, ne_art, valid_measurements, calculate_r2_for_m)
        r2_ech  = calculate_scores(ne_eis, ne_ech, valid_measurements, calculate_r2_for_m)
        r2_ion  = calculate_scores(ne_eis, ne_ion, valid_measurements, calculate_r2_for_m)
        r2_geo  = calculate_scores(ne_eis, ne_geo, valid_measurements, calculate_r2_for_m)
    
        rmse_kian = calculate_scores(ne_eis, ne_kian, valid_measurements, calculate_rmse_for_m)
        rmse_art  = calculate_scores(ne_eis, ne_art, valid_measurements, calculate_rmse_for_m)
        rmse_ech  = calculate_scores(ne_eis, ne_ech, valid_measurements, calculate_rmse_for_m)
        rmse_ion  = calculate_scores(ne_eis, ne_ion, valid_measurements, calculate_rmse_for_m)
        rmse_geo  = calculate_scores(ne_eis, ne_geo, valid_measurements, calculate_rmse_for_m)
    
        total_measurements = ne_eis.shape[1]
        num_valid = len(valid_measurements)
        num_invalid = total_measurements - num_valid
    
        # Define models and colors (same for both styles)
        models = ['KIAN-Net', 'Artist 5.0', 'E-Chaim', 'Iono-CNN', 'Geo-DMLP']
        colors = ['C1', 'C2', 'C3', 'C4', 'C5']
    
        if overlay_pdf:
            # Set up a 1x2 grid where each subplot overlays histogram and PDF
            fig = plt.figure(figsize=(15, 6))
            gs = GridSpec(1, 2, figure=fig, wspace=0.2)
            with sns.axes_style("dark"):
                ax1 = fig.add_subplot(gs[0, 0])  # R² subplot
                ax2 = fig.add_subplot(gs[0, 1])  # RMSE subplot
            
            alph = 0.3  # histogram alpha
    
            # ----- R² Overlay -----
            
            LineStyle = ['-', (0, (1, 1)), '-.', (0, (3, 1)), (0, (3, 1,1,5,5,1))]
            
            bins_r2 = np.linspace(-2, 1, 31)
            for r2_list, model, color, l in zip([r2_kian, r2_art, r2_ech, r2_ion, r2_geo], models, colors, LineStyle):
                # Plot histogram (without legend labels)
                ax1.hist(r2_list, bins=bins_r2, alpha=alph, color=color)
                # Plot PDF on top (without legend labels)
                valid_r2 = r2_list[r2_list != -1]
                if len(valid_r2) > 1:
                    kde = gaussian_kde(valid_r2, bw_method=0.03)
                    x = np.linspace(bins_r2[0], bins_r2[-1], 1000)
                    pdf = kde(x)
                    bin_width = bins_r2[1] - bins_r2[0]
                    pdf_scaled = pdf * len(valid_r2) * bin_width
                    ax1.plot(x, pdf_scaled, color=color, linestyle=l, linewidth=3)
            ax1.axvline(0, color='black', linestyle='--')
            ax1.set_title('(a) R² Score Distribution', fontsize=17)
            ax1.set_xlabel('R²-Score', fontsize=15)
            ax1.set_ylabel('Scaled Density', fontsize=15)
            ax1.grid(True)
            
            # Build custom legend handles for R²
            legend_handles_r2 = []
            for model, color in zip(models, colors):
                hist_patch = Patch(facecolor=to_rgba(color, alpha=alph),
                                   edgecolor=color,
                                   linewidth=2.5,
                                   label=f'{model}')

                legend_handles_r2.extend([hist_patch])
            ax1.legend(handles=legend_handles_r2)
            
            
            
            
            # ----- RMSE Overlay -----
            rmse_lists = [rmse_kian, rmse_art, rmse_ech, rmse_ion, rmse_geo]
            all_rmse = np.concatenate(rmse_lists)
            valid_rmse_all = all_rmse[~np.isnan(all_rmse)]
            if len(valid_rmse_all) > 0:
                bins_rmse = np.linspace(np.min(valid_rmse_all), np.max(valid_rmse_all), 31)
            else:
                bins_rmse = np.linspace(0, 1, 31)
                
            LineStyle = ['-', '-.', '--', '-', '-']
            for rmse_list, model, color, l in zip(rmse_lists, models, colors, LineStyle):
                # Plot histogram (without legend labels)
                ax2.hist(rmse_list, bins=bins_rmse, alpha=alph, color=color)
                # Plot PDF on top (without legend labels)
                valid_rmse = rmse_list[~np.isnan(rmse_list)]
                if len(valid_rmse) > 1:
                    kde = gaussian_kde(valid_rmse, bw_method=0.2)
                    x = np.linspace(bins_rmse[0], bins_rmse[-1], 1000)
                    pdf = kde(x)
                    bin_width = bins_rmse[1] - bins_rmse[0]
                    pdf_scaled = pdf * len(valid_rmse) * bin_width
                    ax2.plot(x, pdf_scaled, color=color, linestyle=l, linewidth=3)
            ax2.set_title('(b) RMSE Distribution', fontsize=17)
            ax2.set_xlabel('RMSE', fontsize=15)
            ax2.set_ylabel('Scaled Density', fontsize=15)
            ax2.grid(True)
            
            # Build custom legend handles for RMSE
            legend_handles_rmse = []
            for model, color in zip(models, colors):
                hist_patch = Patch(facecolor=to_rgba(color, alpha=alph),
                                   edgecolor=color,
                                   linewidth=2.5,
                                   label=f'{model}')

                legend_handles_rmse.extend([hist_patch])
            ax2.legend(handles=legend_handles_rmse)
    
            fig.suptitle('Distribution of R² and RMSE', fontsize=20, y=1.01)
            # plt.tight_layout(rect=[0, 0, 1, 0.95])
            plt.show()
    
        else:
            # Set up a 2x2 grid using GridSpec: top row for histograms, bottom row for PDFs
            fig = plt.figure(figsize=(12, 10))
            gs = GridSpec(2, 2, figure=fig, wspace=0.25, hspace=0.25)
            with sns.axes_style("dark"):
                ax_hist_r2  = fig.add_subplot(gs[0, 0])
                ax_hist_rmse = fig.add_subplot(gs[0, 1])
                
                ax_hist_r2.tick_params(labelbottom=False)
                ax_hist_rmse.tick_params(labelbottom=False)
                ax_pdf_r2   = fig.add_subplot(gs[1, 0])
                ax_pdf_rmse = fig.add_subplot(gs[1, 1])
                
            alph = 0.75
            # ----- R² Plots -----
            bins_r2 = np.linspace(-2, 1, 31)  # Fixed range for R² scores
            # Histograms (top left)
            for r2_list, model, color in zip([r2_kian, r2_art, r2_ech, r2_ion, r2_geo],
                                             models, colors):
                ax_hist_r2.hist(r2_list, bins=bins_r2, alpha=alph, color=color, label=model)
            ax_hist_r2.axvline(0, color='black', linestyle='--')
            ax_hist_r2.set_title('(a) R² Score Histogram', fontsize=17)
            # ax_hist_r2.set_xlabel('R² Score')
            ax_hist_r2.set_ylabel('Samples', fontsize=15)
            ax_hist_r2.legend(fontsize=10)
            ax_hist_r2.grid(True)
            
            # PDFs (bottom left)
            for r2_list, model, color in zip([r2_kian, r2_art, r2_ech, r2_ion, r2_geo],
                                             models, colors):
                valid_r2 = r2_list[r2_list != -1]  # Exclude flag values (insufficient data)
                if len(valid_r2) > 1:
                    kde = gaussian_kde(valid_r2, bw_method=0.03)
                    x = np.linspace(bins_r2[0], bins_r2[-1], 1000)
                    pdf = kde(x)
                    bin_width = bins_r2[1] - bins_r2[0]
                    pdf_scaled = pdf * len(valid_r2) * bin_width
                    ax_pdf_r2.plot(x, pdf_scaled, color=color, linestyle='-', linewidth=2, label=model)
            ax_pdf_r2.axvline(0, color='black', linestyle='--')
            ax_pdf_r2.set_title('(c) R² Score PDF', fontsize=17)
            ax_pdf_r2.set_xlabel('R² Score', fontsize=15)
            ax_pdf_r2.set_ylabel('Scaled Density', fontsize=15)
            ax_pdf_r2.legend(fontsize=10)
            ax_pdf_r2.grid(True)
        
            # ----- RMSE Plots -----
            rmse_lists = [rmse_kian, rmse_art, rmse_ech, rmse_ion, rmse_geo]
            all_rmse = np.concatenate(rmse_lists)
            valid_rmse_all = all_rmse[~np.isnan(all_rmse)]
            if len(valid_rmse_all) > 0:
                bins_rmse = np.linspace(np.min(valid_rmse_all), np.max(valid_rmse_all), 31)
            else:
                bins_rmse = np.linspace(0, 1, 31)
            # Histograms (top right)
            for rmse_list, model, color in zip(rmse_lists, models, colors):
                ax_hist_rmse.hist(rmse_list, bins=bins_rmse, alpha=alph, color=color, label=model)
            ax_hist_rmse.set_title('(b) RMSE Histogram', fontsize=17)
            # ax_hist_rmse.set_xlabel('RMSE')
            ax_hist_rmse.set_ylabel('Samples', fontsize=15)
            ax_hist_rmse.legend(fontsize=10)
            ax_hist_rmse.grid(True)
            
            # PDFs (bottom right)
            for rmse_list, model, color in zip(rmse_lists, models, colors):
                valid_rmse = rmse_list[~np.isnan(rmse_list)]
                if len(valid_rmse) > 1:
                    kde = gaussian_kde(valid_rmse, bw_method=0.2)
                    x = np.linspace(bins_rmse[0], bins_rmse[-1], 1000)
                    pdf = kde(x)
                    bin_width = bins_rmse[1] - bins_rmse[0]
                    pdf_scaled = pdf * len(valid_rmse) * bin_width
                    ax_pdf_rmse.plot(x, pdf_scaled, color=color, linestyle='-', linewidth=2, label=model)
            ax_pdf_rmse.set_title('(d) RMSE PDF', fontsize=17)
            ax_pdf_rmse.set_xlabel('RMSE', fontsize=15)
            ax_pdf_rmse.set_ylabel('Scaled Density', fontsize=15)
            ax_pdf_rmse.legend(fontsize=10)
            ax_pdf_rmse.grid(True)
            
            fig.suptitle(f'Distribution of R² and RMSE\nBetween Predictions and EISCAT UHF', fontsize=20, y=1.01)
            plt.show()
        
        # Print summary information
        print(f"Total measurements: {total_measurements}")
        print(f"Valid measurements (no NaNs in EISCAT UHF): {num_valid}")
        print(f"Excluded measurements: {num_invalid}")
        
        
        
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
    
    
    # def plot_compare_all_peak_densities(self, show_marginal_kde=False):
        
    #     # Merge data dictionaries for all regions
    #     X_eis = merge_nested_peak_dict(self.X_EIS)['All']
    #     X_kian = merge_nested_peak_dict(self.X_KIAN)['All']
    #     X_ion = merge_nested_peak_dict(self.X_ION)['All']
    #     X_geo = merge_nested_peak_dict(self.X_GEO)['All']
    #     X_art = merge_nested_peak_dict(self.X_ART)['All']
    #     X_ech = merge_nested_peak_dict(self.X_ECH)['All']
    
    #     # Prepare peak density data (all in log10(ne))
    #     eis_param_peak = np.log10(X_eis["r_param_peak"])
    #     kian_param_peak = X_kian["r_param_peak"]
    #     ion_param_peak = X_ion["r_param_peak"]
    #     geo_param_peak = X_geo["r_param_peak"]
    #     art_param_peak = np.log10(X_art["r_param_peak"])
    #     ech_param_peak = np.log10(X_ech["r_param_peak"])
    
    #     # Define models with names, data, and colors
    #     models = [
    #         {"name": "EISCAT UHF", "data": eis_param_peak, "color": "C0", "Z_order":0},
    #         {"name": "KIAN-Net", "data": kian_param_peak, "color": "C1", "Z_order":3},
    #         {"name": "Artist 5.0", "data": art_param_peak, "color": "C2", "Z_order":4},
    #         {"name": "E-CHAIM", "data": ech_param_peak, "color": "C3", "Z_order":5},
    #         {"name": "Iono-CNN", "data": ion_param_peak, "color": "C4", "Z_order":2},
    #         {"name": "Geo-DMLP", "data": geo_param_peak, "color": "C5", "Z_order":1},
    #     ]
    
    #     # Convert time to datetime for title
    #     r_time = from_array_to_datetime(X_eis["r_time"])
    #     # date_str = r_time[0].strftime('%b-%Y')
        
        
    #     fig = plt.figure(figsize=(14, 7))
    
    #     # Set up GridSpec based on whether marginal KDEs are requested
    #     if show_marginal_kde:
    #         gs = GridSpec(2, 5, height_ratios=[0.2, 1], width_ratios=[1, 0.2, 0.2, 1, 0.2], hspace=0, wspace=0)
    #         with sns.axes_style("dark"):
    #             ax_e = fig.add_subplot(gs[1, 0])
    #             ax_f = fig.add_subplot(gs[1, 3])
                
    #         ax_e_top = fig.add_subplot(gs[0, 0], sharex=ax_e)
    #         ax_f_top = fig.add_subplot(gs[0, 3], sharex=ax_f)

            
    #         # Turn off x-axis for ax_e_top only
    #         ax_e_top.spines['bottom'].set_visible(False)
    #         ax_e_top.spines['top'].set_visible(False)
    #         ax_e_top.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    #         ax_e_top.spines['left'].set_position('center')
    #         ax_e_top.set_ylabel("PDF", rotation=0, ha='center', va='bottom', labelpad=15, fontsize=12)
    #         ax_e_top.yaxis.set_label_coords(0.49, 1.1)  # Position at center (x=0.5) and top (y=1.0)
    #         ax_e_top.set_ylim(0, 2)
            
    #         # Turn off x-axis for ax_f_top only
    #         ax_f_top.spines['bottom'].set_visible(False)
    #         ax_f_top.spines['top'].set_visible(False)
    #         ax_f_top.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    #         ax_f_top.spines['left'].set_position('center')
    #         ax_f_top.set_ylabel("PDF", rotation=0, ha='center', va='bottom', labelpad=15, fontsize=12)
    #         ax_f_top.yaxis.set_label_coords(0.49, 1.1)  # Position at center (x=0.5) and top (y=1.0)
    #         ax_f_top.set_ylim(0, 2)
            
    #         # Spacing between e/f-reg plots
    #         ax_spacing = fig.add_subplot(gs[:, 3])
    #         ax_spacing.set_axis_off()
            
            
    #         ax_e_right = fig.add_subplot(gs[1, 1], sharey=ax_e)
    #         ax_f_right = fig.add_subplot(gs[1, 4], sharey=ax_f)
            
    #         # Turn off x-axis for ax_e_top only
    #         ax_e_right.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
    #         ax_e_right.spines['bottom'].set_position('center')
    #         ax_e_right.set_xlabel("PDF", rotation=-90, ha='center', va='bottom', labelpad=15, fontsize=12)
    #         ax_e_right.xaxis.set_label_coords(1.1, 0.465)  # Position at center (x=0.5) and top (y=1.0)
            
    #         # Turn off x-axis for ax_f_top only
    #         ax_f_right.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
    #         ax_f_right.spines['bottom'].set_position('center')
    #         ax_f_right.set_xlabel("PDF", rotation=-90, ha='center', va='bottom', labelpad=15, fontsize=12)
    #         ax_f_right.xaxis.set_label_coords(1.1, 0.465)  # Position at center (x=0.5) and top (y=1.0)
            
    #     else:
    #         gs = GridSpec(1, 2, width_ratios=[1, 1], wspace=0.1)
    #         ax_e = fig.add_subplot(gs[0, 0])
    #         ax_f = fig.add_subplot(gs[0, 1])
    
    #     fig.suptitle('Peak Elecron densities\n(EISCAT UHF vs Model)', fontsize=20, y=0.95)
    
    #     def plot_region(ax, region, Min, Max, title, ax_top=None, ax_right=None):
    #         # Plot the diagonal line (perfect agreement)
    #         diagonal_line = np.array([Min, Max])
    #         ax.plot([Min, Max], diagonal_line, color="C0", 
    #                 linewidth=3, zorder=1)
    
    #         # Loop over models to plot scatter, regression lines, metrics, and optional KDEs
    #         for idx, model in enumerate(models):
    #             # Extract valid data for the region
    #             model_data = model["data"][region, :]
    #             valid = ~np.isnan(model_data) & ~np.isnan(eis_param_peak[region, :])
    #             x = model_data[valid]  # Predicted values (model)
    #             y = eis_param_peak[region, valid]  # Observed values (EISCAT)
                
                
    #             # Plot scatter points
    #             if model["name"] == "EISCAT UHF":
    #                 ax.plot(x, y, color=model["color"], label=model["name"], 
    #                             linestyle="-", marker='o', mec = 'black', zorder=model["Z_order"])
    #             else:
    #                 ax.scatter(x, y, s=40, color=model["color"], label=model["name"], 
    #                            edgecolors="black", zorder=model["Z_order"])
                    
    #             # Subplots titles
    #             ax.text(0.3, 0.9, s=f'{title} Peaks', color='black',
    #                     fontsize=15, transform=ax.transAxes)
                
    #             # Check if there are any valid predictions
    #             if valid.any():
                    
    #                 r_corr, _ = pearsonr(x, y)
    #                 r2 = self.compute_r2_score(y, x)
    #                 rmse = np.sqrt(np.mean((y - x) ** 2))
                    
    #                 if model["name"] != "EISCAT UHF":
    #                     # Annotate R² and RMSE with model name
    #                     ax.text(0.6, 0.235 - (0.04 * idx), 
    #                             s=rf"$\mathbf{{({r2:.2f},\ {rmse:.2f},\ {r_corr:.2f})}}$", 
    #                             color=model["color"], fontsize=10, transform=ax.transAxes)
    #                 else:
    #                     pass
                    
    #                 # Plot marginal KDEs if axes are provided
    #                 if ax_top is not None and ax_right is not None:
    #                     # KDE for x (predicted densities)
    #                     kde_x = gaussian_kde(x)
    #                     x_grid = np.linspace(Min, Max, 100)
    #                     kde_x_values = kde_x(x_grid)
    #                     if model["name"] == "EISCAT UHF":
    #                         lw, ls, zo = 3, "--", 5
                        
    #                     elif model["name"] == "KIAN-Net":
    #                         ls, zo = "-", 4
                            
    #                     else:
    #                         lw, ls, zo = 2, "-", 1
                        
    #                     ax_top.plot(x_grid, kde_x_values, color=model["color"], linewidth=lw, linestyle=ls, zorder=zo)
                            
                        
    #                     # KDE for y (observed densities)
    #                     kde_y = gaussian_kde(y)
    #                     y_grid = np.linspace(Min, Max, 100)
    #                     kde_y_values = kde_y(y_grid)
    #                     if model["name"] == "EISCAT UHF":
    #                         lw, ls, zo = 3, "--", 5
                        
    #                     elif model["name"] == "KIAN-Net":
    #                         ls, zo = "-", 4
                        
    #                     else:
    #                         lw, ls, zo = 2, "-", 1
    #                     ax_right.plot(kde_y_values, y_grid, color=model["color"], linewidth=lw, linestyle=ls, zorder=zo)
    #             else:
    #                 # If no valid data, display "N/A"
    #                 ax.text(0.6, 0.2 - (0.04 * idx), 
    #                         s=rf"$\mathbf{{(N/A,\ N/A,\ N/A)}}$", 
    #                         color=model["color"], fontsize=10, transform=ax.transAxes)
                
    #             if idx == 4:
    #                 ax.text(0.6, 0.2 - (0.04 * (-1)), 
    #                         s=rf"$\mathbf{{(R^{2},\ RMSE, \, r)}}$", 
    #                         color="black", fontsize=10, transform=ax.transAxes)
                    
    #         # Set plot aesthetics
    #         ax.set_xlim(Min, Max)
    #         ax.set_ylim(Min, Max)
    #         ax.grid(True)
            
    #         # Configure marginal axes if they exist
    #         if ax_top is not None and ax_right is not None:
    #             # Set limits to start at 0, with automatic upper bounds
    #             ax_top.set_ylim([0, None])
    #             ax_right.set_xlim([0, None])
                
    #             # Hide ticks that overlap with main plot
    #             ax_top.tick_params(labelbottom=False)
    #             ax_right.tick_params(labelleft=False)
                
    #             # Remove unnecessary spines for a cleaner look
    #             ax_top.spines['bottom'].set_visible(False)
    #             ax_top.spines['top'].set_visible(False)
    #             ax_top.spines['right'].set_visible(False)
    #             ax_right.spines['left'].set_visible(False)
    #             ax_right.spines['top'].set_visible(False)
    #             ax_right.spines['right'].set_visible(False)
    
    #     # Plot E-region and F-region with appropriate parameters
    #     if show_marginal_kde:
    #         plot_region(ax_e, region=0, Min=8.5, Max=12.5, title=' (a) E-region', ax_top=ax_e_top, ax_right=ax_e_right)
    #         plot_region(ax_f, region=1, Min=9.5, Max=12.5, title=' (b) F-region', ax_top=ax_f_top, ax_right=ax_f_right)
    #     else:
    #         plot_region(ax_e, region=0, Min=8.5, Max=12, title='E-region')
    #         plot_region(ax_f, region=1, Min=9.5, Max=12.5, title='F-region')
    
    #     # Set axis labels
    #     ax_e.set_xlabel(r'Prediction  $log_{10}\,(n_e)$', fontsize=15)
    #     ax_e.set_ylabel(r'EISCAT UHF  $log_{10}\,(n_e)$', fontsize=15)
    #     ax_f.set_xlabel(r'Model  $log_{10}\,(n_e)$', fontsize=15)
    
    #     # Add legend to F-region plot with white background
    #     ax_e.legend(fontsize=9, facecolor='white', loc='upper left')
    #     ax_f.legend(fontsize=9, facecolor='white', loc='upper left')
    
    #     plt.show()
    
    def plot_compare_all_peak_densities(self, show_marginal_kde=False):
        """Compare model peak densities with EISCAT UHF measurements."""
        #region Data Preparation
        def _prepare_models():
            """Prepare model data with consistent transformations."""
            model_configs = [
                ('EIS', 'EISCAT UHF', "C0", 0, np.log10),
                ('KIAN', 'KIAN-Net', "C1", 3, None),
                ('ART', 'Artist 5.0', "C2", 4, np.log10),
                ('ECH', 'E-CHAIM', "C3", 5, np.log10),
                ('ION', 'Iono-CNN', "C4", 2, None),
                ('GEO', 'Geo-DMLP', "C5", 1, None),
            ]
            
            models = []
            for key, name, color, z_order, transform in model_configs:
                data = merge_nested_peak_dict(getattr(self, f'X_{key}'))['All']
                param_peak = data["r_param_peak"]
                if transform:
                    param_peak = transform(param_peak)
                models.append({
                    'name': name, 'data': param_peak,
                    'color': color, 'z_order': z_order
                })
            return models
    
        models = _prepare_models()
        eis_data = models[0]['data']
        #endregion
    
        #region Figure Setup
        fig = plt.figure(figsize=(14, 7))
        fig.suptitle('Peak Electron Densities\n(EISCAT UHF vs Model)', fontsize=20, y=0.95)
    
        if show_marginal_kde:
            gs = GridSpec(2, 5, height_ratios=[0.2, 1], 
                         width_ratios=[1, 0.2, 0.2, 1, 0.2], hspace=0, wspace=0)
            with sns.axes_style("dark"):
                ax_e = fig.add_subplot(gs[1, 0])
                ax_f = fig.add_subplot(gs[1, 3])
            
            axes = {
                'e': (ax_e, fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[1, 1])),
                'f': (ax_f, fig.add_subplot(gs[0, 3]), fig.add_subplot(gs[1, 4]))
            }
            fig.add_subplot(gs[:, 2]).set_axis_off()
        else:
            gs = GridSpec(1, 2, width_ratios=[1, 1], wspace=0.1)
            with sns.axes_style("dark"):
                axes = {
                    'e': (fig.add_subplot(gs[0, 0]), None, None),
                    'f': (fig.add_subplot(gs[0, 1]), None, None)
                }
    
        # Configure marginal axes with original styling
        if show_marginal_kde:
            for region in ['e', 'f']:
                main_ax, top_ax, right_ax = axes[region]
                
                # Configure top axis (x-axis KDE)
                top_ax.spines['bottom'].set_visible(False)
                top_ax.spines['top'].set_visible(False)
                top_ax.tick_params(axis='x', which='both', 
                                  bottom=False, top=False, labelbottom=False)
                top_ax.spines['left'].set_position('center')
                top_ax.set_ylabel("PDF", rotation=0, ha='center', va='bottom', 
                                labelpad=15, fontsize=12)
                top_ax.yaxis.set_label_coords(0.49, 1.1)
                top_ax.set_ylim(0, 2)
                
                # Configure right axis (y-axis KDE)
                right_ax.tick_params(axis='y', which='both', 
                                    left=False, right=False, labelleft=False)
                right_ax.spines['bottom'].set_position('center')
                right_ax.set_xlabel("PDF", rotation=-90, ha='center', va='bottom',
                                  labelpad=15, fontsize=12)
                right_ax.xaxis.set_label_coords(1.1, 0.465)
                
                # Remove other spines
                for ax in [top_ax, right_ax]:
                    ax.spines['right'].set_visible(False)
                    ax.spines['left'].set_visible(False) if ax == right_ax else None
                    ax.spines['top'].set_visible(False) if ax == right_ax else None
                    ax.spines['bottom'].set_visible(False) if ax == top_ax else None
        #endregion
    
        #region Plotting Functions
        def _plot_diagonal(ax, min_val, max_val):
            """Plot perfect agreement diagonal."""
            ax.plot([min_val, max_val], [min_val, max_val], 
                    color="C0", linewidth=3, zorder=1)
    
        def _plot_model_points(ax, model, x, y):
            """Plot data points for a single model."""
            if model['name'] == "EISCAT UHF":
                ax.plot(x, y, color=model['color'], marker='o', mec='black',
                       linestyle='', markersize=5, zorder=model['z_order'])
            else:
                ax.scatter(x, y, s=40, color=model['color'], 
                          edgecolors='black', zorder=model['z_order'])
    
        
        def _calculate_metrics(x, y):
            """Calculate performance metrics including BC and KL-Divergence."""
            metrics = {
                'r2': self.compute_r2_score(y, x),
                'rmse': np.sqrt(np.mean((y - x)**2)),
                'r_corr': pearsonr(x, y)[0]
            }
            
            # Define grid for integration based on the region's range
            min_val = min(np.min(x), np.min(x))
            max_val = max(np.max(x), np.max(y))
            x_grid = np.linspace(min_val, max_val, 1000)
            
            # Compute KDEs
            kde_eiscat = gaussian_kde(y)  # EISCAT as reference (y)
            kde_model = gaussian_kde(x)   # Model predictions (x)
            
            # Evaluate PDFs on the grid
            p = kde_eiscat(x_grid)
            q = kde_model(x_grid)
            
            # Bhattacharyya Coefficient: ∫ sqrt(p(x) q(x)) dx
            bc = trapz(np.sqrt(p * q), x_grid)
            metrics['bc'] = bc
            
            # KL-Divergence: ∫ p(x) log(p(x)/q(x)) dx
            # Add small epsilon to q to avoid log(0) or division by zero
            q_safe = np.where(q < 1e-10, 1e-10, q)
            kl = trapz(p * np.log(p / q_safe), x_grid)
            metrics['kl'] = kl
            
            return metrics
    
        def _plot_kdes(model, x, y, min_val, max_val, top_ax, right_ax):
            """Plot marginal KDE distributions with custom line widths."""
            style = {
                'linestyle': '--' if model['name'] == "EISCAT UHF" else '-',
                'zorder': model['z_order'],
                'linewidth': 2  # Default linewidth
            }
            
            if model['name'] in ["EISCAT UHF", "KIAN-Net"]:
                style.update({
                    'linewidth': 3,  # Force linewidth=2 for these models
                    'zorder': 7 if model['name'] == "EISCAT UHF" else 6
                })
            
            # X-axis KDE
            kde_x = gaussian_kde(x)
            x_grid = np.linspace(min_val, max_val, 100)
            top_ax.plot(x_grid, kde_x(x_grid), color=model['color'], **style)
            
            # Y-axis KDE
            kde_y = gaussian_kde(y)
            y_grid = np.linspace(min_val, max_val, 100)
            right_ax.plot(kde_y(y_grid), y_grid, color=model['color'], **style)
        
    
        #region Main Plotting Logic
        region_configs = [
            {'key': 'e', 'index': 0, 'min': 8.5, 'max': 12.5, 'title': ' (a) E-region'},
            {'key': 'f', 'index': 1, 'min': 9.5, 'max': 12.5, 'title': ' (b) F-region'}
        ]
    
        for config in region_configs:
            main_ax, top_ax, right_ax = axes[config['key']]
            _plot_diagonal(main_ax, config['min'], config['max'])
            
            # Add region title
            main_ax.text(0.3, 0.9, f"{config['title']} Peaks", 
                        fontsize=15, transform=main_ax.transAxes)
            
            # Create legend elements
            legend_elements = []
            
            for idx, model in enumerate(models):
                # Get valid data
                valid = ~np.isnan(model['data'][config['index']]) & ~np.isnan(eis_data[config['index']])
                
                x = model['data'][config['index'], valid]
                y = eis_data[config['index'], valid]
    
                # Plot main data points
                _plot_model_points(main_ax, model, x, y)
                
                # Add legend element
                if model['name'] == "EISCAT UHF":
                    legend_elements.append(Line2D(
                        [0], [0], marker='o', color=model['color'], markeredgecolor='black',
                        markersize=8, linestyle='', label=model['name']
                    ))
                else:
                    legend_elements.append(Line2D(
                        [0], [0], marker='o', color=model['color'], markeredgecolor='black',
                        markersize=8, linestyle='', label=model['name']
                    ))
    
                # Add metrics for non-EISCAT models
                print(config['key'])
                if model['name'] != "EISCAT UHF":
                    
                    # print
                    print(model['name'])
                    
                    
                    metrics = _calculate_metrics(x, y)
                    
                    print(f'BC = {metrics["bc"]:.2f}')
                    print(f'KL = {metrics["kl"]:.2f}')
                    print("-----------------")
                    if config['key'] == 'e':
                        main_ax.text(0.58, 0.23 - 0.04*idx,
                                   rf"$\mathbf{{({metrics['r2']:.2f},\ {metrics['rmse']:.2f},\ {metrics['r_corr']:.2f})}}$",
                                   color=model['color'], fontsize=10, transform=main_ax.transAxes)
                        main_ax.text(0.58, 0.23,
                                   rf"$\mathbf{{(R^2,\ RMSE,\ r)}}$",
                                   color='black', fontsize=10, transform=main_ax.transAxes)
                    
                    else:
                        main_ax.text(0.64, 0.23 - 0.04*idx,
                                   rf"$\mathbf{{({metrics['r2']:.2f},\ {metrics['rmse']:.2f},\ {metrics['r_corr']:.2f})}}$",
                                   color=model['color'], fontsize=10, transform=main_ax.transAxes)
                        main_ax.text(0.64, 0.23,
                                   rf"$\mathbf{{(R^2,\ RMSE,\ r)}}$",
                                   color='black', fontsize=10, transform=main_ax.transAxes)
                
                # Plot KDEs if enabled
                if show_marginal_kde and top_ax and right_ax:
                    _plot_kdes(model, x, y, config['min'], config['max'], top_ax, right_ax)
            
            # Add legend
            main_ax.legend(handles=legend_elements, 
                          fontsize=9,
                          facecolor='white',
                          loc='upper left',
                          framealpha=1)
    
            # Configure axes limits and labels
            main_ax.set(xlim=(config['min'], config['max']), 
                       ylim=(config['min'], config['max']))
            main_ax.grid(True)

        
    
        #region Final Formatting
        axes['e'][0].set_xlabel(r'Prediction  $\log_{10}\,(n_e)$', fontsize=15)
        axes['e'][0].set_ylabel(r'EISCAT UHF  $\log_{10}\,(n_e)$', fontsize=15)
        axes['f'][0].set_xlabel(r'Prediction  $\log_{10}\,(n_e)$', fontsize=15)
        
        plt.show()
        
    
    def plot_kde_comparison(self, normalize=False):
        """Plot KDE comparisons between models and EISCAT UHF for both regions with optional normalization."""
        # Merge data dictionaries
        X_eis = merge_nested_peak_dict(self.X_EIS)['All']
        X_kian = merge_nested_peak_dict(self.X_KIAN)['All']
        X_ion = merge_nested_peak_dict(self.X_ION)['All']
        X_geo = merge_nested_peak_dict(self.X_GEO)['All']
        X_art = merge_nested_peak_dict(self.X_ART)['All']
        X_ech = merge_nested_peak_dict(self.X_ECH)['All']
    
        # Prepare peak density data
        eis_param_peak  = np.log10(X_eis["r_param_peak"])
        kian_param_peak = X_kian["r_param_peak"]
        ion_param_peak  = X_ion["r_param_peak"]
        geo_param_peak  = X_geo["r_param_peak"]
        art_param_peak  = np.log10(X_art["r_param_peak"])
        ech_param_peak  = np.log10(X_ech["r_param_peak"])
    
        # Define models with names, data, and colors
        models = [
            {"name": "EISCAT UHF", "data": eis_param_peak,  "color": "C0", "Z_order": 4, "ls": "--"},
            {"name": "KIAN-Net",    "data": kian_param_peak, "color": "C1", "Z_order": 5, "ls": "-"},
            {"name": "Artist 5.0",  "data": art_param_peak,  "color": "C2", "Z_order": 3, "ls": "-"},
            {"name": "E-CHAIM",     "data": ech_param_peak,  "color": "C3", "Z_order": 1, "ls": "-"},
            {"name": "Iono-CNN",    "data": ion_param_peak,  "color": "C4", "Z_order": 6, "ls": "-"},
            {"name": "Geo-DMLP",    "data": geo_param_peak,  "color": "C5", "Z_order": 2, "ls": "-"},
        ]
    
        # Create figure and subplots using gridspec
        with sns.axes_style("dark"):
            fig = plt.figure(figsize=(12, 7))
            gs = GridSpec(nrows=1, ncols=2, figure=fig)
            ax_e = fig.add_subplot(gs[0])
            ax_f = fig.add_subplot(gs[1], sharey=ax_e)
            ax_f.tick_params(labelleft=False)
    
        fig.suptitle('Peak Electron Density Distribution Comparison', fontsize=20, y=0.95)
    
        # Set x-axis limits based on normalization
        if normalize:
            Min_e, Max_e = 0, 1
            Min_f, Max_f = 0, 1
            x_label = "Normalized Electron Density"
        else:
            Min_e, Max_e = 8.5, 12.5
            Min_f, Max_f = 9.5, 13
            x_label = r"Electron Density $\log_{10}(n_e)$"
    
        # Plot KDEs for E- and F-regions
        self._plot_region_kde(ax_e, models, region=0, Min=Min_e, Max=Max_e, title='(a) E-region', normalize=normalize)
        self._plot_region_kde(ax_f, models, region=1, Min=Min_f, Max=Max_f, title='(b) F-region', normalize=normalize)
    
        # Formatting
        ax_e.set_ylabel('Probability Density', fontsize=15)
        ax_e.set_xlabel(x_label, fontsize=15)
        ax_f.set_xlabel(x_label, fontsize=15)
        for ax in (ax_e, ax_f):
            ax.grid(True)
            ax.legend(fontsize=11, loc='upper left')
    
        plt.tight_layout(rect=[0, 0, 1, 0.93])
        plt.show()
    
    def _plot_region_kde(self, ax, models, region, Min, Max, title, normalize=False):
        """Helper function to plot KDEs for a single region, compute metrics, and display them as text."""
        x_grid = np.linspace(Min, Max, 1000)
        metrics_list = []
    
        # If normalizing, compute global min and max for this region across all models
        if normalize:
            all_data = np.concatenate([model["data"][region, ~np.isnan(model["data"][region, :])] for model in models])
            global_min = np.min(all_data)
            global_max = np.max(all_data)
        else:
            global_min = None
            global_max = None
    
        # Compute the KDE for the reference model (EISCAT UHF)
        ref_pdf = None
        for model in models:
            if model["name"] == "EISCAT UHF":
                data_ref = model["data"][region, :]
                valid = ~np.isnan(data_ref)
                data_ref = data_ref[valid]
                if normalize and data_ref.size > 0:
                    data_ref = (data_ref - global_min) / (global_max - global_min)
                if len(data_ref) > 0:
                    kde_ref = gaussian_kde(data_ref, bw_method=0.3)
                    ref_pdf = kde_ref.evaluate(x_grid)
                else:
                    ref_pdf = np.zeros_like(x_grid)
                ax.plot(x_grid, ref_pdf,
                        color=model["color"],
                        linestyle=model["ls"],
                        linewidth=3,
                        alpha=1,
                        zorder=model["Z_order"],
                        label=model["name"])
                break
    
        if ref_pdf is None:
            raise ValueError(f"Reference (EISCAT UHF) data is missing for region {region}")
    
        # Compute KDE and metrics for each candidate model
        for model in models:
            if model["name"] == "EISCAT UHF":
                continue
            data = model["data"][region, :]
            valid = ~np.isnan(data)
            data = data[valid]
            if len(data) == 0:
                continue
            if normalize:
                data = (data - global_min) / (global_max - global_min)
            kde = gaussian_kde(data, bw_method=0.3)
            pdf = kde.evaluate(x_grid)
    
            # Bhattacharyya Coefficient
            bc = np.trapz(np.sqrt(ref_pdf * pdf), x_grid)
    
            # KL-Divergence
            epsilon = 1e-10
            kl = np.trapz(pdf * np.log((pdf + epsilon) / (ref_pdf + epsilon)), x_grid)
    
            # Wasserstein Distance
            dx = x_grid[1] - x_grid[0]
            cdf_ref = np.cumsum(ref_pdf) * dx
            cdf_model = np.cumsum(pdf) * dx
            wasserstein = np.trapz(np.abs(cdf_ref - cdf_model), x_grid)
    
            # Store metrics
            metrics_list.append({
                "color": model["color"],
                "bc": bc,
                "kl": kl,
                "wasserstein": wasserstein
            })
    
            # Plot the model PDF
            ax.plot(x_grid, pdf,
                    color=model["color"],
                    linestyle=model["ls"],
                    linewidth=3,
                    alpha=1,
                    zorder=model["Z_order"],
                    label=model["name"])
    
        # Add header and metrics as text in the top right
        ax.text(0.64, 0.98, "(BC, KL, W)",
                transform=ax.transAxes,
                color='black',
                fontsize=13,
                fontweight='bold',
                horizontalalignment='left',
                verticalalignment='top')
        
        ls = 0.045
        for i, metric in enumerate(metrics_list):
            y_pos = 0.98 - ls * (i + 1)
            ax.text(0.64, y_pos, f"({metric['bc']:.2f}, {metric['kl']:.2f}, {metric['wasserstein']:.2f})",
                    transform=ax.transAxes,
                    color=metric['color'],
                    fontsize=13,
                    fontweight='bold',
                    horizontalalignment='left',
                    verticalalignment='top')
    
        # Set plot limits and title
        ax.set_xlim(Min, Max)
        ax.set_ylim(0, 2)
        ax.set_title(title, fontsize=17, pad=12)
        
    def plot_combined_peak_densities(self, normalize=False):
        """Plot a 2x2 grid comparing peak electron densities scatter and KDE distributions."""
        #region Data Preparation
        def _prepare_models():
            """Prepare model data with consistent transformations."""
            model_configs = [
                ('EIS', 'EISCAT UHF', "C0", 0, np.log10, "--"),
                ('KIAN', 'KIAN-Net', "C1", 3, None, "-"),
                ('ART', 'Artist 5.0', "C2", 4, np.log10, "-"),
                ('ECH', 'E-CHAIM', "C3", 5, np.log10, "-"),
                ('ION', 'Iono-CNN', "C4", 2, None, "-"),
                ('GEO', 'Geo-DMLP', "C5", 1, None, "-"),
            ]
            models = []
            for key, name, color, z_order, transform, ls in model_configs:
                data = merge_nested_peak_dict(getattr(self, f'X_{key}'))['All']["r_param_peak"]
                if transform:
                    data = transform(data)
                models.append({
                    'name': name, 'data': data, 'color': color,
                    'z_order': z_order, 'ls': ls
                })
            return models
    
        models = _prepare_models()
        eis_data = models[0]['data']
        #endregion
    
        #region Figure Setup
        with sns.axes_style("dark"):
            fig = plt.figure(figsize=(12, 12.5))
            gs = GridSpec(2, 2, wspace=0.1, hspace=0.2)
            ax_scatter_e = fig.add_subplot(gs[0, 0])
            ax_scatter_f = fig.add_subplot(gs[0, 1])
            ax_kde_e = fig.add_subplot(gs[1, 0], sharex=ax_scatter_e)
            ax_kde_f = fig.add_subplot(gs[1, 1], sharex=ax_scatter_f, sharey=ax_kde_e)
            ax_kde_f.tick_params(labelleft=False)
    
        fig.suptitle('Peak Electron Density Comparisons\n(EISCAT UHF vs Models)', fontsize=20, y=0.97)
        #endregion
    
        #region Helper Functions
        def _plot_diagonal(ax, min_val, max_val):
            """Plot perfect agreement diagonal."""
            ax.plot([min_val, max_val], [min_val, max_val], color="C0", linewidth=3, zorder=1)
    
        def _plot_model_points(ax, model, x, y):
            """Plot data points for a single model."""
            if model['name'] == "EISCAT UHF":
                ax.plot(x, y, color=model['color'], marker='o', mec='black',
                        linestyle='', markersize=5, zorder=model['z_order'])
            else:
                ax.scatter(x, y, s=40, color=model['color'],
                           edgecolors='black', zorder=model['z_order'])
    
        def _calculate_metrics(x, y):
            """Calculate performance metrics including BC and KL-Divergence."""
            metrics = {
                'r2': self.compute_r2_score(y, x),
                'rmse': np.sqrt(np.mean((y - x)**2)),
                'r_corr': pearsonr(x, y)[0]
            }
            min_val = min(np.min(x), np.min(y))
            max_val = max(np.max(x), np.max(y))
            x_grid = np.linspace(min_val, max_val, 1000)
            kde_eiscat = gaussian_kde(y)
            kde_model = gaussian_kde(x)
            p = kde_eiscat(x_grid)
            q = kde_model(x_grid)
            metrics['bc'] = trapz(np.sqrt(p * q), x_grid)
            q_safe = np.where(q < 1e-10, 1e-10, q)
            metrics['kl'] = trapz(p * np.log(p / q_safe), x_grid)
            return metrics
    
        def _plot_kde(ax, models, region, min_val, max_val, normalize=False):
            """Plot KDEs for a single region with metrics."""
            x_grid = np.linspace(min_val, max_val, 1000)
            metrics_list = []
    
            # Normalize data if requested
            if normalize:
                all_data = np.concatenate([model["data"][region, ~np.isnan(model["data"][region, :])]
                                            for model in models])
                global_min = np.min(all_data)
                global_max = np.max(all_data)
            else:
                global_min = global_max = None
    
            # Plot reference (EISCAT UHF)
            ref_pdf = None
            for model in models:
                if model["name"] == "EISCAT UHF":
                    data = model["data"][region, ~np.isnan(model["data"][region, :])]
                    if normalize and data.size > 0:
                        data = (data - global_min) / (global_max - global_min)
                    if data.size > 0:
                        kde = gaussian_kde(data, bw_method=0.3)
                        ref_pdf = kde(x_grid)
                        ax.plot(x_grid, ref_pdf, color=model["color"], linestyle=model["ls"],
                                linewidth=3, zorder=model["z_order"], label=model["name"])
                    break
    
            if ref_pdf is None:
                raise ValueError(f"EISCAT UHF data missing for region {region}")
    
            # Plot other models and compute metrics
            for model in models:
                if model["name"] == "EISCAT UHF":
                    continue
                data = model["data"][region, ~np.isnan(model["data"][region, :])]
                if data.size == 0:
                    continue
                if normalize:
                    data = (data - global_min) / (global_max - global_min)
                kde = gaussian_kde(data, bw_method=0.3)
                pdf = kde(x_grid)
                ax.plot(x_grid, pdf, color=model["color"], linestyle=model["ls"],
                        linewidth=3, zorder=model["z_order"], label=model["name"])
    
                # Compute metrics
                bc = trapz(np.sqrt(ref_pdf * pdf), x_grid)
                epsilon = 1e-10
                kl = trapz(pdf * np.log((pdf + epsilon) / (ref_pdf + epsilon)), x_grid)
                dx = x_grid[1] - x_grid[0]
                cdf_ref = np.cumsum(ref_pdf) * dx
                cdf_model = np.cumsum(pdf) * dx
                wasserstein = trapz(np.abs(cdf_ref - cdf_model), x_grid)
                metrics_list.append({"color": model["color"], "bc": bc, "kl": kl, "wasserstein": wasserstein})
    
            # Add metrics text to the plot
            ax.text(0.64, 0.98, "(BC, KL, W)", transform=ax.transAxes, color='black',
                    fontsize=13, fontweight='bold', ha='left', va='top')
            for i, metric in enumerate(metrics_list):
                y_pos = 0.98 - 0.045 * (i + 1)
                ax.text(0.64, y_pos, f"({metric['bc']:.2f}, {metric['kl']:.2f}, {metric['wasserstein']:.2f})",
                        transform=ax.transAxes, color=metric['color'], fontsize=13, fontweight='bold',
                        ha='left', va='top')
    
            ax.set_xlim(min_val, max_val)
            ax.set_ylim(0, 2)
            # Removed automatic title here; you can now manually add a title later if desired.
        #endregion
    
        #region Plotting Logic
        # Removed the "title" keys from region_configs since we're setting titles manually later
        region_configs = [
            {'key': 'e', 'scatter_ax': ax_scatter_e, 'kde_ax': ax_kde_e, 'index': 0,
             'min': 8.5 if not normalize else 0, 'max': 12.5 if not normalize else 1},
            {'key': 'f', 'scatter_ax': ax_scatter_f, 'kde_ax': ax_kde_f, 'index': 1,
             'min': 9.5 if not normalize else 0, 'max': 12.5 if not normalize else 1}
        ]
    
        for config in region_configs:
            # Scatter plots (first row)
            scatter_ax = config['scatter_ax']
            _plot_diagonal(scatter_ax, config['min'], config['max'])
            # Removed the automated scatter plot title below so you can add custom titles later:
            # scatter_ax.text(0.3, 0.9, f"{config['title']} Peaks", fontsize=15, transform=scatter_ax.transAxes)
            legend_elements = []
    
            for idx, model in enumerate(models):
                valid = ~np.isnan(model['data'][config['index']]) & ~np.isnan(eis_data[config['index']])
                x = model['data'][config['index'], valid]
                y = eis_data[config['index'], valid]
                _plot_model_points(scatter_ax, model, x, y)
    
                # Add legend element
                legend_elements.append(Line2D([0], [0], marker='o', color=model['color'],
                                              markeredgecolor='black', markersize=8, linestyle='',
                                              label=model['name']))
    
                # Add metrics for non-EISCAT models
                if model['name'] != "EISCAT UHF":
                    metrics = _calculate_metrics(x, y)
                    x_pos = 0.58 if config['key'] == 'e' else 0.64
                    scatter_ax.text(x_pos, 0.23 - 0.04 * idx,
                                    rf"$\mathbf{{({metrics['r2']:.2f},\ {metrics['rmse']:.2f},\ {metrics['r_corr']:.2f})}}$",
                                    color=model['color'], fontsize=10, transform=scatter_ax.transAxes)
                    scatter_ax.text(x_pos, 0.23, rf"$\mathbf{{(R^2,\ RMSE,\ r)}}$",
                                    color='black', fontsize=10, transform=scatter_ax.transAxes)
    
            scatter_ax.legend(handles=legend_elements, fontsize=9, facecolor='white',
                              loc='upper left', framealpha=1)
            scatter_ax.set(xlim=(config['min'], config['max']), ylim=(config['min'], config['max']))
            scatter_ax.grid(True)
    
            # KDE plots (second row)
            _plot_kde(config['kde_ax'], models, config['index'], config['min'], config['max'],
                      normalize=normalize)
            config['kde_ax'].grid(True)
            config['kde_ax'].legend(fontsize=9, facecolor='white', loc='upper left', framealpha=1)
    
    
        #region Final Formatting
        ax_scatter_e.set_ylabel(r'EISCAT UHF $\log_{10}\,(n_e)$', fontsize=16)
        ax_scatter_f.tick_params(labelleft=False)
        ax_scatter_e.set_title('(a) E-region Model vs EISCAT', fontsize=18)
        ax_scatter_f.set_title('(b) F-region Model vs EISCAT', fontsize=18)
        ax_kde_e.set_title('(c) E-region KDE', fontsize=18)
        ax_kde_f.set_title('(d) F-region KDE', fontsize=18)
        ax_kde_e.set_xlabel("Normalized Electron Density" if normalize else r"Electron Density $\log_{10}(n_e)$", fontsize=15)
        ax_kde_e.set_ylabel('Probability Density', fontsize=16)
        ax_kde_f.set_xlabel("Normalized Electron Density" if normalize else r"Electron Density $\log_{10}(n_e)$", fontsize=15)
        plt.show()
        #endregion
        
        

