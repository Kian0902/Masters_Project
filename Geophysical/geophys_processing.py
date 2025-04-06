# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 13:32:17 2024

@author: Kian Sartipzadeh
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from geophys_utils import create_samples_csv
import seaborn as sns



class GeophysProcessor:
    def __init__(self, dataset):
        self.dataset = dataset
        self.data = self._load_data()
    
    def _load_data(self):
        """Load and prepare the dataset, setting DateTime as the index."""
        data = self.dataset
        data['DateTime'] = pd.to_datetime(data['DateTime'], format='%d-%b-%Y %H:%M:%S')
        data.set_index('DateTime', inplace=True)
        return data
    
    def get_data_between(self, start_date, end_date):
        """
        Get data between specified start and end datetime.
        
        Args:
            start_date (str): Start date in format 'YYYY-MM-DD'.
            end_date (str): End date in format 'YYYY-MM-DD'.
        Returns:
            pd.DataFrame: Sliced data.
        """
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        if start not in self.data.index or end not in self.data.index:
            raise ValueError("Start or end date is out of range of the dataset.")
        return self.data.loc[start:end]
    
    def _plot_corr_on_ax(self, data, ax, exclude_features=None):
        """
        Plot the correlation matrix on the specified axes, excluding specified features.
        
        Args:
            data (pd.DataFrame): Dataset to compute correlations from.
            ax (matplotlib.axes.Axes): Axes to plot on.
            annot (bool): Whether to annotate the heatmap with correlation values.
            exclude_features (list of str, optional): Features to exclude from the correlation matrix.
        
        Raises:
            KeyError: If any feature in exclude_features is not in the dataset.
            ValueError: If fewer than two features remain after exclusion.
        """
        # Filter out excluded features if provided
        if exclude_features:
            data = data.drop(columns=exclude_features, errors='raise')
        
        corr_matrix = data.corr()
        if corr_matrix.shape[0] < 2:
            raise ValueError("At least two features are required to plot a correlation matrix.")
        
        
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(
            corr_matrix,
            # mask=mask,
            annot=False,
            fmt='.2f',
            cmap='bwr_r',
            vmin=-1, vmax=1,
            linewidths=0.0,
            # cbar_kws={'label': 'Correlation Coefficient'},
            ax=ax
        )
        # ax.set_title('(a) Correlation Matrix', fontsize=17, y=1.02)
    
    def _plot_hist_group(self, data, group, ax):
        """
        Plot histograms for a group of features on the specified axes with y-label and ticks on the right.
        
        Args:
            data (pd.DataFrame): Dataset to plot histograms from.
            group (list of str): List of feature names to plot.
            ax (matplotlib.axes.Axes): Axes to plot on.
        """
        for feature in group:
            ax.hist(data[feature], alpha=0.5, label=feature, density=True, bins=30)
        ax.legend()
        ax.yaxis.tick_right()  # Move y-axis ticks to the right
    
    def plot_correlation_and_histograms(self, start_date=None, end_date=None, exclude_features=None):
        """
        Create a 4x2 grid plot with a correlation matrix in the first column and histograms of specified feature groups in the second column.
        
        Args:
            start_date (str, optional): Start date in 'YYYY-MM-DD' format. If None, uses full dataset.
            end_date (str, optional): End date in 'YYYY-MM-DD' format. If None, uses full dataset.
            annot (bool): Whether to annotate the correlation matrix with values. Default is False.
            exclude_features (list of str, optional): Features to exclude from the correlation matrix.
        
        Raises:
            ValueError: If any required histogram features are missing from the dataset or if too few features remain for correlation.
            KeyError: If any feature in exclude_features is not in the dataset.
        """
        # Define histogram feature groups
        hist_groups = [
            ['SME', 'SML', 'SMU'],
            ['SYM_D', 'SYM_H', 'ASY_D', 'ASY_H'],
            ['Sunspot', 'f107', 'Lyman', 'ap_index'],
            ['mean_X', 'mean_Y', 'mean_Z']
        ]
        
        # Get the data
        if start_date and end_date:
            data = self.get_data_between(start_date, end_date)
        else:
            data = self.data
        
        # Check for required features for histograms
        required_features = set(feature for group in hist_groups for feature in group)
        missing_features = required_features - set(data.columns)
        if missing_features:
            raise ValueError(f"Missing features required for histograms: {missing_features}")
        
        # Create figure and gridspec
        fig = plt.figure(figsize=(14, 8))
        gs = fig.add_gridspec(4, 2, width_ratios=[1, 0.5], wspace=0.1)
        
        ax_corr = fig.add_subplot(gs[:, 0])  # Correlation matrix spans all rows in first column
        ax_corr.set_title('(a) Correlation Matrix', fontsize=17, y=1.045)
        ax_hist1 = fig.add_subplot(gs[0, 1])  # First histogram
        ax_hist2 = fig.add_subplot(gs[1, 1])  # Second histogram
        ax_hist3 = fig.add_subplot(gs[2, 1])  # Third histogram
        ax_hist4 = fig.add_subplot(gs[3, 1])  # Fourth histogram
        
        self._plot_corr_on_ax(data, ax_corr, exclude_features=exclude_features)
        self._plot_hist_group(data, hist_groups[0], ax_hist1)
        ax_hist1.set_title('(b) Geophysical Parameters', fontsize=17, y=1.2)
        self._plot_hist_group(data, hist_groups[1], ax_hist2)
        self._plot_hist_group(data, hist_groups[2], ax_hist3)
        self._plot_hist_group(data, hist_groups[3], ax_hist4)
        ax_hist4.set_xlabel("Normalized Values", fontsize=12)

        plt.tight_layout()
        plt.show()

    def plot_time_series(self, start_date, end_date):
        """
        Plot geophysical features over time in a 5x1 grid.
        
        Args:
            start_date (str): Start date in format 'YYYY-MM-DD'.
            end_date (str): End date in format 'YYYY-MM-DD'.
        
        Raises:
            ValueError: If no data is available for the specified range or if required features are missing.
        """
        # Fetch data for the specified date range
        data = self.get_data_between(start_date, end_date)
        if data.empty:
            raise ValueError("No data available for the specified date range.")
            
            
            
        # Define feature groups with larger time scale, labels, colors, and line styles
        feature_groups = [
            ['Sunspot', 'f107', 'Lyman'],
            ['DOY_sin', 'DOY_cos', 'SolarZenithAngle'],
            
        ]
        
        ylabels = [
            'Solar Indices',
            'Day of Year',
            
        ]
        
        color = [
            ["red", "black", "lightblue"],
            ["C1", "C2", "C0"],
            
        ]
        
        line_style = [
            ["-", "-", "-"],
            ["-", "-", "-"],
            
        ]
        
        z_order = [
            [1, 2, 3],
            [2, 3, 1],
            
        ]
        
        
        
        # Define feature groups with shorter time scale, labels, colors, and line styles
        # feature_groups = [
        #     ['mean_X', 'mean_Y', 'mean_Z'],
        #     ['SYM_D', 'SYM_H', 'ASY_D', 'ASY_H'],
        #     ['SME', 'SML', 'SMU'],
        #     ['ap_index'],
        #     ['TOD_sin', 'TOD_cos']
        # ]
        
        # ylabels = [
        #     'B-field (x,y,z)',
        #     'SYM_D/H and ADY_D/H',
        #     'SM_E/L/U',
        #     'ap_index',
        #     'TOD_sin/cos',
        # ]
        
        # color = [
        #     ["C0", "C2", "C1"],
        #     ["red", "purple", "red", "purple"],
        #     ["C0", "C2", "C1"],
        #     ['C0'],
        #     ['C1', 'C2']
        # ]
        
        # line_style = [
        #     ["-", "-", "-"],
        #     ["-", "-", ":", ":"],
        #     ["-", "-", "-"],
        #     ['-'],
        #     ['-', '-']
        # ]
        
        # Check for missing features in the data
        all_features = [feature for group in feature_groups for feature in group]
        missing_features = [feature for feature in all_features if feature not in data.columns]
        if missing_features:
            raise ValueError(f"Missing features: {missing_features}")
        

        # Create a figure with 5 subplots stacked vertically
        fig, axes = plt.subplots(2, 1, figsize=(12, 6), height_ratios=[1,0.5], sharex=True)
        fig.suptitle("Normalized Geophysical state parameters", fontsize=17)
        
        
        for i, (group, ylabel, col, style, z_ord) in enumerate(zip(feature_groups, ylabels, color, line_style, z_order)):
            ax = axes[i]
            ax.set_facecolor('ghostwhite')
            for feature, c, sty, zo in zip(group, col, style, z_ord):
                ax.plot(data.index, data[feature], label=feature, color=c, linestyle=sty, linewidth=2, zorder=zo)
            ax.legend(loc='upper center', ncol=len(group), facecolor="white")
            ax.set_ylabel(ylabel,fontsize=13)
            ax.grid(True, alpha=0.2)
        
        # Set x-axis label and rotate tick labels for readability
        axes[-1].set_xlabel('Time (UT)', fontsize=12)
        plt.setp(axes[-1].xaxis.get_majorticklabels(), rotation=45, ha='center')
        
        # Adjust layout to prevent overlap
        plt.tight_layout()
        
        # Display the plot
        plt.show()


    def plot_multi_scale_features(self, start_date, end_date):
        """
        Plot two figures: large-scale features (2012-2022) and short-scale features (user-defined interval).
        The large-scale plot highlights the user-defined interval.
        
        Args:
            start_date (str): Start date in format 'YYYY-MM-DD' for short-scale plot.
            end_date (str): End date in format 'YYYY-MM-DD' for short-scale plot.
        
        Raises:
            ValueError: If dates are out of range or data is unavailable.
        """
        # Define fixed date range for large-scale plot
        large_start = '2012-01-01'
        large_end = '2022-12-31'
    
        # Fetch data
        large_data = self.get_data_between(large_start, large_end)
        short_data = self.get_data_between(start_date, end_date)
    
        # Large-scale plot definitions
        feature_groups_large = [
            ['Sunspot', 'f107', 'Lyman'],
            ['DOY_sin', 'DOY_cos', 'SolarZenithAngle'],
        ]
        ylabels_large = [
            'Solar Indices',
            'Day of Year',
        ]
        color_large = [
            ["red", "black", "lightblue"],
            ["C1", "C2", "C0"],
        ]
        line_style_large = [
            ["-", "-", "-"],
            ["-", "-", "-"],
        ]
        z_order_large = [
            [1, 2, 3],
            [2, 3, 1],
        ]
    
        # Short-scale plot definitions
        feature_groups_short = [
            ['mean_X', 'mean_Y', 'mean_Z'],
            ['SYM_D', 'SYM_H', 'ASY_D', 'ASY_H','ap_index'],
            ['SME', 'SML', 'SMU'],
            # [],
        ]
        ylabels_short = [
            'Local\nMagnetometer',
            'Magnetic\ndisturbance',
            'Auroral\nelectrojet',
            # 'Geomagnetic\ndisturbance',
        ]
        color_short = [
            ["C0", "C2", "C1"],
            ["red", "purple", "red", "purple", "black"],
            ["royalblue", "brown", "darkorange"],
            # ['C3'],
        ]
        line_style_short = [
            ["-", "-", "-"],
            ["-", "-", ":", ":","-"],
            ["-", "-", "-"],
        ]
        z_order_short = [
            [1, 2, 3],
            [5, 4, 3, 2, 1],
            [1, 2, 3],
        ]
        
        # Create large-scale figure with 2 subplots
        fig_large, axes_large = plt.subplots(2, 1, figsize=(11, 5), height_ratios=[1, 0.8], sharex=True)
        fig_large.suptitle("(a) Large-scale Geophysical Parameters (2012-2022)", fontsize=17, y=0.98)
        for i, (group, ylabel, col, style, z_ord) in enumerate(zip(feature_groups_large, ylabels_large, color_large, line_style_large, z_order_large)):
            ax = axes_large[i]
            ax.set_facecolor('ghostwhite')
            for feature, c, sty, zo in zip(group, col, style, z_ord):
                ax.plot(large_data.index, large_data[feature], label=feature, color=c, linestyle=sty, linewidth=2, zorder=zo)
            
            # ax.legend(loc='upper left', ncol=len(group), facecolor="white")
            if i == 0:
                ax.axvspan(pd.to_datetime(start_date), pd.to_datetime(end_date), color='gray', alpha=0.5, label='Segment')
                add = 1
            else:
                ax.axvspan(pd.to_datetime(start_date), pd.to_datetime(end_date), color='gray', alpha=0.5)
                add = 0
            ax.legend(loc='upper left', ncol=len(group)+add, facecolor="white")
            ax.set_ylabel(ylabel, fontsize=13)
            ax.grid(True, alpha=0.2)
            
            # Highlight user-defined interval
            # ax.axvspan(pd.to_datetime(start_date), pd.to_datetime(end_date), color='gray', alpha=0.5, label='Segment')
        axes_large[-1].set_xlabel('Time [UT]', fontsize=10.5)
        plt.tight_layout()
        plt.setp(axes_large[-1].xaxis.get_majorticklabels(), rotation=45, ha='center')
    
        # Create short-scale figure with 5 subplots
        fig_short, axes_short = plt.subplots(3, 1, figsize=(12, 6.5), sharex=True)
        fig_short.suptitle(f"(b) Short-scale Geophysical Parameters\n({start_date} to {end_date})", fontsize=16, y=0.96)
        for i, (group, ylabel, col, style, z_ord_sh) in enumerate(zip(feature_groups_short, ylabels_short, color_short, line_style_short, z_order_short)):
            ax = axes_short[i]
            ax.set_facecolor('ghostwhite')
            for feature, c, sty, zo_sh in zip(group, col, style, z_ord_sh):
                ax.plot(short_data.index, short_data[feature], label=feature, color=c, linestyle=sty, linewidth=2, zorder=zo_sh)
            
            if i == 1:
                ax.legend(ncol=len(group)-2, loc='upper left')
            else:
                ax.legend(ncol=len(group), loc='upper left')
            ax.set_ylabel(ylabel, fontsize=13)
            ax.grid(True, alpha=0.2)
            
        axes_short[-1].set_xlabel('Time [UT]')
        axes_short[-1].xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        plt.setp(axes_short[-1].xaxis.get_majorticklabels(), rotation=45, ha='center')
    
        # Display both figures
        plt.show()

    def plot_feature_grid(self):
        """
        Plot a 5x5 grid where each cell displays a shortened version of a geophysical parameter name.
        """
        # List of the 25 geophysical parameters in the desired order
        features = [
            "DOY_sin", "DOY_cos", "TOD_sin", "TOD_cos", "SolarZenithAngle",
            "SME", "SML", "SMU", "SYM_D", "SYM_H",
            "ASY_D", "ASY_H", "Sunspot", "f107", "Lyman",
            "ap_index", "mean_X", "mean_Y", "mean_Z", "min_X",
            "min_Y", "min_Z", "max_X", "max_Y", "max_Z"
        ]
        
        # Dictionary mapping full feature names to their shortened versions
        # Each key maps to a tuple: (shortened name, background color)
        short_names = {
            "DOY_sin": ("DSin", "goldenrod"),
            "DOY_cos": ("DCos", "goldenrod"),
            "TOD_sin": ("TSin", "gold"),
            "TOD_cos": ("TCos", "gold"),
            "SolarZenithAngle": ("SolZenith", "yellow"),
            "SME": ("SME", "green"),
            "SML": ("SML", "yellowgreen"),
            "SMU": ("SMU", "greenyellow"),
            "SYM_D": ("SYM_D", "tomato"),
            "SYM_H": ("SYM_H", "coral"),
            "ASY_D": ("ASY_D", "darkorange"),
            "ASY_H": ("ASY_H", "orange"),
            "Sunspot": ("R", "lavender"),
            "f107": ("f10.7", "lightsteelblue"),
            "Lyman": ("Lyman", "powderblue"),
            "ap_index": ("ap", "darkcyan"),
            "mean_X": (r"$\mathbf{\overline{B}_X}$", "mediumorchid"),
            "mean_Y": (r"$\mathbf{\overline{B}_Y}$", "violet"),
            "mean_Z": (r"$\mathbf{\overline{B}_Z}$", "plum"),
            "min_X": (r"$\mathbf{min\,B_X}$", "C0"),
            "min_Y": (r"$\mathbf{min\,B_Y}$", "cornflowerblue"),
            "min_Z": (r"$\mathbf{min\,B_Z}$", "royalblue"),
            "max_X": (r"$\mathbf{max\,B_X}$", "brown"),
            "max_Y": (r"$\mathbf{max\,B_Y}$", "indianred"),
            "max_Z": (r"$\mathbf{max\,B_Z}$", "lightcoral")
        }
        
        # Create a 5x5 grid of subplots with no spacing between cells
        fig, axs = plt.subplots(5, 5, figsize=(10, 10), gridspec_kw={'wspace': 0, 'hspace': 0})
        # fig.suptitle("Geophysical Parameters", fontsize=16)
        fig.patch.set_visible(False)
        # Populate each subplot with the corresponding shortened feature name and background color
        for i, feature in enumerate(features):
            row = i // 5
            col = i % 5
            ax = axs[row, col]
            # Extract the short name and the color for the feature
            short_text, color = short_names[feature]
            # Set the background color of the cell
            ax.set_facecolor(color)
            # Place the shortened feature name text in the center
            ax.text(0.5, 0.5, short_text, fontsize=25, ha='center', va='center', weight='bold')
            ax.set_xticks([])
            ax.set_yticks([])
            # Draw cell borders for clarity
            for spine in ax.spines.values():
                spine.set_visible(True)
        
        # Use the full figure area for the grid
        plt.subplots_adjust(left=0, right=1, top=0.95, bottom=0)
        plt.show()


"""
Features:
    
DOY_sin, DOY_cos, TOD_sin,TOD_cos,
SolarZenithAngle,
SME,SML,SMU,
SYM_D,SYM_H,
ASY_D,ASY_H,
Sunspot,
f107,
ap_index,
Lyman,
mean_X,mean_Y,mean_Z,
min_X,min_Y,min_Z,
max_X,max_Y,max_Z
"""


file_path = 'data_normalized/data_normalized.csv'
data = pd.read_csv(file_path)


# processor = GeophysProcessor(data)
# processor.plot_correlation_and_histograms(start_date='2012-01-01', end_date='2022-12-31',
#                                           exclude_features=['DOY_sin', 'DOY_cos', 'TOD_sin','TOD_cos', 'SolarZenithAngle',
#                                                             'min_X', 'min_Y', 'min_Z', 'max_X', 'max_Y', 'max_Z'])



processor = GeophysProcessor(data)
# processor.plot_feature_grid()
# processor.plot_time_series(start_date='2012-01-01', end_date='2022-12-31')
processor.plot_multi_scale_features(start_date='2016-11-21', end_date='2016-11-29')
