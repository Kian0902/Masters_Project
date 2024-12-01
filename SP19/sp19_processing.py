# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 13:32:17 2024

@author: Kian Sartipzadeh
"""

# custom_header=['DoY/366', 'ToD/1440', 'Solar_Zenith/44', 'Kp', 'R', 'Dst',
#                 'ap', 'F10_7', 'AE', 'AL', 'AU', 'PC_potential', 'Lyman_alpha',
#                 'Bx', 'By', 'Bz', 'dBx', 'dBy', 'dBz']





import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from geophys_utils import filename_to_datetime

from scipy.signal import medfilt

import seaborn as sns
sns.set(style="dark", context=None, palette=None)



class GeophysProcessing:
    def __init__(self, data, datetimes, feature_names):
        """
        Initializes the GeophysProcessing object.

        Args:
            data (numpy.ndarray): The dataset of shape (N, 19).
            datetimes (list of datetime): List of datetime objects corresponding to the data samples.
            feature_names (list of str): List of feature names.
        """
        self.data = data
        self.datetimes = datetimes
        self.feature_names = feature_names
        self.df = self._create_dataframe()

    def _create_dataframe(self):
        """
        Creates a pandas DataFrame from the data, datetimes, and feature names.

        Returns:
            pd.DataFrame: The resulting DataFrame.
        """
        df = pd.DataFrame(self.data, columns=self.feature_names)
        df['datetime'] = self.datetimes
        df.set_index('datetime', inplace=True)
        return df

    def filter_by_time_period(self, start_time, end_time):
        """
        Filters the DataFrame for data within the specified time period.

        Args:
            start_time (datetime): The start of the time period.
            end_time (datetime): The end of the time period.

        Returns:
            pd.DataFrame: The filtered DataFrame.
        """
        return self.df.loc[start_time:end_time]

    def visualize_features(self, start_time, end_time, features=None):
        """
        Visualizes specified features over the specified time period.

        Args:
            start_time (datetime): The start of the time period.
            end_time (datetime): The end of the time period.
            features (list of str): The features to visualize. If None, all features are plotted.
        """
        filtered_df = self.filter_by_time_period(start_time, end_time)

        if features is None:
            features = self.feature_names

        filtered_df[features].plot(subplots=True, figsize=(10, 5), title="Feature Trends Over Time")
        plt.tight_layout()
        plt.grid(True)
        plt.show()

    def resample_data(self, rule, aggregation_func='mean'):
        """
        Resamples the DataFrame to a different time resolution.

        Args:
            rule (str): The resampling rule (e.g., 'H' for hourly, 'D' for daily).
            aggregation_func (str or callable): Aggregation function to apply during resampling.

        Returns:
            pd.DataFrame: The resampled DataFrame.
        """
        return self.df.resample(rule).apply(aggregation_func)

    def visualize_resampled(self, rule, features=None, aggregation_func='mean'):
        """
        Visualizes resampled data for the specified features.

        Args:
            rule (str): The resampling rule (e.g., 'H' for hourly, 'D' for daily).
            features (list of str): The features to visualize. If None, all features are plotted.
            aggregation_func (str or callable): Aggregation function to apply during resampling.
        """
        resampled_df = self.resample_data(rule, aggregation_func)

        if features is None:
            features = self.feature_names

        resampled_df[features].plot(subplots=True, figsize=(10, 5), title=f"Resampled Feature Trends ({rule})")
        plt.tight_layout()
        plt.grid(True)
        plt.show()
















