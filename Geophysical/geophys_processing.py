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





class FeatureFilter:
    def __init__(self, geophys_processor):
        """
        Initializes the FeatureFilter object.

        Args:
            geophys_processor (GeophysProcessing): The GeophysProcessing object containing the dataset.
        """
        self.geophys_processor = geophys_processor
        self.df = geophys_processor.df

    def apply_median_filter(self, feature, threshold, window_size):
        """
        Applies a median filter to a specified feature, replacing outliers with the local median.

        Args:
            feature (str): The feature to filter.
            threshold (float): The threshold for outlier detection.
            window_size (int): The size of the window for the median filter.
        """
        if feature not in self.geophys_processor.feature_names:
            raise ValueError(f"Feature '{feature}' not found in the dataset.")

        # Extract the feature data
        feature_data = self.df[feature].copy()

        # Identify outliers
        median_filtered = medfilt(feature_data, kernel_size=window_size)
        outliers = np.abs(feature_data - median_filtered) > threshold

        # Replace outliers with local median
        feature_data[outliers] = median_filtered[outliers]

        # Update the DataFrame
        self.df[feature] = feature_data

    def update_geophys_processor(self):
        """
        Updates the GeophysProcessing object with the filtered data.
        """
        self.geophys_processor.df = self.df

    def visualize_filtered_feature(self, feature, start_time, end_time):
        """
        Visualizes the specified feature after filtering over the given time period.

        Args:
            feature (str): The feature to visualize.
            start_time (datetime): The start of the time period.
            end_time (datetime): The end of the time period.
        """
        filtered_df = self.geophys_processor.filter_by_time_period(start_time, end_time)

        plt.figure(figsize=(10, 5))
        plt.plot(filtered_df.index, filtered_df[feature], label=f"{feature} (Filtered)")
        plt.title(f"Filtered {feature} from {start_time} to {end_time}")
        plt.xlabel("Time")
        plt.ylabel("Value")
        plt.legend()
        plt.grid()
        plt.show()



# class GeophysProcessing:
    
#     def __init__(self, dataset):
#         self.dataset = dataset
#         self.feature_names = ['DoY/366', 'ToD/1440', 'Solar_Zenith/44', 'Kp', 'R', 'Dst',
#                             'ap', 'F10_7', 'AE', 'AL', 'AU', 'PC_potential', 'Lyman_alpha',
#                             'Bx', 'By', 'Bz', 'dBx', 'dBy', 'dBz']
    
    
#     def check_quality(self):
        
#         # Convert dataset to pandas DataFrame
#         df = pd.DataFrame(self.dataset, columns=self.feature_names)
        
#         if df.isnull().values.any():
#             print("Bad values detected!")
            
#             # Check for NaN, inf, and None values
#             nan_count = df.isna().sum()                    # Count NaN values
#             inf_count = df.isin([np.inf, -np.inf]).sum()   # Count inf values
#             none_count = df.isin([None]).sum()             # Count None values
    
            
#             # Combine results into a summary DataFrame
#             quality_summary = pd.DataFrame({
#                 'Feature': self.feature_names,
#                 'NaN Count': nan_count.values,
#                 'Inf Count': inf_count.values,
#                 'None Count': none_count.values
#             })
    
#             # Print the results
#             print("\nDataset Quality Summary:")
#             print(quality_summary)
        
#         else:
#             print("Data quality is good!")
            
    
    
#     def filter_missing_values(self):
        
#         if self.dataset.isnull().values.any():
#             self.dataset = self.dataset.interpolate(method='linear', axis=0, limit_direction='forward')
        
#         else:
#             print("No missing values detected!")
    
    
    
#     def plot_hist(self, feature=None):
        
#         data = self.dataset.to_numpy()
        
#         print(len(data))
        
#         sns.histplot(data[:,feature], bins=int(len(data)/1000), kde=True, color="C0")
#         plt.xlabel('Values')
#         plt.ylabel('Density')
#         plt.title(f'Feature {feature}')
#         plt.show()
        


        





if __name__ == "__main__":
    data = np.load("Geophysical.npy")
    feature_names = ['DoY/366', 'ToD/1440', 'Solar_Zenith/44', 'Kp', 'R', 'Dst',
                        'ap', 'F10_7', 'AE', 'AL', 'AU', 'PC_potential', 'Lyman_alpha',
                        'Bx', 'By', 'Bz', 'dBx', 'dBy', 'dBz']

    # process = GeophysProcessing(data)
    # process.check_quality()
    
    
    
    folder_name="ionograms_1D"
    datetimes = filename_to_datetime(folder_name)

    
    # Initialize the processor
    process = GeophysProcessing(data, datetimes, feature_names)

    # Example: Visualize features over a specific period
    start = pd.Timestamp("2018-09-17")
    end = pd.Timestamp("2022-12-18")
    process.visualize_features(start, end, features=['Bx', 'By', 'Bz'])
    
    
    
    # Initialize FeatureFilter
    filter_tool = FeatureFilter(process)

    # Apply median filter to 'Kp' feature with threshold=2 and window_size=5
    filter_tool.apply_median_filter('Bx', threshold=6, window_size=29)
    filter_tool.apply_median_filter('By', threshold=6, window_size=29)
    filter_tool.apply_median_filter('Bz', threshold=6, window_size=29)
    # filter_tool.visualize_filtered_feature('Bx', start, end)
    filter_tool.update_geophys_processor()
    
    
    process.visualize_features(start, end, features=['Bx', 'By', 'Bz'])
    # process.visualize_resampled(rule="H", features=['Bx'])

















