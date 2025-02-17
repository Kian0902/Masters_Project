# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 17:24:01 2024

@author: Kian Sartipzadeh
"""

import os
import pickle
import numpy as np
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch
from matplotlib.dates import DateFormatter
import matplotlib.colors as colors
import seaborn as sns

def load_dict(file_name):
    with open(file_name, 'rb') as f:
        dataset = pickle.load(f)
    return dataset

def save_dict(dataset: dict, file_name: str):
    with open(file_name, 'wb') as file:
        pickle.dump(dataset, file)



# def plot_day(data):
#     r_time = from_array_to_datetime(data['r_time'])
#     r_h = data['r_h'].flatten()
#     r_param = data['r_param']
#     r_error = data['r_error'] 
    
    
#     fig, ax = plt.subplots()
    
#     ne=ax.pcolormesh(r_time, r_h, r_param, shading="auto", cmap="turbo", norm=colors.LogNorm(vmin=1e10, vmax=1e11))
#     ax.set_xlabel("Time (UT)")
#     ax.set_ylabel("Altitudes (km)")
#     ax.xaxis.set_major_formatter(DateFormatter('%H:%M'))
#     fig.colorbar(ne, ax=ax, orientation='vertical')
#     plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='center')
#     plt.show()



def get_day_data(dataset, day_idx):
    list_days = list(dataset.keys())
    day = list_days[day_idx]
    return dataset[day]


def get_day(dataset, day_idx):
    list_days = list(dataset.keys())
    day = list_days[day_idx]
    return {day: dataset[day]}



def from_strings_to_datetime(data_strings):
    data = from_strings_to_array(data_strings)
    
    # Convert time arrays to datetime objects
    r_time = np.array([datetime(year, month, day, hour, minute, second) 
                            for year, month, day, hour, minute, second in data])
    
    return r_time


def from_array_to_datetime(data):
    # Convert time arrays to datetime objects
    r_time = np.array([datetime(year, month, day, hour, minute, second) 
                            for year, month, day, hour, minute, second in data])
    
    return r_time



# Function to preprocess the input strings
def from_strings_to_array(date_strings):
    result = []
    for date_string in date_strings:
        date_part, time_part = date_string.split("_")
        year = int(date_part[:4])
        month = int(date_part[4:6])
        day = int(date_part[6:8])
        hour = int(time_part[:2])
        minute = int(time_part[2:4])
        second = 0  # Assuming seconds are always zero
        result.append(np.array([year, month, day, hour, minute, second]))
        
    return result








def inspect_dict(d, indent=0):
    """
    Recursively print all keys in a nested dictionary with the shape of their values.
    
    :param d: The dictionary to process.
    :param indent: The current indentation level (used for nested items).
    """
    for key, value in d.items():
        # Create indentation based on the depth level
        prefix = '  ' * indent
        
        # Determine the type and shape of the value
        if isinstance(value, dict):
            value_shape = f"dict with {len(value)} keys"
        elif isinstance(value, list):
            value_shape = f"list of length {len(value)}"
        elif isinstance(value, set):
            value_shape = f"set of length {len(value)}"
        elif isinstance(value, tuple):
            value_shape = f"tuple of length {len(value)}"
        elif isinstance(value, np.ndarray):
            value_shape = f"numpy array with shape {value.shape}"
        else:
            value_shape = f"type {type(value).__name__}"
        
        # Print the key along with its value shape
        print(f"{prefix}{key}: ({value_shape})")
        
        # If the value is also a dictionary, recursively call the function
        if isinstance(value, dict):
            inspect_dict(value, indent=indent + 1)








class MatchingFiles:
    """
    Class for handeling matching files between two folders.
    
    This becomes useful when faced with two data sources with matching
    filenames. Here the user has the option to delete the matching files
    from one or the other folders.
    
    Example: We have two folders containing radar data VHF and UHF but want
    to prioritize keeping the UHF data. Here, the sure has the option to delete
    the matching files in VHF folder.
    """
    def __init__(self, folder_1, folder_2):
        self.folder_1 = folder_1
        self.folder_2 = folder_2
    
    
    def list_mat_files(self, folder):
        return [f for f in os.listdir(folder) if f.endswith('.mat')]

    
    def get_matching_filenames(self):
        filenames_1 = self.list_mat_files(self.folder_1)
        filenames_2 = self.list_mat_files(self.folder_2)
        
        folder_1_filenames = set(os.path.splitext(f)[0] for f in filenames_1)
        folder_2_filenames = set(os.path.splitext(f)[0] for f in filenames_2)
        
        return sorted(list(folder_1_filenames.intersection(folder_2_filenames)))
        
    def remove_matching_vhf_files(self):
        matching_filenames = self.get_matching_filenames()
        for filename in matching_filenames:
            file_path = os.path.join(self.folder_1, f"{filename}.mat")
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"Removed: {file_path}")
            else:
                print(f"File not found: {file_path}")





def plot_available_data(data_dict):
    # Extract dates and M values
    dates = []
    m_values = []
    
    for date, measurements in data_dict.items():
        dates.append(date)
        m_values.append(measurements['r_time'].shape[0])
    
    # Convert dates to datetime objects
    dates = pd.to_datetime(dates)
    
    df = pd.DataFrame({
        'Date': dates,
        'M_samples': m_values
    })
    
    # Sort by date
    df = df.sort_values('Date')
    
    # Calculate the total number of days in the date range
    start_date = df['Date'].min()
    end_date = df['Date'].max()
    all_possible_days = pd.date_range(start=start_date, end=end_date, freq='D')
    total_days = len(all_possible_days)
    available_days = len(df)
    
    # Calculate total number of samples available
    total_samples_available = df['M_samples'].sum()
    
    # Define the maximum possible samples per day (e.g., based on the maximum observed M)
    max_samples_per_day = df['M_samples'].max()
    
    # Calculate the total possible samples if data were available every day
    total_possible_samples = total_days * max_samples_per_day
    
    # Print statistics
    print(f"Number of available days: {available_days}/{total_days}")
    print(f"Percentage of days with data: {available_days / total_days * 100:.2f}%")
    print(f"Total samples available: {total_samples_available}")
    print(f"Total possible samples: {total_possible_samples}")
    print(f"Percentage of samples available: {total_samples_available / total_possible_samples * 100:.2f}%")
    
    

    # Load and process sunspot data (same as before)
    data = np.genfromtxt('sunspots.csv', delimiter=';', dtype=float)
    years = data[:, 0]
    months = data[:, 1]
    sunspot_numbers = data[:, 3]
    sunspot_std_dev = data[:, 4]
    dates = [datetime(int(year), int(month), 1) for year, month in zip(years, months)]
    df_sun = pd.DataFrame({
        'Date': dates,
        'Sunspot Number': sunspot_numbers,
        'Standard Deviation': sunspot_std_dev
    })
    df_sun.set_index('Date', inplace=True)
    
    # Filter sunspot data to radar's time range
    radar_start_date = df['Date'].min()
    radar_end_date = df['Date'].max()
    df_sun_filtered = df_sun.loc[radar_start_date:radar_end_date]






    # Create figure with 2 subplots using GridSpec
    fig = plt.figure(figsize=(10, 6))
    gs = plt.GridSpec(2, 1, height_ratios=[0.6, 1], hspace=0.3)
    
    with sns.axes_style("dark"):
        # Top subplot (Radar data statistics)
        ax0 = fig.add_subplot(gs[0])
        
    ax0.bar(df['Date'], df['M_samples'], color='teal', width=2, alpha=0.65)
    

    
    # Configure top plot
    ax0.set_title("Number of 15 min samples within each available day")
    ax0.set_ylabel("Samples/Day")
    ax0.grid(True)
    ax0.set_xticklabels([])
    
    
    # Bottom subplot (Sunspots + Radar days)
    ax1 = fig.add_subplot(gs[1])
    
    # Calculate statistics
    days_ratio = f"{available_days} / {total_days} ({available_days/total_days:.1%})"
    samples_ratio = f"{total_samples_available} / {total_possible_samples} ({total_samples_available/total_possible_samples:.1%})"
    
    # Create legend text
    legend_text = (f"Days: {days_ratio}\n"
                   f"Samples: {samples_ratio}")
    
    # Add legend-like text box
    ax1.text(0.015, 0.65, legend_text,
             transform=ax1.transAxes,
             ha='left', va='top',
             bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round', alpha=0.9),
             fontsize=10)
    
    
    # Add radar availability lines
    for date in df['Date']:
        ax1.axvline(x=date, color='C0', alpha=0.7, linewidth=0.2, zorder=-2)
    
    # Plot sunspot data
    ax1.plot(df_sun_filtered.index, df_sun_filtered['Sunspot Number'], 
            color='red', label='Sunspot Number', zorder=2)
    ax1.fill_between(df_sun_filtered.index,
                    df_sun_filtered['Sunspot Number'] - df_sun_filtered['Standard Deviation'],
                    df_sun_filtered['Sunspot Number'] + df_sun_filtered['Standard Deviation'],
                    color='darkorange', alpha=0.5, label='±1 STD', zorder=2)
    

    
    # Configure bottom plot
    ax1.set_title('Sunspots with available days of data')
    ax1.set_xlabel('Date (UT)')
    ax1.set_ylabel('Sunspot Number')
    ax1.grid(axis='y', alpha=0.3)
    
    # Create unified legend
    handles = [
        plt.Line2D([0], [0], color='red', lw=1, label='Sunspots'),
        Patch(color='darkorange', alpha=0.5, label='Sunspot STD'),
        plt.Line2D([0], [0], color='dodgerblue', lw=1, label='Available Data')
    ]
    ax1.legend(handles=handles, loc='upper left', framealpha=0.9)
    
    for ax in [ax0, ax1]:
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='center')
    
    # plt.tight_layout()
    plt.show()


from matplotlib.colors import LogNorm


def plot_compare_all(data1, data2, data3, date_str):
    
    # Original Data
    r1_time  = from_array_to_datetime(data1['r_time'])
    r1_h     = data1['r_h'].flatten()
    r1_param = data1['r_param']
    r1_error = data1['r_error'] 
    
    
    # Filtered Data
    r2_time  = from_array_to_datetime(data2['r_time'])
    r2_h     = data2['r_h'].flatten()
    r2_param = data2['r_param']
    r2_error = data2['r_error'] 
    
    # Averaged Data
    r3_time  = from_array_to_datetime(data3['r_time'])
    r3_h     = data3['r_h'].flatten()
    r3_param = data3['r_param']
    r3_error = data3['r_error'] 
    
    
    
    fig = plt.figure(figsize=(14, 5))
    fig.suptitle(f'Before and After processing\n{date_str}', fontsize=20, y=1.07)
    gs = GridSpec(1, 4, width_ratios=[1, 1, 1, 0.05], wspace=0.2)
    
    
    # ___________ Defining axes ___________
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[0, 2])
    cax2 = fig.add_subplot(gs[0, 3])
    
    
    MIN, MAX = 1e10, 1e12
    subtit_size, xlab_size, ylab_size, cbar_size = 17, 15, 15, 15
    
    # Org
    r1_Ne = ax0.pcolormesh(r1_time, r1_h, r1_param, shading='auto', cmap='turbo', norm=LogNorm(vmin=MIN, vmax=MAX))
    ax0.set_title('Original', fontsize=subtit_size)
    ax0.set_ylabel('Altitude [km]', fontsize=ylab_size)
    ax0.set_xlabel('Time [UT]', fontsize=ylab_size)
    ax0.xaxis.set_major_formatter(DateFormatter('%H:%M'))
    
    
    # Filt
    ax1.pcolormesh(r2_time, r2_h, r2_param, shading='auto', cmap='turbo', norm=LogNorm(vmin=MIN, vmax=MAX))
    ax1.set_title('Filtered', fontsize=subtit_size)
    ax1.set_xlabel('Time [UT]', fontsize=ylab_size)
    ax1.xaxis.set_major_formatter(DateFormatter('%H:%M'))
    # ax1.tick_params(labelleft=False)
    
    # Avg
    ax2.pcolormesh(r3_time, r3_h, r3_param, shading='auto', cmap='turbo', norm=LogNorm(vmin=MIN, vmax=MAX))
    ax2.set_title('Averaged', fontsize=subtit_size)
    ax2.set_xlabel('Time [UT]', fontsize=ylab_size)
    ax2.xaxis.set_major_formatter(DateFormatter('%H:%M'))
    
    # Colorbar
    cbar2 = fig.colorbar(r1_Ne, cax=cax2, orientation='vertical')
    cbar2.set_label('$n_e$ [n m$^{-3}$]', fontsize=cbar_size)
    
    # Rotate x-axis labels
    for ax in [ax0, ax1, ax2]:
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='center')

    
    plt.show()



def plot_compare(data1, data2, date_str):
    
    # Original Data
    org_time  = from_array_to_datetime(data1['r_time'])
    org_h     = data1['r_h'].flatten()
    org_param = data1['r_param']
    org_error = data1['r_error'] 
    
    
    # Averaged Data
    avg_time  = from_array_to_datetime(data2['r_time'])
    avg_h     = data2['r_h'].flatten()
    avg_param = data2['r_param']
    avg_error = data2['r_error'] 
    
    
    
    
    fig = plt.figure(figsize=(12, 5))
    fig.suptitle(date_str, fontsize=17)
    gs = GridSpec(1, 3, width_ratios=[1, 1, 0.05], wspace=0.2)
    
    
    # ___________ Defining axes ___________
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    cax2 = fig.add_subplot(gs[0, 2])
    
    
    MIN, MAX = 1e10, 1e12
    subtit_size, xlab_size, ylab_size, cbar_size = 17, 13, 13, 13
    
    # EISCAT UHF
    org_Ne = ax0.pcolormesh(org_time, org_h, org_param, shading='auto', cmap='turbo', norm=LogNorm(vmin=MIN, vmax=MAX))
    ax0.set_title('Original', fontsize=subtit_size)
    ax0.set_ylabel('Altitude [km]', fontsize=ylab_size)
    ax0.set_xlabel('Time [UT]', fontsize=ylab_size)
    ax0.xaxis.set_major_formatter(DateFormatter('%H:%M'))
    
    
    # KIAN-Net
    ax1.pcolormesh(avg_time, avg_h, avg_param, shading='auto', cmap='turbo', norm=LogNorm(vmin=MIN, vmax=MAX))
    ax1.set_title('Filtered', fontsize=subtit_size)
    ax1.set_xlabel('Time [UT]', fontsize=ylab_size)
    ax1.xaxis.set_major_formatter(DateFormatter('%H:%M'))
    # ax1.tick_params(labelleft=False)
    
    # Colorbar
    cbar2 = fig.colorbar(org_Ne, cax=cax2, orientation='vertical')
    cbar2.set_label('$n_e$ [n m$^{-3}$]', fontsize=cbar_size)
    
    # Rotate x-axis labels
    for ax in [ax0, ax1]:
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='center')

    
    plt.show()




def plot_day(data, date):
    r_time = from_array_to_datetime(data['r_time'])
    r_h = data['r_h'].flatten()
    r_param = data['r_param']
    r_error = data['r_error'] 
    
    
    
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle(f"Original EISCAT UHF\n{date}", fontsize=17, x=0.47)
    
    ne=ax.pcolormesh(r_time, r_h, r_param, shading="auto", cmap="turbo", norm=colors.LogNorm(vmin=1e10, vmax=1e12))
    ax.set_xlabel("Time (UT)", fontsize=13)
    ax.set_ylabel("Altitude [km]", fontsize=13)
    ax.xaxis.set_major_formatter(DateFormatter('%H:%M'))
    
    cbar = fig.colorbar(ne, ax=ax, orientation='vertical')
    cbar.set_label('$n_e$  [n m$^{-3}$]', fontsize=13)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='center')
    plt.show()
    

def plot_sample(data, j):
    r_time = from_array_to_datetime(data['r_time'])
    r_h = data['r_h'].flatten()
    r_param = data['r_param']
    r_error = data['r_error'] 
    
    
    for i in range(0, len(r_time)):
        
        plt.plot(r_param[:, i], r_h)
        plt.errorbar(r_param[:, i], r_h, xerr=r_error[:, i])
        plt.xscale("log")
        plt.show()
        
        if i == j:
            break


