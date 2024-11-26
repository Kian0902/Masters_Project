# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 15:11:24 2024

@author: Kian Sartipzadeh
"""



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from matplotlib.dates import DateFormatter
import pickle




class RadarDataProcessor:
    def __init__(self, file_path):
        self.file_path = file_path
        self.raw_data = None
        self.processed_data = {}

    def load_data(self):
        """
        Loads the radar data from the CSV file.
        """
        self.raw_data = pd.read_csv(self.file_path)
        self.raw_data['Timestamp'] = pd.to_datetime(self.raw_data['Timestamp'], format='%d-%b-%Y %H:%M:%S')

    def process_data(self):
        """
        Processes the raw data into the nested dictionary format.
        """
        if self.raw_data is None:
            raise ValueError("Data not loaded. Please run load_data() first.")

        # Get the unique dates in the dataset
        unique_dates = self.raw_data['Timestamp'].dt.date.unique()

        # Define the altitudes (columns excluding Timestamp)
        altitude_columns = self.raw_data.columns[1:]
        N = len(altitude_columns)

        # Loop over each unique date to create the nested structure
        for date in unique_dates:
            formatted_date = f"{date.year}-{date.month}-{date.day}"
            
            # Filter the dataframe for the current date
            daily_data = self.raw_data[self.raw_data['Timestamp'].dt.date == date]

            # Create the numpy arrays for the current date
            M = len(daily_data)

            # Extract the 'r_time' array
            r_time = np.array([[timestamp.year, timestamp.month, timestamp.day, timestamp.hour, timestamp.minute, timestamp.second]
                               for timestamp in daily_data['Timestamp']])

            # Extract the 'r_h' array (altitudes)
            r_h = np.array([list(map(lambda x: int(x.split('_')[1]), altitude_columns))])

            # Extract the 'r_param' array (electron densities)
            r_param = ((daily_data[altitude_columns].to_numpy()*1e6)**2)/9**2

            # Assign the arrays to the nested dictionary
            self.processed_data[formatted_date] = {
                "r_time": r_time,
                "r_h": r_h,
                "r_param": r_param
            }

    def get_processed_data_for_date(self, date):
        """
        Returns the processed data for a specific date.
        """
        if not self.processed_data:
            raise ValueError("Data not processed. Please run process_data() first.")
        return self.processed_data.get(date, "Date not found.")

    def return_data(self):
        """
        Returns the entire processed data.
        """
        if not self.processed_data:
            raise ValueError("Data not processed. Please run process_data() first.")
        return self.processed_data




def plot(data):
    """
    Plot a comparison of original and averaged data using pcolormesh.

    Input (type)                 | DESCRIPTION
    ------------------------------------------------
    original_data (dict)         | Dictionary containing the original data.
    """
    
    # Convert time arrays to datetime objects
    r_time = np.array([datetime(year, month, day, hour, minute, second) 
                            for year, month, day, hour, minute, second in data['r_time']])
    r_h = data['r_h']
    r_param = data['r_param']
    # r_error = data['r_error']
    
    # Date
    date_str = r_time[0].strftime('%Y-%m-%d')
    
    # Creating the plots
    fig, ax = plt.subplots(1, 2, figsize=(10, 8))
    fig.suptitle(f'Date: {date_str}', fontsize=15)
    fig.tight_layout()
    
    
    # Plotting original data
    pcm_ne=ax[0].pcolormesh(r_time, r_h.flatten(), np.log10(r_param.T), shading='auto', cmap='turbo', vmin=9, vmax=12)
    ax[0].set_title('EISCAT UHF', fontsize=17)
    ax[0].set_xlabel('Time [hours]')
    ax[0].set_ylabel('Altitude [km]')
    ax[0].xaxis.set_major_formatter(DateFormatter('%H:%M'))
    fig.autofmt_xdate()
    
    # Add colorbar for the original data
    cbar = fig.colorbar(pcm_ne, ax=ax[0], orientation='vertical')
    cbar.set_label(r'$log_{10}(n_e)$ [g/cm$^3$]', fontsize=17)
    
    for i in range(0, r_param.shape[0]):
        
        ax[1].plot(np.log10(r_param[i]), r_h.flatten())

    
    
    # Display the plots
    plt.show()



def plot2(data1, data2):
    """
    Plot a comparison of original and averaged data using pcolormesh.

    Input (type)                 | DESCRIPTION
    ------------------------------------------------
    original_data (dict)         | Dictionary containing the original data.
    """
    
    # Convert time arrays to datetime objects
    r_time1 = np.array([datetime(year, month, day, hour, minute, second) 
                            for year, month, day, hour, minute, second in data1['r_time']])
    # Convert time arrays to datetime objects
    r_time2 = np.array([datetime(year, month, day, hour, minute, second) 
                            for year, month, day, hour, minute, second in data2['r_time']])
    r_h1 = data1['r_h']
    r_h2 = data2['r_h']
    r_param1 = data1['r_param']
    r_param2 = data2['r_param']
    
    # Date
    date_str = r_time1[0].strftime('%Y-%m-%d')
    
    # Creating the plots
    fig, ax = plt.subplots(1, 2, figsize=(10, 8), sharey=True)
    fig.suptitle(f'Date: {date_str}', fontsize=15)
    fig.tight_layout()
    
    x_limits = [r_time1[0], r_time1[-1]]
    # y_limits = [r_h1.min(), r_h1.max()]
    
    # Plotting original data
    ne_EISCAT = ax[0].pcolormesh(r_time1, r_h1.flatten(), np.log10(r_param1), shading='auto', cmap='turbo', vmin=10, vmax=12)
    ax[0].set_title('EISCAT UHF', fontsize=17)
    ax[0].set_xlabel('Time [hours]')
    ax[0].set_ylabel('Altitude [km]')
    ax[0].xaxis.set_major_formatter(DateFormatter('%H:%M'))
    fig.autofmt_xdate()
    
    # Add colorbar for the original data
    # cbar = fig.colorbar(pcm_ne, ax=ax[0], orientation='vertical', fraction=0.03, pad=0.04, aspect=20, shrink=1)
    # cbar.set_label('log10(n_e) [g/cm^3]')
    
    # fig.autofmt_xdate()
    
    # Plotting original data
    ne_Artist = ax[1].pcolormesh(r_time2, r_h2.flatten(), np.log10(r_param2.T), shading='auto', cmap='turbo', vmin=10, vmax=12)
    ax[1].set_title('Artist', fontsize=17)
    ax[1].set_xlabel('Time [hours]')
    # ax[1].set_ylabel('Altitude [km]')
    ax[1].xaxis.set_major_formatter(DateFormatter('%H:%M'))
    ax[1].set_xlim(x_limits)
    # ax[1].set_ylim(y_limits)
    # fig.autofmt_xdate()
    
    # Add colorbar for the original data
    cbar = fig.colorbar(ne_EISCAT, ax=ax[1], orientation='vertical', fraction=0.03, pad=0.04, aspect=44, shrink=3)
    cbar.set_label(r'$log_{10}(n_e)$ [g/cm$^3$]', fontsize=17)
    
    
    # Display the plots
    plt.show()



def import_file(file_name):
    with open(file_name, 'rb') as f:
        dataset = pickle.load(f)
    
    return dataset






# Example usage
file_path = 'artist5_2020.csv'
radar_processor = RadarDataProcessor(file_path)
radar_processor.load_data()
radar_processor.process_data()

X_artist = radar_processor.processed_data
X_EISCAT = import_file(file_name="X_averaged_2020")






for day in X_EISCAT:
    # plot2(X_EISCAT[day], X_artist[day])
    
    
    for i in range(0, X_artist[day]['r_param'].shape[0]):
        
        plt.plot(np.log10(X_artist[day]['r_param'][i]), X_artist[day]['r_h'].flatten())




    plt.show()












