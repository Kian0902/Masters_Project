# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 15:58:50 2024

@author: Kian Sartipzadeh
"""




import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from matplotlib.dates import DateFormatter

class EISCATAverager:
    """
    Class for averaging EISCAT radar measurement data over specified time intervals.
    """
    def __init__(self, dataset: dict):
        """
        Initialize with data to be averaged.
        
        Attributes (type)    | DESCRIPTION
        ------------------------------------------------
        dataset (dict)       | Dictionary containing the EISCAT data to be averaged.
        """
        self.dataset = dataset
    
    
    
    def batch_averaging(self, save_plot=False):
        """
        Function for applying the averaging to the entire dataset by looping
        through the global keys (days).
        """
        # Loop through day
        for key in list(self.dataset.keys()):
            self.dataset[key] = self.average_over_period(self.dataset[key], save_plot=save_plot)

            
    
    def round_time(self, data_time):
        """
        Rounds the data time array to the nearest minute.
        
        Input (type)             | DESCRIPTION
        ------------------------------------------------
        data_time (np.ndarray)   | Array with dates and times.
        
        Return (type)                      | DESCRIPTION
        ------------------------------------------------
        rounded_dates_array (np.ndarray)   | Array with rounded dates and times.
        """
        # Convert each row with date and time into datetime objects
        datetimes = [datetime(year, month, day, hour, minute, second) 
             for year, month, day, hour, minute, second in data_time]
        
        # Rounding datetimes into closest minute
        rounded_datetimes = [dt.replace(second=0, microsecond=0) + timedelta(minutes=1) if dt.second >= 30 
                     else dt.replace(second=0, microsecond=0) 
                     for dt in datetimes]
        
        # Convert back to numpy array
        rounded_dates_array = np.array([[dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second] 
                                        for dt in rounded_datetimes])
        return rounded_dates_array
    
    
    
    def average_over_period(self, data: dict, period_min: int=15, save_plot=False):
        """
        Average the radar data over a specified time period.
        
        Input (type)              | DESCRIPTION
        ------------------------------------------------
        period_minutes (int)      | Time period over which to average (in minutes).
        
        Return (type)             | DESCRIPTION
        ------------------------------------------------
        averaged_data (dict)      | Dictionary containing the averaged data.
        """

        # Definin keys
        r_time  = data['r_time']
        r_h     = data['r_h']
        r_param = data['r_param']
        r_error = data['r_error']
        
    
        # Making new dict for storing averaged data
        avg_data = {'r_time': [],
            'r_h': r_h,
            'r_param': [],
            'r_error': []}
        
        
        # Finding indices where minute = 00, 15, 30 and 45
        time15_ind = np.where(r_time[:, 4] % period_min == 0)[0]

        
        # Handle the first interval separately (from start to the first 15-minute mark)
        if time15_ind[0] > 0:
            ind_s = 0
            ind_f = time15_ind[0]
            
            # Averaging between indices
            r_param_avg = np.nanmean(r_param[:, ind_s: ind_f], axis=1)
            r_error_avg = np.nanmean(r_error[:, ind_s: ind_f], axis=1)
            
            # Appending averaged values
            avg_data['r_param'].append(r_param_avg)
            avg_data['r_error'].append(r_error_avg)
            avg_data['r_time'].append(r_time[ind_f])
            
        
        for i in range(0, len(time15_ind) - 1):
            
            # Index for current and next 15 min interval
            ind_s = time15_ind[i]
            ind_f = time15_ind[i + 1]
            
            # Averaging between indices
            r_param_avg = np.nanmean(r_param[:, ind_s: ind_f], axis=1)
            r_error_avg = np.nanmean(r_error[:, ind_s: ind_f], axis=1)
            
            # Appending averaged values
            avg_data['r_param'].append(r_param_avg)
            avg_data['r_error'].append(r_error_avg)
            avg_data['r_time'].append(r_time[ind_f])
            
            
        # Handling the leftover data at the end of the time array
        if time15_ind[-1] < len(r_time) - 1:
            ind_s = time15_ind[-1]
            ind_f = len(r_time)  # end of the array
            
            # Averaging the leftover data
            r_param_avg = np.nanmean(r_param[:, ind_s: ind_f], axis=1)
            r_error_avg = np.nanmean(r_error[:, ind_s: ind_f], axis=1)
            
            # Appending the last averaged values
            avg_data['r_param'].append(r_param_avg)
            avg_data['r_error'].append(r_error_avg)
            avg_data['r_time'].append(r_time[-1])
        
        # Converting list to numpy arrays for consistency
        avg_data['r_param'] = np.array(avg_data['r_param']).T
        avg_data['r_error'] = np.array(avg_data['r_error']).T
        avg_data['r_time'] = np.array(avg_data['r_time'])
        
        print(f'Num of 15min:  {avg_data["r_time"].shape}   Num of 1min {r_time.shape} ')
        
        if save_plot is True:
            self.plot_and_save_comparison(data, avg_data)
        
        
        return avg_data
    
    
    def plot_and_save_comparison(self, original_data: dict, averaged_data: dict):
        """
        Plot a comparison of original and averaged data using pcolormesh.

        Input (type)                 | DESCRIPTION
        ------------------------------------------------
        original_data (dict)         | Dictionary containing the original data.
        averaged_data (dict)         | Dictionary containing the averaged data.
        """
        
        
        
        # # Extracting the necessary data
        # r_time_orig = original_data['r_time'][:, 3] + original_data['r_time'][:, 4] / 60.0  # Convert to hours
        # r_time_avg = averaged_data['r_time'][:, 3] + averaged_data['r_time'][:, 4] / 60.0  # Convert to hours
        # r_h = original_data['r_h']  # Altitude data (same for original and averaged)
        
        # # Select the parameter to plot
        # r_param_orig = original_data['r_param'][param_index, :]
        # r_param_avg = averaged_data['r_param'][param_index, :]
        
        
        
        # Convert time arrays to datetime objects
        r_time_orig = np.array([datetime(year, month, day, hour, minute) 
                                for year, month, day, hour, minute, second in original_data['r_time']])
        
        # r_time_orig = original_data['r_time'][:, 3] + original_data['r_time'][:, 4] / 60.0  # Convert to hours
        # r_time_orig = original_data['r_time']
        r_h_orig = original_data['r_h']
        r_param_orig = original_data['r_param']
        
        print("Original")
        print(r_time_orig.shape)
        print(r_h_orig.shape)
        print(r_param_orig.shape)
        
        r_time_avg = np.array([datetime(year, month, day, hour, minute) 
                               for year, month, day, hour, minute, second in averaged_data['r_time']])
        # r_time_avg = averaged_data['r_time'][:, 3] + averaged_data['r_time'][:, 4] / 60.0  # Convert to hours
        # r_time_avg = averaged_data['r_time']
        r_h_avg = averaged_data['r_h']
        r_param_avg = averaged_data['r_param']
        
        print("\nAveraged")
        print(r_time_avg.shape)
        print(r_h_avg.shape)
        print(r_param_avg.shape)
        
        
        # Determine the color scale limits based on the original data
        vmin = np.nanmin(r_param_orig)
        vmax = np.nanmax(r_param_orig)
        
        # Determine the overall time range to ensure both plots show the same x-axis
        time_min = min(r_time_orig[0], r_time_avg[0])
        time_max = max(r_time_orig[-1], r_time_avg[-1])
        
        
        # Creating the plots
        fig, ax = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)

        # Plotting original data
        pcm_orig = ax[0].pcolormesh(r_time_orig, r_h_orig.flatten(), r_param_orig, shading='auto', cmap='turbo', vmin=vmin, vmax=vmax)
        ax[0].set_title('Original Data')
        ax[0].set_xlabel('Time (hours)')
        ax[0].set_ylabel('Altitude (km)')
        ax[0].set_xlim(time_min, time_max)
        ax[0].xaxis.set_major_formatter(DateFormatter('%d %H:%M'))
        fig.autofmt_xdate()
        # fig.colorbar(pcm_orig, ax=ax[0], label='Parameter Value')
        
        
        # Add colorbar for the original data
        cbar = fig.colorbar(pcm_orig, ax=ax, orientation='vertical', fraction=0.03, pad=0.04, aspect=40, shrink=1)
        cbar.set_label('Parameter Value')
        
        # Plotting averaged data
        pcm_avg = ax[1].pcolormesh(r_time_avg, r_h_avg.flatten(), r_param_avg, shading='auto', cmap='turbo', vmin=vmin, vmax=vmax)
        ax[1].set_title('Averaged Data')
        ax[1].set_xlabel('Time (hours)')
        ax[1].set_ylabel('Altitude (km)')
        ax[1].set_xlim(time_min, time_max)
        ax[1].xaxis.set_major_formatter(DateFormatter('%d %H:%M'))
        fig.autofmt_xdate()
        # fig.colorbar(pcm_avg, ax=ax[1], label='Parameter Value')
        
        # Display the plots
        plt.show()
    
    
    
    
    def return_data(self):
        """
        Returns self.data
        """
        return self.dataset











