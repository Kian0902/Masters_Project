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
    
    
    def batch_averaging(self, save_plot=False, weighted=False, log_scale=False):
        """
        Function for applying the averaging to the entire dataset by looping
        through the global keys (days).
        """
        # Loop through day
        for key in list(self.dataset.keys()):
            self.dataset[key] = self.average_over_period(self.dataset[key], save_plot=save_plot, weighted=weighted, log_scale=log_scale)
    
    
    
    
    
    def round_time(self, data_time):

        
        
        datetimes = [datetime(year, month, day, hour, minute, second) 
              for year, month, day, hour, minute, second in data_time]
        
        rounded_datetimes = [dt.replace(second=0, microsecond=0) + timedelta(minutes=1) if dt.second >= 30 
                      else dt.replace(second=0, microsecond=0) for dt in datetimes]
        
        
        # L = []
        # for i, dt in enumerate(rounded_datetimes):
        #     if (dt.minute % 15 == 1) and (rounded_datetimes[i-1].minute % 15 !=0):
        #         if rounded_datetimes[i-1].minute % 15 != 14:
        #             dt = dt.replace(second=0, microsecond=0) - timedelta(minutes=1)
        #             L.append(dt)
        #         else:
        #             dt = rounded_datetimes[i-1].replace(second=0, microsecond=0) + timedelta(minutes=1)
        #             L.append(dt)

        #     else:
        #         L.append(dt)
        
        
        adjusted_datetimes = [(dt.replace(second=0, microsecond=0) - timedelta(minutes=1)) if (dt.minute % 15 == 1 and rounded_datetimes[i-1].minute % 15 != 0 and rounded_datetimes[i-1].minute % 15 != 14)
        else (rounded_datetimes[i-1].replace(second=0, microsecond=0) + timedelta(minutes=1)) if (dt.minute % 15 == 1 and rounded_datetimes[i-1].minute % 15 == 14)
        else dt for i, dt in enumerate(rounded_datetimes)]
        

        # Convert back to numpy array
        rounded_dates_array = np.array([[dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second] 
                                        for dt in adjusted_datetimes])
        return rounded_dates_array
    
    
    def get_weights(self, errors: np.ndarray, eps: float = 1e-10)-> np.ndarray:
        """
        Calculate and normalize weights inversely proportional to the errors.
        
        Input (type)           | DESCRIPTION
        ------------------------------------------------
        errors (np.ndarray)    | Array of error measurements.
        eps (float)            | Constant to avoid division by zero. Default is 1e-10.
        
        Return (type)              | DESCRIPTION
        ------------------------------------------------
        weights (np.ndarray)       | Array of calculated normalized weights.
        
        
        """
        weights = 1 / (errors + eps)  # calculate weights
        weights /= np.sum(weights, axis=1, keepdims=True)  # Normalize weights such that they sum up to 1
        return weights
        
        
    
    
    def average_over_period(self, data: dict, period_min: int=15, save_plot=False, weighted=False, log_scale=False) -> dict:
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
        r_time  = self.round_time(data['r_time'])
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
        
        
        # Check if time15_ind is empty, if so, skip processing
        if len(time15_ind) == 0:
            print(f"No 15-minute time increment found. Skipping data with date: {r_time[0, :3]}")
            return data
        
        
        # Handle the first interval separately (from start to the first 15-minute mark)
        if time15_ind[0] > 0:
            ind_s = 0
            ind_f = time15_ind[0]
            
            if weighted:
                weights = self.get_weights(r_error[:, ind_s:ind_f])
                r_param_avg = np.nansum(r_param[:, ind_s:ind_f] * weights, axis=1)
            else:
                r_param_avg = np.nanmean(r_param[:, ind_s:ind_f], axis=1)
            
            r_error_avg = np.nanmean(r_error[:, ind_s:ind_f], axis=1)
            
            avg_data['r_param'].append(r_param_avg)
            avg_data['r_error'].append(r_error_avg)
            avg_data['r_time'].append(r_time[ind_f])
        
        for i in range(0, len(time15_ind) - 1):
            ind_s = time15_ind[i]
            ind_f = time15_ind[i + 1]
            
            if weighted:
                weights = self.get_weights(r_error[:, ind_s:ind_f])
                r_param_avg = np.nansum(r_param[:, ind_s:ind_f] * weights, axis=1)
            else:
                r_param_avg = np.nanmean(r_param[:, ind_s:ind_f], axis=1)
            
            r_error_avg = np.nanmean(r_error[:, ind_s:ind_f], axis=1)
            
            avg_data['r_param'].append(r_param_avg)
            avg_data['r_error'].append(r_error_avg)
            avg_data['r_time'].append(r_time[ind_f])
        
        if time15_ind[-1] < len(r_time) - 1:
            ind_s = time15_ind[-1]
            ind_f = len(r_time)
            
            if weighted:
                weights = self.get_weights(r_error[:, ind_s:ind_f])
                r_param_avg = np.nansum(r_param[:, ind_s:ind_f] * weights, axis=1)
            else:
                r_param_avg = np.nanmean(r_param[:, ind_s:ind_f], axis=1)
            
            r_error_avg = np.nanmean(r_error[:, ind_s:ind_f], axis=1)
            
            avg_data['r_param'].append(r_param_avg)
            avg_data['r_error'].append(r_error_avg)
            avg_data['r_time'].append(r_time[-1])
        
        avg_data['r_param'] = np.array(avg_data['r_param']).T
        avg_data['r_error'] = np.array(avg_data['r_error']).T
        avg_data['r_time'] = np.array(avg_data['r_time'])
        
        print(f'Num of 15min:  {avg_data["r_time"].shape}   Num of 1min {r_time.shape} ')
            
        if save_plot is True:
            self.plot_and_save_comparison(data, avg_data, log_scale=log_scale)
        
        
        return avg_data
    
    
    def plot_and_save_comparison(self, original_data: dict, averaged_data: dict, log_scale=False):
        """
        Plot a comparison of original and averaged data using pcolormesh.

        Input (type)                 | DESCRIPTION
        ------------------------------------------------
        original_data (dict)         | Dictionary containing the original data.
        averaged_data (dict)         | Dictionary containing the averaged data.
        log_scale (bool)             | Whether to use a logarithmic scale for the color mapping.
        """
        
        # Convert time arrays to datetime objects
        r_time_orig = np.array([datetime(year, month, day, hour, minute) 
                                for year, month, day, hour, minute, second in original_data['r_time']])
        r_h_orig = original_data['r_h']
        r_param_orig = original_data['r_param']
        
        r_time_avg = np.array([datetime(year, month, day, hour, minute) 
                                for year, month, day, hour, minute, second in averaged_data['r_time']])
        r_h_avg = averaged_data['r_h']
        r_param_avg = averaged_data['r_param']
        
        if log_scale:
            r_param_orig = np.log10(r_param_orig)
            r_param_avg = np.log10(r_param_avg)
            vmin = 10  # Logarithmic scale, adjust as necessary
            vmax = 11  # Logarithmic scale, adjust as necessary
        else:
            vmin = 1e9  # Linear scale, adjust as necessary
            vmax = 6e11  # Linear scale, adjust as necessary
        
        # Date
        date_str = r_time_orig[0].strftime('%Y-%m-%d')
        
        # Creating the plots
        fig, ax = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
        fig.suptitle(f'Date: {date_str}', fontsize=20)
        
        # Plotting original data
        pcm_orig = ax[0].pcolormesh(r_time_orig, r_h_orig.flatten(), r_param_orig, shading='auto', cmap='turbo', vmin=vmin, vmax=vmax)
        ax[0].set_title(f'Original Data {r_param_orig.shape}')
        ax[0].set_xlabel('Time (hours)')
        ax[0].set_ylabel('Altitude (km)')
        ax[0].xaxis.set_major_formatter(DateFormatter('%d %H:%M'))
        fig.autofmt_xdate()
        
        # Add colorbar for the original data
        cbar = fig.colorbar(pcm_orig, ax=ax, orientation='vertical', fraction=0.03, pad=0.04, aspect=20, shrink=1)
        cbar.set_label('Parameter Value' + (' (log scale)' if log_scale else ''))
        
        # Plotting averaged data
        pcm_avg = ax[1].pcolormesh(r_time_avg, r_h_avg.flatten(), r_param_avg, shading='auto', cmap='turbo', vmin=vmin, vmax=vmax)
        ax[1].set_title(f'Averaged Data {r_param_avg.shape}')
        ax[1].set_xlabel('Time (hours)')
        ax[1].set_ylabel('Altitude (km)')
        ax[1].xaxis.set_major_formatter(DateFormatter('%d %H:%M'))
        fig.autofmt_xdate()
        
        # Display the plots
        plt.show()
        
    
    
    
    def return_data(self):
        """
        Returns self.data
        """
        return self.dataset











