# # -*- coding: utf-8 -*-
# """
# Created on Tue Aug 27 15:58:50 2024

# @author: Kian Sartipzadeh
# """


import numpy as np
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.dates import DateFormatter
from matplotlib.colors import LogNorm

from data_utils import from_array_to_datetime



class EISCATAverager:
    """
    Class for averaging EISCAT radar measurement data over specified time intervals.
    """
    def __init__(self, dataset: dict, plot_result: bool=False):
        """
        Initialize with data to be averaged.
        
        Attributes (type)    | DESCRIPTION
        ------------------------------------------------
        dataset (dict)       | Dictionary containing the EISCAT data to be averaged.
        """
        self.dataset = dataset
        self.plot_result = plot_result
        
    def average_15min(self) -> dict:
        """
        Averages the radar data into 15-minute intervals, assigning the average to the next 15-minute mark.

        Returns:
            dict: A dictionary with the same structure as the input dataset, but with data averaged into 15-minute intervals.
                  Includes a new key 'num_avg_samp' indicating the number of profiles averaged for each time.
        """
        averaged_dataset = {}
        for date_str, data_dict in self.dataset.items():
            # Parse the current date
            try:
                date_obj = datetime.strptime(date_str, "%Y-%m-%d")
            except ValueError:
                print(f"Skipping invalid date key: {date_str}")
                continue
            year, month, day = date_obj.year, date_obj.month, date_obj.day

            # Extract original times and convert to datetime objects
            r_time = data_dict['r_time']
            if r_time.size == 0:
                continue  # Skip if no data
            times = [datetime(*map(int, t)) for t in r_time]

            # Generate all 15-minute bins for the current day
            start_of_day = datetime(year, month, day, 0, 0, 0)
            bins = []
            current = start_of_day
            while current.date() == start_of_day.date():
                bins.append(current)
                current += timedelta(minutes=15)

            # Prepare lists to hold averaged data for the current day
            averaged_r_time = []
            averaged_r_param = []
            averaged_r_error = []
            num_avg_samp = []  # List to store the number of profiles averaged for each bin

            for bin_start in bins:
                bin_end = bin_start + timedelta(minutes=15)
                # Collect indices of times within the current bin
                indices = [i for i, t in enumerate(times) if bin_start <= t < bin_end]
                if not indices:
                    continue  # Skip bins with no data

                # Average the parameters and errors
                param_slice = data_dict['r_param'][:, indices]
                error_slice = data_dict['r_error'][:, indices]
                
                avg_param = np.mean(param_slice, axis=1, keepdims=True)
                avg_error = np.sqrt(np.sum(error_slice ** 2, axis=1, keepdims=True)) / len(indices)
                
                # Append the averaged data
                averaged_r_time.append([
                    bin_start.year, bin_start.month, bin_start.day,
                    bin_start.hour, bin_start.minute, bin_start.second
                ])
                
                averaged_r_param.append(avg_param)
                averaged_r_error.append(avg_error)
                num_avg_samp.append(len(indices))  # Store the number of profiles averaged

            if not averaged_r_time:
                continue  # Skip days with no data after averaging

            # Convert lists to numpy arrays
            averaged_r_time = np.array(averaged_r_time)
            averaged_r_param = np.hstack(averaged_r_param)
            averaged_r_error = np.hstack(averaged_r_error)
            num_avg_samp = np.array(num_avg_samp)  # Convert to numpy array

            # Construct the averaged data dictionary for the current date
            averaged_data = {
                'r_time': averaged_r_time,
                'r_h': data_dict['r_h'],
                'r_param': averaged_r_param,
                'r_error': averaged_r_error,
                'num_avg_samp': num_avg_samp  # Add the number of profiles averaged
            }
            
            
            
            if self.plot_result:
                self.plot_compare(self.dataset[date_str], averaged_data)
            
            self.dataset[date_str] = averaged_data 
    
    
    
    
    def plot_compare(self, org_data, avg_data):
        
        # Original Data
        org_time  = from_array_to_datetime(org_data['r_time'])
        org_h     = org_data['r_h'].flatten()
        org_param = org_data['r_param']
        org_error = org_data['r_error'] 
        
        
        # Averaged Data
        avg_time  = from_array_to_datetime(avg_data['r_time'])
        avg_h     = avg_data['r_h'].flatten()
        avg_param = avg_data['r_param']
        avg_error = avg_data['r_error'] 
        
        
        
        
        fig = plt.figure(figsize=(12, 6))
        gs = GridSpec(1, 3, width_ratios=[1, 1, 0.05], wspace=0.1)
        
        
        # ___________ Defining axes ___________
        ax0 = fig.add_subplot(gs[0, 0])
        ax1 = fig.add_subplot(gs[0, 1], sharey=ax0)
        cax2 = fig.add_subplot(gs[0, 2])
        
        
        MIN, MAX = 1e10, 1e12
        subtit_size, xlab_size, ylab_size, cbar_size = 17, 13, 13, 13
        
        # EISCAT UHF
        org_Ne = ax0.pcolormesh(org_time, org_h, org_param, shading='auto', cmap='turbo', norm=LogNorm(vmin=MIN, vmax=MAX))
        # ax0.set_title('EISCAT UHF', fontsize=subtit_size)
        # ax0.set_ylabel('Altitude [km]', fontsize=ylab_size)
        ax0.xaxis.set_major_formatter(DateFormatter('%H:%M'))

        
        # KIAN-Net
        ax1.pcolormesh(avg_time, avg_h, avg_param, shading='auto', cmap='turbo', norm=LogNorm(vmin=MIN, vmax=MAX))
        # ax1.set_title('KIAN-Net', fontsize=subtit_size)
        ax1.xaxis.set_major_formatter(DateFormatter('%H:%M'))
        ax1.tick_params(labelleft=False)
        
        # Colorbar
        cbar2 = fig.colorbar(org_Ne, cax=cax2, orientation='vertical')
        cbar2.set_label('$n_e$ [n/m$^3$]', fontsize=cbar_size)
        
        # Rotate x-axis labels
        for ax in [ax0, ax1]:
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='center')

        
        plt.show()
    
    
    
    def return_data(self):
        """
        Returns self.data
        """
        return self.dataset












    
# import numpy as np
# import matplotlib.pyplot as plt
# from datetime import datetime, timedelta
# from matplotlib.dates import DateFormatter

# class EISCATAverager:
#     """
#     Class for averaging EISCAT radar measurement data over specified time intervals.
#     """
#     def __init__(self, dataset: dict):
#         """
#         Initialize with data to be averaged.
        
#         Attributes (type)    | DESCRIPTION
#         ------------------------------------------------
#         dataset (dict)       | Dictionary containing the EISCAT data to be averaged.
#         """
#         self.dataset = dataset
    
    
#     def batch_averaging(self, save_plot=False, weighted=False, log_scale=False):
#         """
#         Function for applying the averaging to the entire dataset by looping
#         through the global keys (days).
#         """
#         # Loop through day
#         for key in list(self.dataset.keys()):
#             self.dataset[key] = self.average_over_period(self.dataset[key], save_plot=save_plot, weighted=weighted, log_scale=log_scale)
    
    
    
    
    
#     def round_time(self, data_time):

        
        
#         datetimes = [datetime(year, month, day, hour, minute, second) 
#               for year, month, day, hour, minute, second in data_time]
        
#         rounded_datetimes = [dt.replace(second=0, microsecond=0) + timedelta(minutes=1) if dt.second >= 30 
#                       else dt.replace(second=0, microsecond=0) for dt in datetimes]
        
        
#         # L = []
#         # for i, dt in enumerate(rounded_datetimes):
#         #     if (dt.minute % 15 == 1) and (rounded_datetimes[i-1].minute % 15 !=0):
#         #         if rounded_datetimes[i-1].minute % 15 != 14:
#         #             dt = dt.replace(second=0, microsecond=0) - timedelta(minutes=1)
#         #             L.append(dt)
#         #         else:
#         #             dt = rounded_datetimes[i-1].replace(second=0, microsecond=0) + timedelta(minutes=1)
#         #             L.append(dt)

#         #     else:
#         #         L.append(dt)
        
        
#         adjusted_datetimes = [(dt.replace(second=0, microsecond=0) - timedelta(minutes=1)) if (dt.minute % 15 == 1 and rounded_datetimes[i-1].minute % 15 != 0 and rounded_datetimes[i-1].minute % 15 != 14)
#         else (rounded_datetimes[i-1].replace(second=0, microsecond=0) + timedelta(minutes=1)) if (dt.minute % 15 == 1 and rounded_datetimes[i-1].minute % 15 == 14)
#         else dt for i, dt in enumerate(rounded_datetimes)]
        

#         # Convert back to numpy array
#         rounded_dates_array = np.array([[dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second] 
#                                         for dt in adjusted_datetimes])
#         return rounded_dates_array
    
    
#     def get_weights(self, errors: np.ndarray, eps: float = 1e-10)-> np.ndarray:
#         """
#         Calculate and normalize weights inversely proportional to the errors.
        
#         Input (type)           | DESCRIPTION
#         ------------------------------------------------
#         errors (np.ndarray)    | Array of error measurements.
#         eps (float)            | Constant to avoid division by zero. Default is 1e-10.
        
#         Return (type)              | DESCRIPTION
#         ------------------------------------------------
#         weights (np.ndarray)       | Array of calculated normalized weights.
        
        
#         """
#         weights = 1 / (errors + eps)  # calculate weights
#         weights /= np.sum(weights, axis=1, keepdims=True)  # Normalize weights such that they sum up to 1
#         return weights
        
        
    
    
#     def average_over_period(self, data: dict, period_min: int=15, save_plot=False, weighted=False, log_scale=False) -> dict:
#         """
#         Average the radar data over a specified time period.
        
#         Input (type)              | DESCRIPTION
#         ------------------------------------------------
#         period_minutes (int)      | Time period over which to average (in minutes).
        
#         Return (type)             | DESCRIPTION
#         ------------------------------------------------
#         averaged_data (dict)      | Dictionary containing the averaged data.
#         """

#         # Definin keys
#         r_time  = self.round_time(data['r_time'])
#         r_h     = data['r_h']
#         r_param = data['r_param']
#         r_error = data['r_error']
        
#         # Making new dict for storing averaged data
#         avg_data = {'r_time': [],
#             'r_h': r_h,
#             'r_param': [],
#             'r_error': []}
        
        
#         # Finding indices where minute = 00, 15, 30 and 45
#         time15_ind = np.where(r_time[:, 4] % period_min == 0)[0]
        
        
#         # Check if time15_ind is empty, if so, skip processing
#         if len(time15_ind) == 0:
#             print(f"No 15-minute time increment found. Skipping data with date: {r_time[0, :3]}")
#             return data
        
        
#         # Handle the first interval separately (from start to the first 15-minute mark)
#         if time15_ind[0] > 0:
#             ind_s = 0
#             ind_f = time15_ind[0]
            
#             if weighted:
#                 weights = self.get_weights(r_error[:, ind_s:ind_f])
#                 r_param_avg = np.nansum(r_param[:, ind_s:ind_f] * weights, axis=1)
#             else:
#                 r_param_avg = np.nanmean(r_param[:, ind_s:ind_f], axis=1)
            
#             r_error_avg = np.nanmean(r_error[:, ind_s:ind_f], axis=1)
            
#             avg_data['r_param'].append(r_param_avg)
#             avg_data['r_error'].append(r_error_avg)
#             avg_data['r_time'].append(r_time[ind_f])
        
#         for i in range(0, len(time15_ind) - 1):
#             ind_s = time15_ind[i]
#             ind_f = time15_ind[i + 1]
            
#             if weighted:
#                 weights = self.get_weights(r_error[:, ind_s:ind_f])
#                 r_param_avg = np.nansum(r_param[:, ind_s:ind_f] * weights, axis=1)
#             else:
#                 r_param_avg = np.nanmean(r_param[:, ind_s:ind_f], axis=1)
            
#             r_error_avg = np.nanmean(r_error[:, ind_s:ind_f], axis=1)
            
#             avg_data['r_param'].append(r_param_avg)
#             avg_data['r_error'].append(r_error_avg)
#             avg_data['r_time'].append(r_time[ind_f])
        
#         if time15_ind[-1] < len(r_time) - 1:
#             ind_s = time15_ind[-1]
#             ind_f = len(r_time)
            
#             if weighted:
#                 weights = self.get_weights(r_error[:, ind_s:ind_f])
#                 r_param_avg = np.nansum(r_param[:, ind_s:ind_f] * weights, axis=1)
#             else:
#                 r_param_avg = np.nanmean(r_param[:, ind_s:ind_f], axis=1)
            
#             r_error_avg = np.nanmean(r_error[:, ind_s:ind_f], axis=1)
            
#             avg_data['r_param'].append(r_param_avg)
#             avg_data['r_error'].append(r_error_avg)
#             avg_data['r_time'].append(r_time[-1])
        
#         avg_data['r_param'] = np.array(avg_data['r_param']).T
#         avg_data['r_error'] = np.array(avg_data['r_error']).T
#         avg_data['r_time'] = np.array(avg_data['r_time'])
        
#         print(f'Num of 15min:  {avg_data["r_time"].shape}   Num of 1min {r_time.shape} ')
            
#         if save_plot is True:
#             self.plot_and_save_comparison(data, avg_data, log_scale=log_scale)
        
        
#         return avg_data
    
    
#     def plot_and_save_comparison(self, original_data: dict, averaged_data: dict, log_scale=False):
#         """
#         Plot a comparison of original and averaged data using pcolormesh.

#         Input (type)                 | DESCRIPTION
#         ------------------------------------------------
#         original_data (dict)         | Dictionary containing the original data.
#         averaged_data (dict)         | Dictionary containing the averaged data.
#         log_scale (bool)             | Whether to use a logarithmic scale for the color mapping.
#         """
        
#         # Convert time arrays to datetime objects
#         r_time_orig = np.array([datetime(year, month, day, hour, minute) 
#                                 for year, month, day, hour, minute, second in original_data['r_time']])
#         r_h_orig = original_data['r_h']
#         r_param_orig = original_data['r_param']
        
#         r_time_avg = np.array([datetime(year, month, day, hour, minute) 
#                                 for year, month, day, hour, minute, second in averaged_data['r_time']])
#         r_h_avg = averaged_data['r_h']
#         r_param_avg = averaged_data['r_param']
        
#         if log_scale:
#             r_param_orig = np.log10(r_param_orig)
#             r_param_avg = np.log10(r_param_avg)
#             vmin = 10  # Logarithmic scale, adjust as necessary
#             vmax = 11  # Logarithmic scale, adjust as necessary
#         else:
#             vmin = 1e9  # Linear scale, adjust as necessary
#             vmax = 6e11  # Linear scale, adjust as necessary
        
#         # Date
#         date_str = r_time_orig[0].strftime('%Y-%m-%d')
        
#         # Creating the plots
#         fig, ax = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
#         fig.suptitle(f'Date: {date_str}', fontsize=20)
        
#         # Plotting original data
#         pcm_orig = ax[0].pcolormesh(r_time_orig, r_h_orig.flatten(), r_param_orig, shading='auto', cmap='turbo', vmin=vmin, vmax=vmax)
#         ax[0].set_title(f'Original Data {r_param_orig.shape}')
#         ax[0].set_xlabel('Time (hours)')
#         ax[0].set_ylabel('Altitude (km)')
#         ax[0].xaxis.set_major_formatter(DateFormatter('%d %H:%M'))
#         fig.autofmt_xdate()
        
#         # Add colorbar for the original data
#         cbar = fig.colorbar(pcm_orig, ax=ax, orientation='vertical', fraction=0.03, pad=0.04, aspect=20, shrink=1)
#         cbar.set_label('Parameter Value' + (' (log scale)' if log_scale else ''))
        
#         # Plotting averaged data
#         pcm_avg = ax[1].pcolormesh(r_time_avg, r_h_avg.flatten(), r_param_avg, shading='auto', cmap='turbo', vmin=vmin, vmax=vmax)
#         ax[1].set_title(f'Averaged Data {r_param_avg.shape}')
#         ax[1].set_xlabel('Time (hours)')
#         ax[1].set_ylabel('Altitude (km)')
#         ax[1].xaxis.set_major_formatter(DateFormatter('%d %H:%M'))
#         fig.autofmt_xdate()
        
#         # Display the plots
#         plt.show()
        
    
    
    
#     def return_data(self):
#         """
#         Returns self.data
#         """
#         return self.dataset











