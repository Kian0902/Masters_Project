# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 15:58:50 2024

@author: Kian Sartipzadeh
"""




import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta


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
    
    
    
    def round_time(self, data_time):
        
        
        datetimes = [datetime(year, month, day, hour, minute, second) 
             for year, month, day, hour, minute, second in data_time]
        
        rounded_datetimes = [dt.replace(second=0, microsecond=0) + timedelta(minutes=1) if dt.second >= 30 
                     else dt.replace(second=0, microsecond=0) 
                     for dt in datetimes]
        
        # Convert back to numpy array (if necessary)
        rounded_dates_array = np.array([[dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second] 
                                        for dt in rounded_datetimes])
        
        
        return rounded_dates_array
    
    

    def average_over_period(self, period_min: int=15):
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
        r_time  = self.dataset['r_time']
        r_h     = self.dataset['r_h']
        r_param = self.dataset['r_param']
        r_error = self.dataset['r_error']
        
        
        # print(r_time.shape)
        
        r_time_new = self.round_time(r_time)
        
        
        for i in range(0, len(r_time)):
            print(r_time[i,:], r_time_new[i,:])
            print("\n")
        

        
        # # Making new dict for storing averaged data
        # avg_data = {'r_time': [],
        #     'r_h': r_h,
        #     'r_param': [],
        #     'r_error': []}
        
        
        # # Finding indices where minute = 00, 15, 30 and 45
        # time15_ind = np.where(r_time[:, 4] % period_min == 0)[0]

        # print(f'Num of 15min:  {time15_ind.shape}   Num of 1min {r_time.shape} ')

        
        # for i in range(0, len(time15_ind) - 1):
            
        #     # Index for current and next 15 min interval
        #     ind_s = time15_ind[i]
        #     ind_f = time15_ind[i + 1]
            
        #     # Averaging between indices
        #     r_param_avg = np.nanmean(r_param[:, ind_s: ind_f], axis=1)
        #     r_error_avg = np.nanmean(r_error[:, ind_s: ind_f], axis=1)
            
        #     # Appending averaged values
        #     avg_data['r_param'].append(r_param_avg)
        #     avg_data['r_error'].append(r_error_avg)
        #     avg_data['r_time'].append(r_time[ind_f])
        
        # # Converting list to numpy arrays for consistancy
        # avg_data['r_param'] = np.array(avg_data['r_param']).T
        # avg_data['r_error'] = np.array(avg_data['r_error']).T
        # avg_data['r_time'] = np.array(avg_data['r_time'])
        # return avg_data
        
        

    # def return_data(self):
    #     """
    #     Returns self.data
    #     """
    #     return self.dataset











