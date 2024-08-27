# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 15:58:50 2024

@author: Kian Sartipzadeh
"""




import numpy as np
import matplotlib.pyplot as plt



class EISCATAverager:
    """
    Class for averaging EISCAT radar measurement data over specified time intervals.
    """
    def __init__(self, data: dict):
        """
        Initialize with data to be averaged.
        
        Attributes (type)    | DESCRIPTION
        ------------------------------------------------
        data (dict)          | Dictionary containing the EISCAT data to be averaged.
        """
        self.data = data


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

        
        r_time  = self.data['r_time']
        r_h     = self.data['r_h']
        r_param = self.data['r_param']
        r_error = self.data['r_error']
        
        
        
        avg_data = {'r_time': [],
            'r_h': r_h,
            'r_param': [],
            'r_error': []}
        
        # for i, time in enumerate(r_time):
            
        #     if time[4] % 15 == 0:
        #         print(i, time)
        
        
        # get indices of 15 min intervals
        time15_ind = np.where(r_time[:, 4] % 15 == 0)[0]

        print(f'Num 15min intervals:  {time15_ind.shape}')

        # print(r_param.T.shape)
        
        for i in range(0, len(time15_ind) - 1):
            
            ind_s = time15_ind[i]
            ind_f = time15_ind[i + 1]
            
            
            # print(ind_s, ind_f)
            
            # print(r_param[:, ind_s: ind_f])
            # print(r_param[:, ind_s: ind_f].shape)
            
            r_param_avg = np.mean(r_param[:, ind_s: ind_f], axis=1)
            r_error_avg = np.mean(r_error[:, ind_s: ind_f], axis=1)
            # print(r_param_avg.shape)
            avg_data['r_param'].append(r_param_avg)
            avg_data['r_error'].append(r_error_avg)
            
            avg_data['r_time'].append(r_time[ind_f])
            
        avg_data['r_param'] = np.array(avg_data['r_param']).T
        avg_data['r_error'] = np.array(avg_data['r_error']).T
        avg_data['r_time'] = np.array(avg_data['r_time'])
        
        return avg_data
        
        
        
        
            # plt.plot(r_param[:, ind_s: ind_f], r_h)
            # plt.show()
            
            # plt.plot(r_param_avg, r_h)
            # plt.show()
            
            
            
            
    
            
            
            
            
















            







