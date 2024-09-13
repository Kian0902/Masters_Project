# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 12:19:21 2024

@author: Kian Sartipzadeh
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt

from datetime import datetime, timedelta
from matplotlib.dates import DateFormatter


class CurvefittingEvaluation:
    """
    Class for evaluating the double chapman curvefitting model performance.
    
    Methods:
        - residual_norm
        - chi_square
        - rmse
    """
    def __init__(self, dataset, fitted_dataset):
        
        self.dataset = dataset
        self.fitted_dataset = fitted_dataset
    
    
    def _to_datetime(self, time_data):
        
        time = np.array([datetime(year, month, day, hour, minute) 
                         for year, month, day, hour, minute, second in time_data])
        return time
    
    
    
    def residual_norm(self, data, data_fit, plot=False):
        

        # True data
        t  =  self._to_datetime(data['r_time'])
        z        =  data['r_h'].flatten()
        ne       =  data['r_param']
        ne_error =  data['r_error']
        
        # Fitted data
        ne_fit   =  data_fit['r_param']
        
        # Normalized residuals
        residuals = (ne - ne_fit)/ne_error
        
        
        print(t.shape, z.shape, residuals.shape, ne_error.shape)
        
        if plot:
            plt.pcolormesh(t, z, residuals)
            plt.show()
        
        return residuals
        
    def plot_eval(self):
        ...




def import_file(file_name):
    with open(file_name, 'rb') as f:
        dataset = pickle.load(f)
    
    return dataset






# Import files
file_org = "Ne_vhf_avg"
file_fit = "Ne_vhf_avg_lmfit_curvefits"

# Get datasets
X_org = import_file(file_org)
X_fit = import_file(file_fit)

key_choise = list(X_org.keys())[:1]


x_org = {key: X_org[key] for key in key_choise}['2018-11-10']
x_fit = {key: X_fit[key] for key in key_choise}['2018-11-10']



E = CurvefittingEvaluation(X_org, X_fit)
E.residual_norm(x_org, x_fit, plot=True)



















