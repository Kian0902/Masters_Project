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
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors



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
    
    
    
    def batch_detection(self, eval_method: str, show_plot=False, save_plot=False):
        
        # print(show_plot, save_plot)
        
        for key in self.dataset.keys():
            print(f"Date: {key}")
            if eval_method == "Normalized Residuals":
                self.residual_norm(self.dataset[key], self.fitted_dataset[key], show_plot=show_plot, save_plot=save_plot)
            
            if eval_method == "Chi-square":
                self.residual_norm(self.dataset[key], self.fitted_dataset[key], show_plot=show_plot, save_plot=save_plot)
            
        
    
    def residual_norm(self, data, data_fit, show_plot=False, save_plot=False):
        # True data
        t  =  self._to_datetime(data['r_time'])
        z        =  data['r_h'].flatten()
        ne       =  data['r_param']
        ne_error =  data['r_error']
        
        # Fitted data
        ne_fit   =  data_fit['r_param']
        
        # Normalized residuals
        residuals = (ne - ne_fit)/ne_error
        
        if show_plot:
            self.plot_eval(t, z, ne, ne_fit, residuals, "Normalized Residuals", show_plot=show_plot, save_plot=save_plot)
        
        return residuals
    
    
    def chi_square(self, data, data_fit, show_plot=False, save_plot=False):
        

        # True data
        t  =  self._to_datetime(data['r_time'])
        z        =  data['r_h'].flatten()
        ne       =  data['r_param']
        ne_error =  data['r_error']
        
        # Fitted data
        ne_fit   =  data_fit['r_param']
        
        # Normalized residuals
        residuals = (ne - ne_fit)/ne_error
        
        # Calculate the chi-square statistic
        chi_square_statistic = residuals**2
        
        if show_plot:
            # print(show_plot, save_plot)
            self.plot_eval(t, z, ne, ne_fit, chi_square_statistic, "Chi-square", show_plot=show_plot, save_plot=save_plot)
        
        return chi_square_statistic
    
    
    
    
    
    def plot_eval(self, t, z, ne, ne_fit, eval_metric, eval_method, show_plot=False, save_plot=False):
        
        if eval_method == "Normalized Residuals":
            cmap = "bwr"
        else:
            cmap = "inferno"
        
        vmin = 1e10
        vmax = 1e12
        
        
        # Create the figure and GridSpec layout
        
        fig = plt.figure(figsize=(25/2.54, 20/2.54))
        gs = gridspec.GridSpec(2, 2, width_ratios=[2, 2], height_ratios=[1, 1])
        
        
        date_str = t[0].strftime('%Y-%m-%d')
        fig.suptitle(f'Date: {date_str}', fontsize=16)
        
        # ne plot
        ax0 = fig.add_subplot(gs[0, 0])
        c0  = ax0.pcolormesh(t, z, ne, norm=mcolors.LogNorm(vmin=vmin, vmax=vmax), cmap='turbo')
        ax0.set_title("True")
        ax0.xaxis.set_major_formatter(DateFormatter('%d %H:%M'))
        ax0.set_xticklabels([])
        fig.colorbar(c0, ax=ax0)
        
        # ne_fit plot
        ax1 = fig.add_subplot(gs[1, 0])
        c1  = ax1.pcolormesh(t, z, ne_fit, norm=mcolors.LogNorm(vmin=vmin, vmax=vmax), cmap='turbo')
        ax1.set_title("Fitted")
        ax1.xaxis.set_major_formatter(DateFormatter('%d %H:%M'))
        fig.colorbar(c1, ax=ax1)
        
        # residual plot
        ax2 = fig.add_subplot(gs[:, 1])  # spans both rows
        c2  = ax2.pcolormesh(t, z, eval_metric, cmap=cmap, vmin=-10, vmax=10)
        ax2.set_title(f"{eval_method}")
        ax2.xaxis.set_major_formatter(DateFormatter('%d %H:%M'))
        fig.colorbar(c2, ax=ax2)
        
        # Adjust layout to prevent overlapping
        fig.tight_layout()
        fig.autofmt_xdate()
        
        
        
        if save_plot:
            file_name= t[0].strftime('%Y_%m_%d')
            plt.savefig("Curvefitting_plots/vhf/" + file_name, bbox_inches='tight')
            
        if show_plot:
            plt.show()
        # return fig
        
        
        



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

key_choise = list(X_org.keys())[:]


x_org = {key: X_org[key] for key in key_choise}
x_fit = {key: X_fit[key] for key in key_choise}



E = CurvefittingEvaluation(x_org, x_fit)
E.batch_detection(eval_method="Normalized Residuals", show_plot=True, save_plot=True)
# E.residual_norm(x_org, x_fit, show_plot=True, save_plot=True)
# E.chi_square(x_org, x_fit, show_plot=True)


















