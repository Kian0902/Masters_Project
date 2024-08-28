# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 11:04:56 2024

@author: Kian Sartipzadeh
"""

import numpy as np
from scipy.stats import zscore
from datetime import datetime
import matplotlib.pyplot as plt



class OutlierDetection:
    """
    Class for detecting outliers present in EISCAT data.
    """
    def __init__(self, dataset: dict):
        """
        Initialize with dataset containing EISCAT dat from one day.
        
        Attributes (type) | DESCRIPTION
        ------------------------------------------------
        dataset (dict)    | Dictionary containing the NaN filtered EISCAT data
        """
        self.dataset = dataset
        self.detection_methods = {'z-score': self.z_score_method,
                                  'IQR': self.iqr_method}
    
    
    def z_score_method(self, data: np.array, threshold: int=3):
        """
        Detect outliers using the Z-score method.
        
        Input (type)        | DESCRIPTION
        ------------------------------------------------
        data  (np.ndarray)  | Data from one key to be analyzed.
        threshold (int)     | Z-score threshold for identifying outliers (default is 3).
        
        
        Return (type)                    | DESCRIPTION
        ------------------------------------------------
        detected_outliers  (np.ndarray)  | Boolean array where True indicates an outlier.
        """
        z_score = zscore(data, axis=0)  # get z-scores
        detected_outliers = np.abs(z_score) > threshold
        return detected_outliers
    
    
    def iqr_method(self, data: np.array, lower_percent: int=5, upper_percent: int=95):
        """
        Detect outliers using the Interquartile Range (IQR) method.
    
        Input (type)             | DESCRIPTION
        ------------------------------------------------
        data (np.ndarray)        | Data from one key to be analyzed.
        lower_percent (int)      | Percentile to determine the lower bound (default is 5).
        upper_percent (int)      | Percentile to determine the upper bound (default is 95).
    
        Return (type)                    | DESCRIPTION
        ------------------------------------------------
        detected_outliers (np.ndarray)   | Boolean array where True indicates an outlier.
        """
        
        Q1 = np.percentile(data, lower_percent, axis=0)
        Q3 = np.percentile(data, upper_percent, axis=0)
        
        
        IQR = Q3 - Q1
        
        
        lower_fence = Q1 - 1.5*IQR
        upper_fence = Q1 + 1.5*IQR        
    
        return (data < lower_fence) | (data > upper_fence)
    
    
    def detect_outliers(self, method_name: str, plot_outliers: bool=False):
        """
        Detects outliers using the specified method.
        """
        
        if method_name not in self.detection_methods:
            raise ValueError(f"Method {method_name} not recognized.")
        
        
        
        r_time  = self.dataset['r_time']
        r_h     = self.dataset['r_h']
        r_param = self.dataset['r_param']
        r_error = self.dataset['r_error']
        
        # print('\n')
        # print(f'r_time:  {r_time.shape}')
        # print(f'r_h:     {r_h.shape}')
        # print(f'r_param: {r_param.shape}')
        # print(f'r_error: {r_error.shape}')
        # print('\n')
        
        
        r_param = self.dataset['r_param']
        outliers = self.detection_methods[method_name](r_param[:, :])
        
        # Find indices of minutes (rows) where any outlier is detected
        minutes_with_outliers = np.any(outliers, axis=0)
        outlier_indices = np.where(minutes_with_outliers)[0]
        
        # print(outlier_indices)
        
        # Option for plotting the outliers
        if plot_outliers is True:
            for i, ind in enumerate(outlier_indices):
                plt.plot(r_param[:, ind], self.dataset['r_h'].flatten(), label='ne')
                plt.xlabel('Electron Density')
                plt.ylabel('Altitude')
                plt.title(f'Outlier Detected using {method_name}   time: {datetime(*self.dataset["r_time"][ind])}')
                plt.xscale('log')
                plt.legend()
                plt.show()
        
        
        
        
        
        
        
        
        
















