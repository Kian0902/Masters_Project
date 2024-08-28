# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 11:04:56 2024

@author: Kian Sartipzadeh
"""

import numpy as np
from scipy.stats import zscore
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
    
    
    
    def get_zscore(self, data: np.array, threshold: int=3):
        """
        Get the z-score of the data with a standard threshold of Z=3.
        
        Input (type)        | DESCRIPTION
        ------------------------------------------------
        data  (np.ndarray)  | Data from one key to be analyzed.
        threshold (int)     | Z value threshold
        
        Return (type)                    | DESCRIPTION
        ------------------------------------------------
        detected_outliers  (np.ndarray)  | Array with bool values of detected outliers. True if detected.
        """
        z_score = zscore(data, axis=0)  # get z-scores
        detected_outliers = np.abs(z_score) > threshold
        return detected_outliers
    
    

    
    def detect_outliers(self):
        
        
        
        
        r_time  = self.dataset['r_time']
        r_h     = self.dataset['r_h']
        r_param = self.dataset['r_param']
        r_error = self.dataset['r_error']
        
        print('\n')
        print(f'r_time:  {r_time.shape}')
        print(f'r_h:     {r_h.shape}')
        print(f'r_param: {r_param.shape}')
        print(f'r_error: {r_error.shape}')
        print('\n')
        
        
        ne = r_param[:, :]
        
        plt.plot(ne, r_h.flatten())
        plt.show()
        
        z_outlier = self.get_zscore(ne)
        
        
        outlier_min = np.any(z_outlier, axis=0)
        
        ind_outlier_min = np.where(outlier_min)[0]
        
        
        print(ind_outlier_min)
        
        
        ne_outlier = ne[:, ind_outlier_min]
        
        print(ne_outlier)
        
        for i, ne in enumerate(ne_outlier.T):
            # print(ne.shape)
            plt.plot(ne, r_h.flatten())
            plt.xscale("log")
            plt.show()
        
        
        
        # for i in np.arange(0, len(z_scores)):
        #     print(ne[i], z_scores[i])
        
        
        
        



















