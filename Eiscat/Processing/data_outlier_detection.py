# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 11:04:56 2024

@author: Kian Sartipzadeh
"""

import numpy as np
from scipy.stats import zscore


class OutlierDetection:
    """
    Class for detecting outliers present in EISCAT data.
    """
    def __init__(self, data: dict):
        """
        Initialize with data.
        
        Attributes (type)    | DESCRIPTION
        ------------------------------------------------
        data (dict)          | Dictionary containing the NaN filtered EISCAT data
        """
        self.data = data
    
    
    
    
    
        
    def detect_outliers(self):
        
        
        
        
        r_time  = self.data['r_time']
        r_h     = self.data['r_h']
        r_param = self.data['r_param']
        r_error = self.data['r_error']
        
        print('\n')
        print(f'r_time:  {r_time.shape}')
        print(f'r_h:     {r_h.shape}')
        print(f'r_param: {r_param.shape}')
        print(f'r_error: {r_error.shape}')
        print('\n')
        
        
        
        
        
        
        



















