# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 15:58:50 2024

@author: Kian Sartipzadeh
"""




import numpy as np




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






























