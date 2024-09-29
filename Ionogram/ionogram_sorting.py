# -*- coding: utf-8 -*-
"""
Created on Sun Sep 29 11:56:05 2024

@author: Kian Sartipzadeh
"""


import os
import numpy as np



class IonogramSorting:
    def __init__(self):
        self.ionogram_dataset = {}
    
    
    
    
    def import_data(self, datapath: str):
        """
        This function handles ionogram data in form of text files that has been
        pre-processed by the "SAO explorer" software.
        
        Each of these text files consist of 24-hour worth of ionosonde measurements
        with 15 minutes interval per data update. In other words, each 15 min
        interval ("batch") has a time and a date header followed by the ionosonde measurements.
        Each measurement (one row) has 8 ionosonde features represented as the
        columns as such: [Freq  Range  Pol  MPA  Amp  Doppler  Az  Zn].
        
        The number of measurements (rows) per "batch" changes depending on whether
        or not the Ionosonde was able to receive a backscatter signal. So each
        "batch" can contain different number of measurements.
        
        
        Input (type)    | DESCRIPTION
        ------------------------------------------------
        datapath (str)  | Path to folder that contains original ionograms txt files
        
        Return (type)              | DESCRIPTION
        ------------------------------------------------
        ionogram_data (np.ndarray) | Procrssed ionogram data
        ionogram_time (np.ndarray) | Timestamps of ionogram data
        """
        
        ionogram_time = []
        ionogram_data = []
        with open(datapath, "r") as file:
            
            lines = file.readlines() # Reading all lines in txt file
    
            data_batch = []
            for line in lines:
                
                """ # When encountering new header containing date and time (Ex: "2018.09.21 (264) 00:00:00.000") """
                if len(line) == 30:                                               # length of header containing date and time which is 30
                    iono_date = line[0:10]                                        # length of date (Ex: "2018.09.21" has length=10)
                    iono_time = f"{line[-13:-11]}-{line[-10:-8]}-{line[-7:-5]}"   # defining new time format (Ex: 20-15-00)
                    iono_datetime = f"{iono_date}_{iono_time}"                    # changing the format to be "yyyy.MM.dd_hh-mm-ss"
                    ionogram_time.append(iono_datetime)
                
                
                """ When encountering ionogram data (Ex: 3.400  315.0  90  24  33  -1.172 270.0  30.0) """
                if len(line) == 46:                             # length of each line containing ionogram values which is 46
                    line_split = line.split()                   # splitting strings in line by the whitespace between values e.g., ["3.14 0.4"] to ["3.14", "0.4"]
                    line_final= [float(x) for x in line_split]  # Converting strings to floats
                    data_batch.append(line_final)
                
                
                """ When encountering space between each batch of 15 min data """
                if len(line) == 1:                              # length of whitespace which is 1
                    ionogram_data.append(np.array(data_batch))  # appending the "batch" to the total data list 
                    data_batch = []                             # resetting the batch list 
                
                else:
                    continue
        
            # Converting list into np.ndarrays
            ionogram_time = np.array(ionogram_time, dtype=object)
            ionogram_data = np.array(ionogram_data, dtype=object)
        
        # Store ionogram_time and ionogram_data as key-value pairs in the dictionary
        for i, time in enumerate(ionogram_time):
            self.ionogram_dataset[time] = ionogram_data[i]

        return ionogram_time, ionogram_data
    
    
    
    def return_dataset(self):
        return self.ionogram_dataset








































