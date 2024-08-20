# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 15:06:46 2024

@author: Kian Sartipzadeh
"""


import os
import pickle
import numpy as np
from scipy.io import loadmat
from filter_data import DataFiltering

"""
To do:
    - Add more explanation to 'sort_data' method
"""




"""
dataset is a nested dictionary.   key_global: key_local1, key_local2
                                  key_local1: ndarray
"""



class EISCATDataSorter:
    """
    Class for importing and sorting EISCAT data.
    """
    def __init__(self, global_folder_path: str, global_folder_name: str):
        """
        Attributes (type)          | DESCRIPTION
        ------------------------------------------------
        global_folder_path (str)   | Global path of folder containing EISCAT data
        global_folder_name (str)   | Name of folder containing EISCAT data
        dataset            (dict)  | Dict for storing data
        """
        self.global_folder_path = global_folder_path
        self.global_folder_name = global_folder_name
        self.dataset = {}  # for storing dataset
    
    
    def get_subfolder_names(self):
        """
        Get subfolder names in the global folder. Each subfolder represents
        a days worth of EISCAT measurements.
        
        Return (type)            | DESCRIPTION
        ------------------------------------------------
        subfolder_names (list)   | list containing subfolder names
        """
        subfolder_names = [os.path.join(self.global_folder_name, subdir) 
                           for root, dirs, files in os.walk(self.global_folder_path) 
                           for subdir in dirs]
        return subfolder_names



    def process_file(self, file: str):
        """
        Load and filter data from a single matlab file
        
        Input (type) | DESCRIPTION
        ------------------------------------------------
        file  (str)  | matlab file name
        
        Return (type) | DESCRIPTION
        ------------------------------------------------
        data  (dict)  | Dictionary containing data from wanted keys
        """
        data = loadmat(file)  # importing matlab file as dictionary
        include = ["r_time", "r_h", "r_param", "r_error"]  # keys to include
        data = {key: data[key] for key in data if key in include}
        return data
    
    

    def sort_data(self, save_data: bool=False):
        """
        Sort the data from the global folder.
        """
        
        subfolder_names = self.get_subfolder_names()
        
        for i, subfolder in enumerate(subfolder_names):
            
            
            # Condition for reducing number of print statements
            if len(subfolder_names) > 100:
                print(f'Processed: {i+1} of {len(subfolder_names)} days')
            else:
                print(f'Processed: {i+1} of {len(subfolder_names)} days')
                
                
            # Getting matlab files from subfolder
            files = [filename.path for filename in os.scandir(subfolder)
                                if (filename.is_file()) and (filename.path[-3:] == 'mat')]
            
            # Getting data stat at 15min interval
            files_to_process = [file for file in files if loadmat(file)["r_time"][0][4] % 15 == 0]
            
            day = subfolder[10:20]  # date of subfolder containing data
            self.dataset[day] = {'r_time': [], 'r_h': [], 'r_param': [], 'r_error': []}
            
            # Processing each matlab file
            for file in files_to_process:
                data = self.process_file(file)
                
                # Assigning data to corresponding key
                for key in self.dataset[day]:
                    self.dataset[day][key].append(np.array(data[key]))
                    
        if save_data == True:
            self.save_dataset()

    def save_dataset(self, output_file='testing_sorted_data.pkl'):
        with open(output_file, 'wb') as file:
            pickle.dump(self.dataset, file)

    
    def return_data(self):
            """Returns the sorted dataset."""
            return self.dataset
    
    




