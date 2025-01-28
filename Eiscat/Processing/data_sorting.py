# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 10:26:02 2024

@author: Kian Sartipzadeh
"""



import os
import pickle
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
import warnings
from data_averaging import EISCATAverager




class EISCATDataSorter:
    """
    Class for importing and sorting EISCAT data. Here it is assumed that the
    data to be processed consists of .mat files where each file contains a
    whole day worth of measurements.
    """
    def __init__(self, folder_name: str):
        """
        Attributes (type)   | DESCRIPTION
        ------------------------------------------------
        folder_name (str)   | Name of folder containing all .mat data files.
        dataset     (dict)  | Dict for storing processed data.
        """
        self.full_path = os.path.abspath(os.path.join(os.getcwd(), folder_name))  # Find full dir path
        self.dataset = {}


    
    def get_file_paths(self) -> list:
        """
        Get the full dir path of each .mat data file.
        
        Return (type)                   | DESCRIPTION
        ------------------------------------------------
        file_paths (list[str, str,...]) | List containing full .mat path names
        """
        file_paths = [os.path.join(root, file) 
                      for root, dirs, files in os.walk(self.full_path) 
                      for file in files if file.endswith('.mat')]
        return file_paths
    
    
    
    def process_file(self, file: str) -> dict:
        """
        Load radar measurements from a single .mat data file.
        
        Input (type) | DESCRIPTION
        ------------------------------------------------
        file  (str)  | .mat file name
        
        Return (type) | DESCRIPTION
        ------------------------------------------------
        data  (dict)  | Dictionary containing data per wanted key
        """
        
        data = loadmat(file)  # importing .mat file as dict
        include = ["r_time", "r_h", "r_param", "r_error"]
        
        # includes keys in same order as in the include list
        data = {key: (data[key] if key == "r_time" else data[key].T) for key in include if key in data}
        return data
    
    
    
    def sort_data(self):
        """
        Sort the data from folder containing .mat data files.
        """
        files = self.get_file_paths()  # getting full path of .mat files
        
        for i, file in enumerate(files):
            data = self.process_file(file)           # open and convert matlab file to dict
            file_name = os.path.basename(file)[:-4]  # only get date from filename
            
            # Check if the data contains only zeros
            if 'r_param' in data and np.all(data['r_param'] == 0):
                warnings.warn(f"Data for {file_name} is corrupted (contains only zeros) and will be removed.")
                continue
            
            self.dataset[file_name] = data           # assign data to date of measurement
        
    
    
    
    def save_dataset(self, output_filename):
        """
        Saves dataset locally as a .pkl file.
        """
        with open(output_filename, 'wb') as file:
            pickle.dump(self.dataset, file)



    def return_data(self):
        """
        Returns the sorted dataset.
        """
        return self.dataset

