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
    
    
    
    def process_file(self, file: str, testing: bool=False) -> dict:
        """
        Load radar measurements from a single .mat data file.
         -   This method also has a testing mode which is activated if 
             'test_dataflow' is called upon.
        
        Input (type) | DESCRIPTION
        ------------------------------------------------
        file  (str)  | .mat file name
        
        Return (type) | DESCRIPTION
        ------------------------------------------------
        data  (dict)  | Dictionary containing data per wanted key
        """
        
        # ===========================================
        #                 Testing mode
        if testing is True:
            print("Keys before processing:")
            for key in loadmat(file):
                print(f" - {key}")
            print("\n")
        # ===========================================
        
        data = loadmat(file)  # importing .mat file as dict
        include = ["r_time", "r_h", "r_param", "r_error", "r_systemp"]
        
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





    def test_dataflow(self, return_data: bool=False):
        """
        Tests the dataflow through the entire process using one file.
        Prints the type and shape of the data at each step.
        """
        files = self.get_file_paths()
        
        if not files:
            return print("No .mat files found in the specified directory.")
        
        
        test_file = files[0]
        data = self.process_file(test_file, testing=True)
        self.dataset = {}  # Reset the dataset for the test
        self.dataset[os.path.basename(test_file)[:-4]] = data
        final_data = self.return_data()
        
        
        if return_data is True:
            return final_data
        
        else:
            # Use the first file for testing
            test_file = files[0]
            print(f"Testing with file: {os.path.basename(test_file)}")
            
            
            # Step 1 get_file path
            print("\nStep 1: get_file_paths()\n")
            print(f"Output Type: {type(files)}")
            print(f"Output Length: {len(files)}")
            print(f"First File Path: {files[0] if files else 'None'}")
            print("_____________________________________________")
            
            
            # Step 2 process_file
            print("\nStep 2: process_file()\n")
            data = self.process_file(test_file, testing=True)
            print(f"Input Type: {type(test_file)}")
            print(f"Input: {test_file}")
            print(f"Output Type: {type(data)}")
            print(f"Output Keys: {list(data.keys())}")
            for key in data:
                print(f" - {key}: Shape = {data[key].shape}")
            print("_____________________________________________")
            
            
            # Step 3 sort_data
            print("\nStep 3: sort_data()\n")
            self.dataset = {}  # Reset the dataset for the test
            self.dataset[os.path.basename(test_file)[:-4]] = data
            print(f"Dataset Keys: {list(self.dataset.keys())}")
            for key in self.dataset:
                print(f" - {key}: Type = {type(self.dataset[key])}")
                for k in self.dataset[key]:
                    print(f" - {k}: Shape = {self.dataset[key][k].shape}")
                    
                    
                    
            print("_____________________________________________")
            # Return the sorted data
            print("\nFinal Data:")
            final_data = self.return_data()
            print(f"Type: {type(final_data)}")
            print(f"Number of Entries: {len(final_data)}")
            for key in final_data:
                print(f" - {key}: Type = {type(final_data[key])}, Keys = {list(final_data[key].keys())}")
            
                
