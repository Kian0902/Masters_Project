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




class EISCATDataSorter:
    """
    Class for importing and sorting EISCAT data.
    """
    def __init__(self, folder_name: str):
        """
        Attributes (type)          | DESCRIPTION
        ------------------------------------------------
        folder_name (str)   | Name of folder containing .mat data files
        dataset     (dict)  | Dict for storing data
        """
        self.full_path = os.path.abspath(os.path.join(os.getcwd(), folder_name))  # Find full dir path
        self.dataset = {}



    def get_file_paths(self):
        """
        Getting the full dir path of each .mat data file.
        
        Return (type)            | DESCRIPTION
        ------------------------------------------------
        file_paths (list[str])   | List containing full .mat path names
        """
        
        file_paths = [os.path.join(root, file) 
                      for root, dirs, files in os.walk(self.full_path) 
                      for file in files if file.endswith('.mat')]
        
        return file_paths



    def process_file(self, file: str):
        """
        Load data from a single .mat data file
        
        Input (type) | DESCRIPTION
        ------------------------------------------------
        file  (str)  | matlab file name
        
        Return (type) | DESCRIPTION
        ------------------------------------------------
        data  (dict)  | Dictionary containing data per wanted key
        """

        data = loadmat(file)  # importing .mat file as dict
        include = ["r_h", "r_param", "r_error"]
        data = {key: data[key] for key in data if key in include}
        
        # if shapes are different
        if data['r_h'].shape != data["r_param"].shape:
            data['r_h'] = np.tile(data['r_h'], (1, data["r_param"].shape[0])).T  # copying r_h such that is has the same shape as r_param
        else:
            pass
        
        return data
    
    
    
    def sort_data(self, save_data: bool=False):
        """
        Sort the data from folder containing .mat data files.
        """
        
        files = self.get_file_paths()  # getting full path of .mat files
        
        
        for i, file in enumerate(files):

            data = self.process_file(file)       # open and convert matlab file to dict
            
            file_name = os.path.basename(file)[:-4]  # only get date from filename
            self.dataset[file_name] = data           # assign data to date of measurement


    def return_data(self):
        """Returns the sorted dataset."""
        return self.dataset





# Use the local folder name instead of the full path
folder_name = "Ne"
A = EISCATDataSorter(folder_name)

a = A.sort_data()

print(len(A.dataset.keys()))
print(A.dataset['2018-11-10']['r_param'].shape)

# file_paths = A.get_file_paths()

# A.sort_data()


# class EISCATDataSorter:
#     """
#     Class for importing and sorting EISCAT data.
#     """
#     def __init__(self, full_path_to_folder: str):

        
#         self.full_path = full_path_to_folder
#         self.dataset = {}  # for storing dataset
        
#     def get_file_paths(self):

#         file_paths = [os.path.join(root, file) 
#                       for root, dirs, files in os.walk(self.full_path) 
#                       for file in files if file.endswith('.mat')]
        
        
#         return file_paths

#     def process_file(self, file: str):

#         data = loadmat(file)  # importing matlab file as dictionary
#         include = ["r_h", "r_param", "r_error"]  # keys to include
#         data = {key: data[key] for key in data if key in include}
        
#         return data
    
    
    
#     def sort_data(self, save_data: bool=False):
        
#         files = self.get_file_paths()
        
#         for i, file in enumerate(files):
            
#             data = self.process_file(file)
            
                    
                    
#     def return_data(self):
#             """Returns the sorted dataset."""
#             return self.dataset






# # Main path to Folder containing EISCAT data
# path = "C:\\Users\\kian0\\OneDrive\\Desktop\\UiT Courses\\FYS_3931\\Scripts\\Masters_Project\\Eiscat\\Processing\\Ne"


# folder = os.listdir(path)
# A = EISCATDataSorter(path)

# file_paths = A.get_file_paths()


# A.sort_data()

# file_paths = [os.path.join(root, file) 
#               for root, dirs, files in os.walk(path) 
#               for file in files if file.endswith('.mat')]

# First file
# f = file_paths[0]

# Convert .mat to dict
# data = A.process_file(f)

# ne = data['r_param'].T
# z  = data['r_h']

# print(ne.shape)
# print(z.shape)

# plt.plot(ne[:,0:10], z)
# plt.xscale("log")
# plt.show()





























