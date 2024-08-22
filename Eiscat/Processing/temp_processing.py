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
        
        self.full_path = os.path.abspath(os.path.join(os.getcwd(), folder_name))
        self.dataset = {}  # for storing dataset



    def get_file_paths(self):
        
        file_paths = [os.path.join(root, file) 
                      for root, dirs, files in os.walk(self.full_path) 
                      for file in files if file.endswith('.mat')]
        
        return file_paths



    def process_file(self, file: str):
        print("h")
        data = loadmat(file)  # importing matlab file as dictionary
        include = ["r_h", "r_param", "r_error"]  # keys to include
        data = {key: data[key] for key in data if key in include}
        
        return data




    def return_data(self):
        """Returns the sorted dataset."""
        return self.dataset


# Use the local folder name instead of the full path
folder_name = "Ne"
A = EISCATDataSorter(folder_name)

a = A.get_file_paths()

print(A.dataset)

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





























