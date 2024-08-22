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












# Main path to Folder containing EISCAT data
path = "C:\\Users\\kian0\\OneDrive\\Desktop\\UiT Courses\\FYS_3931\\Scripts\\Masters_Project\\Eiscat\\Processing\\Ne"


folder = os.listdir(path)

file_paths = [os.path.join(root, file) 
              for root, dirs, files in os.walk(path) 
              for file in files if file.endswith('.mat')]

# First file
f = file_paths[0]

# Convert .mat to dict
data = loadmat(f)

ne = data['r_param'].T
z  = data['r_h']

print(ne.shape)
print(z.shape)

plt.plot(ne[:,0:10], z)
plt.xscale("log")
plt.show()





























