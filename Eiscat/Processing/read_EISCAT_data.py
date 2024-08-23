# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 14:17:03 2024

@author: Kian Sartipzadeh
"""







import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import scipy.io as sio



# 1 - Define paths
# Define the path to the data folder (where the input HDF5 files are stored)
datapath = "C:\\Users\\kian0\\OneDrive\\Desktop\\UiT Courses\\FYS_3931\\Scripts\\Masters_Project\\Eiscat\\Processing\\beata_vhf"
# Define the path to the results folder (where the output will be saved)
resultpath = "C:\\Users\\kian0\\OneDrive\\Desktop\\UiT Courses\\FYS_3931\\Scripts\\Masters_Project\\Eiscat\\Processing\\Ne_new"

# 2 - Find datafiles
# Change the current directory to the data folder
os.chdir(datapath)
# Get a list of all HDF5 files in the directory
datafiles = [f for f in os.listdir(datapath) if f.endswith('.hdf5')]

# 3 - Read files

# Get the total number of experiments/files
nE = len(datafiles)

# Loop through each experiment (file)
for iE in range(0, nE):
    print(f'File {iE+1}/{nE}')

    os.chdir(datapath)
    file = datafiles[iE]
    

























