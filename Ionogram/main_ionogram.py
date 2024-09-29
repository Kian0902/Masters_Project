# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 12:37:28 2024

@author: Kian Sartipzadeh
"""



import os
import numpy as np
import matplotlib.pyplot as plt
from ionogram_sorting import IonogramSorting
from process_ionograms import ionogram_processing







# resultpath = "justmadeiono"
datapath_folder = "TXT"


for file in os.listdir(datapath_folder):
    
    
    file_path = os.path.join(datapath_folder, file)
    
    
    
    A = IonogramSorting()
    times, data = A.import_data(file_path)
    # data = A.return_dataset()
    
    t, x = times[100:115], data[100:115]
    
    ionogram_processing(x, t, plot=True)
    
    
    # print(data.shape, times.shape)
    
    
    

    break












































