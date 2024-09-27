# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 12:37:28 2024

@author: Kian Sartipzadeh
"""



import os
import numpy as np
import matplotlib.pyplot as plt
from process_ionograms import import_data, ionogram_processing







# resultpath = "justmadeiono"
datapath_folder = "ionograms_txt_data"


for file in os.listdir(datapath_folder):
    
    
    file_path = os.path.join(datapath_folder, file)

    times, data = import_data(file_path)
    
    print(f'Date {times[0][:10]}')
    
    # print(data.shape, times.shape)
    
    
    ionogram_processing(data, times, plot=True)

    break












































