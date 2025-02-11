# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 12:37:28 2024

@author: Kian Sartipzadeh
"""



import os
import numpy as np
import matplotlib.pyplot as plt
from ionogram_sorting import IonogramSorting
from iono_utils import save_dict, load_dict, inspect_dict


from process_ionograms import IonogramProcessing
from tqdm import tqdm



# X = load_dict("sorted_ionogram_dicts/2019-1-1.pkl")



datapath_folder = "Iono_TXT"


for file in tqdm(os.listdir(datapath_folder)):
    
    print(f"Processing data from {file[6:12]}\n")
    
    file_path = os.path.join(datapath_folder, file)
    
    A = IonogramSorting()
    print("-Sorting Mothly file into 15 min ionogram samples...")
    A.import_data(file_path)
    A.save_as_dict(folder_path="sorted_ionogram_dicts")
    # A.save_dataset()
    # data = A.return_dataset()
    # times, data = A.import_data(file_path)
    print("-Sorting Complete!\n")
    # break
    
    # B = IonogramProcessing()
    # print("-Making Ionogram images...")
    # # B.process_ionogram(data, times, plot=False, result_path="Ionogram_Images")
    # B.process_ionogram(data, times, plot=True)
    # print("-Making Ionograms Complete!\n")
    # print("==========================================================")
    # break











































