# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 12:37:28 2024

@author: Kian Sartipzadeh
"""



import os
import numpy as np
import matplotlib.pyplot as plt
from ionogram_sorting import IonogramSorting
# from process_ionograms import IonogramProcessing
from tqdm import tqdm







datapath_folder = "Ionogram_TXT"


for file in tqdm(os.listdir(datapath_folder)):
    
    print(f"Processing data from {file[6:12]}\n")
    
    file_path = os.path.join(datapath_folder, file)
    
    A = IonogramSorting()
    print("-Sorting Mothly file into 15 min ionogram samples...")
    A.import_data(file_path)
    # A.save_dataset()
    # data = A.return_dataset()
    # times, data = A.import_data(file_path)
    print("-Sorting Complete!\n")
    
    # B = IonogramProcessing()
    # print("-Making Ionogram images...")
    # # B.process_ionogram(data, times, plot=False, result_path="Ionogram_Images")
    # B.process_ionogram(data, times, plot=True)
    # print("-Making Ionograms Complete!\n")
    # print("==========================================================")
    # break











































