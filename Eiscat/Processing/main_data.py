# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 14:13:45 2024

@author: Kian Sartipzadeh
"""


import numpy as np
import pickle
import matplotlib.pyplot as plt


from read_EISCAT_data import EISCATDataProcessor
from data_sorting import EISCATDataSorter
from data_averaging import EISCATAverager
from data_filtering import EISCATDataFilter
from data_outlier_detection import EISCATOutlierDetection




def save_data(dataset: dict, file_name: str):
    with open(file_name, 'wb') as file:
        pickle.dump(dataset, file)



def detect_nan_in_arrays(data_dict):
    for key, array in data_dict.items():
        if np.isnan(array).any():
            print(f"NaN detected {key}: {array.shape}")


# Use the local folder name containing data
folder_name = "Ne_uhf"


# Sorting data
Esicat = EISCATDataSorter(folder_name)
Esicat.sort_data()  # sort data


# VHF data
X_Esicat = Esicat.return_data()  # returning dict data



# Clipping range and Filtering data for nan
filt = EISCATDataFilter(X_Esicat, filt_range=True, filt_nan=True) 
filt.batch_filtering()
X_filtered = filt.return_data()



key_choise = list(X_filtered.keys())[:]
X = {key: X_filtered[key] for key in key_choise}

# print(X['2021-3-10']['r_time'])


# Detecting outliers
Outlier = EISCATOutlierDetection(X)
Outlier.batch_detection(method_name="IQR", save_plot=False)
X_outliers = Outlier.return_outliers()


# Filtering outliers
outlier_filter = EISCATDataFilter(X, filt_outlier=True)
outlier_filter.batch_filtering(dataset_outliers=X_outliers, filter_size=3, plot_after_each_day=False)
X_outliers_filtered = outlier_filter.return_data()


# Averaging data
AVG = EISCATAverager(X_outliers_filtered)
AVG.batch_averaging(save_plot=True, weighted=False)
X_avg = AVG.return_data()


# print(X_avg['2021-3-10']['r_time'])


save_data(X_avg, file_name=folder_name + "_avg")





# guisdap_data_folder_name = "beata_vhf"
# result_folder_name = "Ne_vhf"


# process_data = EISCATDataProcessor(guisdap_data_folder_name, result_folder_name)
# process_data.process_all_files()






