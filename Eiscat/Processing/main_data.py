# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 14:13:45 2024

@author: Kian Sartipzadeh
"""

import copy
from read_EISCAT_data import EISCATDataProcessor
from data_sorting import EISCATDataSorter
from data_averaging import EISCATAverager
from data_filtering import EISCATDataFilter
from data_plotting import EISCATPlotter, plot_available_data3
from data_outlier_detection import EISCATOutlierDetection
from data_utils import save_dict, load_dict, inspect_dict, plot_compare_all


def execute_tot_process(folder_name):
    # __________ Sorting data __________ 
    Eiscat = EISCATDataSorter(folder_name)
    Eiscat.sort_data()
    X_eis = Eiscat.return_data()
    X_EIS = copy.deepcopy(X_eis)
    
    # __________ Clipping range and Filtering data for nan __________ 
    filt = EISCATDataFilter(X_eis, filt_range=True, filt_nan=True, filt_interpolate=True) 
    filt.batch_filtering(num_of_ref_alt=27)
    X_filt = filt.return_data()
    
    
    # __________ Detecting outliers __________ 
    Outlier = EISCATOutlierDetection(X_filt)
    Outlier.batch_detection(method_name="IQR", save_plot=False)
    X_outliers = Outlier.return_outliers()
    
    
    # __________ Filtering outliers __________ 
    outlier_filter = EISCATDataFilter(X_filt, filt_outlier=True)
    outlier_filter.batch_filtering(dataset_outliers=X_outliers, filter_size=5, plot_after_each_day=False)
    X_outliers_filtered = outlier_filter.return_data()
    X_OUT_FILT = copy.deepcopy(X_outliers_filtered)
    
    # __________  Averaging data __________ 
    AVG = EISCATAverager(X_outliers_filtered, plot_result=False)
    AVG.average_15min()
    X_avg = AVG.return_data()
    return X_EIS, X_OUT_FILT, X_avg

# Use the local folder name containing data
# folder_name_in  = "EISCAT_Madrigal/2012"
folder_name_out = "EISCAT_MAT/All"


X_eis, X_filt, X_avg = execute_tot_process(folder_name_out)
# plot_compare_all(X_eis, X_filt, X_avg, '2022-1-30', True)

X_test_avg = execute_tot_process("EISCAT_MAT/All_test_sunspots")[-1]

plot_available_data3(X_avg, X_test_avg)





# # Extract info from hdf5 files
# madrigal_processor = EISCATDataProcessor(folder_name_in, folder_name_out)
# madrigal_processor.process_all_files()



# # __________ Sorting data __________ 
# Eiscat = EISCATDataSorter(folder_name_out)
# Eiscat.sort_data()
# X_eis = Eiscat.return_data()




# # __________ Clipping range and Filtering data for nan __________ 
# filt = EISCATDataFilter(X_eis, filt_range=True, filt_nan=True, filt_interpolate=True) 
# filt.batch_filtering(num_of_ref_alt=27)
# X_filt = filt.return_data()



# # __________ Detecting outliers __________ 
# Outlier = EISCATOutlierDetection(X_filt)
# Outlier.batch_detection(method_name="IQR", save_plot=False)
# X_outliers = Outlier.return_outliers()



# # __________ Filtering outliers __________ 
# outlier_filter = EISCATDataFilter(X_filt, filt_outlier=True)
# outlier_filter.batch_filtering(dataset_outliers=X_outliers, filter_size=5, plot_after_each_day=False)
# X_outliers_filtered = outlier_filter.return_data()


# # save_dict(X_outliers_filtered, file_name="X_outliers_filtered")



# # __________  Averaging data __________ 
# AVG = EISCATAverager(X_outliers_filtered, plot_result=False)
# AVG.average_15min()
# X_avg = AVG.return_data()

# plot_available_data(X_avg)

# # save_dict(X_avg, file_name="X_eiscat_new_test_data")


