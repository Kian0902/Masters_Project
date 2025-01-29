# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 14:13:45 2024

@author: Kian Sartipzadeh
"""


import numpy as np
import pickle
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from datetime import datetime, timedelta
from read_EISCAT_data import EISCATDataProcessor
from data_sorting import EISCATDataSorter
from data_averaging import EISCATAverager
from data_filtering import EISCATDataFilter
from data_plotting import EISCATPlotter
from data_outlier_detection import EISCATOutlierDetection
from matplotlib.dates import DateFormatter
import matplotlib.colors as colors
from data_utils import from_array_to_datetime, inspect_dict, get_day, get_day_data, MatchingFiles, from_strings_to_datetime, save_dict
from scipy.interpolate import interp1d
import numpy as np



def plot_day(data):
    r_time = from_array_to_datetime(data['r_time'])
    r_h = data['r_h'].flatten()
    r_param = data['r_param']
    r_error = data['r_error'] 
    
    
    fig, ax = plt.subplots()
    
    ne=ax.pcolormesh(r_time, r_h, r_param, shading="auto", cmap="turbo", norm=colors.LogNorm(vmin=1e10, vmax=1e11))
    ax.set_xlabel("Time (UT)")
    ax.set_ylabel("Altitudes (km)")
    ax.xaxis.set_major_formatter(DateFormatter('%H:%M'))
    fig.colorbar(ne, ax=ax, orientation='vertical')
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='center')
    plt.show()


def plot_sample(data, j):
    r_time = from_array_to_datetime(data['r_time'])
    r_h = data['r_h'].flatten()
    r_param = data['r_param']
    r_error = data['r_error'] 
    
    
    for i in range(0, len(r_time)):
        
        plt.plot(r_param[:, i], r_h)
        plt.errorbar(r_param[:, i], r_h, xerr=r_error[:, i])
        plt.xscale("log")
        plt.show()
        
        if i == j:
            break
    
    
    
# Use the local folder name containing data
# folder_name_in  = "EISCAT_Madrigal/Example_UHF"
# folder_name_out = "EISCAT_MAT/Example_UHF_out"

# # Extract info from hdf5 files
# madrigal_processor = EISCATDataProcessor(folder_name_in, folder_name_out)
# madrigal_processor.process_all_files()



# VHF_folder = "EISCAT_MAT/VHF_All"
# both_folder = "EISCAT_MAT/EISCAT_test_data"

both_folder = "EISCAT_MAT/UHF_All"


# Match = MatchingFiles(VHF_folder, UHF_folder)
# Match.remove_matching_vhf_files()


# __________ Sorting data __________ 
Eiscat = EISCATDataSorter(both_folder)
Eiscat.sort_data()
X_Eiscat = Eiscat.return_data()



# __________ Clipping range and Filtering data for nan __________ 
filt = EISCATDataFilter(X_Eiscat, filt_range=True, filt_nan=True, filt_interpolate=True) 
filt.batch_filtering(plot_after_each_day=False, num_of_ref_alt=27)
X_filt = filt.return_data()



# __________ Detecting outliers __________ 
Outlier = EISCATOutlierDetection(X_filt)
Outlier.batch_detection(method_name="Z-score", save_plot=False)
# Outlier.pca_all(reduce_to_dim=2)
X_outliers = Outlier.return_outliers()


# __________ Filtering outliers __________ 
outlier_filter = EISCATDataFilter(X_filt, filt_outlier=True)
outlier_filter.batch_filtering(dataset_outliers=X_outliers, filter_size=3, plot_after_each_day=False)
X_outliers_filtered = outlier_filter.return_data()



# __________  Averaging data __________ 
AVG = EISCATAverager(X_outliers_filtered, plot_result=True)
AVG.average_15min()
X_avg = AVG.return_data()




# save_dict(X_avg, file_name="X_avg_test_new_errors")






























# =============================================================================
# Old Code
# =============================================================================


# arrays = [sub_array for sub_dict in X_avg.values() for key, sub_array in sub_dict.items() if key.endswith('r_param')]


# # Determine the target size for the first dimension (e.g., maximum or specific value)
# target_rows = int(np.median([arr.shape[0] for arr in arrays]))

# adjusted_arrays = []
# for arr in arrays:
#     current_rows, cols = arr.shape
    
#     if current_rows == target_rows:
#         # No need to adjust if it already matches
#         adjusted_arrays.append(arr)
#         continue
    
#     # Create an interpolator for each column in the array
#     x_current = np.linspace(0, 1, current_rows)
#     x_target = np.linspace(0, 1, target_rows)
#     interpolated_array = np.zeros((target_rows, cols))
    
#     for col in range(cols):
#         interpolator = interp1d(x_current, arr[:, col], kind='linear', fill_value="extrapolate")
#         interpolated_array[:, col] = interpolator(x_target)
    
#     adjusted_arrays.append(interpolated_array)



# result = np.hstack(adjusted_arrays)
# print(result.shape)

# log_norm = np.log10(result)
# print(log_norm.shape)



# O = EISCATOutlierDetection(log_norm)
# X_red = O.pca(log_norm, 3).T

# num_samp = 1000
# X_red = X_red1[:, :]

# # List to store BIC values for different n_components
# bic = []
# n_components_range = range(1, 21)  # Adjust the range as needed

# for n_components in n_components_range:
#     gmm = GaussianMixture(n_components=n_components, random_state=42)
#     gmm.fit(X_red)
#     bic.append(gmm.bic(X_red))

# # Plot BIC values against number of components
# plt.figure(figsize=(8, 4))
# plt.plot(n_components_range, bic, marker='o')
# plt.title('BIC for Gaussian Mixture Model')
# plt.xlabel('Number of Components')
# plt.ylabel('BIC')
# plt.xticks(n_components_range)
# plt.grid()
# plt.show()

# # Determine the optimal number of components
# optimal_n_components = n_components_range[np.argmin(bic)]
# print(f'Optimal number of components: {optimal_n_components}')




# gmm = GaussianMixture(n_components=6, random_state=42)

# # Fit the model to your data
# gmm.fit(X_red)

# # Predict the cluster labels
# labels = gmm.predict(X_red)

# # Print or inspect the labels to see the assigned cluster for each point
# print("Cluster labels:", labels)



# from sklearn.cluster import DBSCAN

# clustering = DBSCAN(eps=1.1, min_samples=100).fit(X_red)
# labels = clustering.labels_


# # num_samp = 10000

# fig, ax = plt.subplots(1, 2, figsize=(14, 8))
# ax[0].scatter(X_red[:, 0], X_red[:, 1], s=30, edgecolors="black", linewidths=0.3)
# ax[0].set_xlabel("X1")
# ax[0].set_ylabel("X2")

# ax[1].scatter(X_red[:, 0], X_red[:, 1], c=labels, s=30, edgecolors="black", linewidths=0.3)
# ax[1].set_xlabel("X1")
# ax[1].set_ylabel("X2")
# plt.show()




# fig, ax = plt.subplots(1, 2, figsize=(14, 8))
# ax[0].scatter(X_red[:, 0], X_red[:, 2], s=30, edgecolors="black", linewidths=0.3)
# ax[0].set_xlabel("X1")
# ax[0].set_ylabel("X3")

# ax[1].scatter(X_red[:, 0], X_red[:, 2], c=labels, s=30, edgecolors="black", linewidths=0.3)
# ax[1].set_xlabel("X1")
# ax[1].set_ylabel("X3")
# plt.show()



# fig, ax = plt.subplots(1, 2, figsize=(14, 8))
# ax[0].scatter(X_red[:, 1], X_red[:, 2], s=30, edgecolors="black", linewidths=0.3)
# ax[0].set_xlabel("X2")
# ax[0].set_ylabel("X3")

# ax[1].scatter(X_red[:, 1], X_red[:, 2], c=labels, s=30, edgecolors="black", linewidths=0.3)
# ax[1].set_xlabel("X2")
# ax[1].set_ylabel("X3")
# plt.show()




# # Creating figure
# fig = plt.figure(figsize=(10, 7))
# ax = plt.axes(projection="3d")
# ax.scatter3D(X_red[:, 0], X_red[:, 1], X_red[:, 2], c=labels, s=30, edgecolors="black", linewidths=0.3)
# ax.set_xlabel("X1")
# ax.set_ylabel("X2")
# ax.set_zlabel("X3")
# plt.show()




# lab, count = np.unique(labels)
# outliers = np.where(labels==-1)[0]
# print(outliers)

# ne = result.T


# for idx in outliers:
#     plt.plot(ne[idx], np.linspace(90, 400, 28))
#     plt.xscale("log")
#     plt.show()



