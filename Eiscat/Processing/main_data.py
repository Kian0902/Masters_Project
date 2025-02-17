# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 14:13:45 2024

@author: Kian Sartipzadeh
"""

import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch
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
from data_utils import from_array_to_datetime, inspect_dict, get_day, get_day_data, MatchingFiles, from_strings_to_datetime, save_dict, load_dict, plot_compare_all
from scipy.interpolate import interp1d
# import seaborn as sns




    
    
    
# # Use the local folder name containing data
# folder_name_in  = "EISCAT_Madrigal/2018"
# folder_name_out = "EISCAT_MAT/2018"

# # Extract info from hdf5 files
# madrigal_processor = EISCATDataProcessor(folder_name_in, folder_name_out)
# madrigal_processor.process_all_files()



# VHF_folder = "EISCAT_MAT/VHF_All"
# both_folder = "EISCAT_MAT/EISCAT_test_data"

# both_folder = "EISCAT_MAT/New"
folder_name_out = "EISCAT_MAT/2022"

# Match = MatchingFiles(VHF_folder, UHF_folder)
# Match.remove_matching_vhf_files()


# __________ Sorting data __________ 
Eiscat = EISCATDataSorter(folder_name_out)
Eiscat.sort_data()
X_eis = Eiscat.return_data()

# save_dict(X_eis, file_name="X_eis")



# __________ Clipping range and Filtering data for nan __________ 
filt = EISCATDataFilter(X_eis, filt_range=True, filt_nan=True, filt_interpolate=True) 
filt.batch_filtering(num_of_ref_alt=27)
X_filt = filt.return_data()



# __________ Detecting outliers __________ 
Outlier = EISCATOutlierDetection(X_filt)
Outlier.batch_detection(method_name="IQR", save_plot=False)
# Outlier.pca_all(reduce_to_dim=2)
X_outliers = Outlier.return_outliers()



# __________ Filtering outliers __________ 
outlier_filter = EISCATDataFilter(X_filt, filt_outlier=True)
outlier_filter.batch_filtering(dataset_outliers=X_outliers, filter_size=5, plot_after_each_day=False)
X_outliers_filtered = outlier_filter.return_data()


# save_dict(X_outliers_filtered, file_name="X_outliers_filtered")



# __________  Averaging data __________ 
AVG = EISCATAverager(X_outliers_filtered, plot_result=False)
AVG.average_15min()
X_avg = AVG.return_data()



# save_dict(X_avg, file_name="X_avg")

X_1 = load_dict("X_eis")['2022-1-30']
X_2 = load_dict("X_outliers_filtered")['2022-1-30']
X_3 = load_dict("X_avg")['2022-1-30']


# plot_day(X_1, '2022-1-30')
# plot_day(X_2, '2022-1-30')


# plot_compare(X_1, X_2, '2022-1-30')

plot_compare_all(X_1, X_2, X_3, '2022-1-30')





# plot_day(X_avg['2022-1-30'], '2022-1-30')
# save_dict(X_filt, file_name="X_avg_test_data")

# for day in X_avg:
#     print(X_avg[day]['num_avg_samp'])


# plot_day(X_avg['2012-1-19'])

# plot_available_data(X_avg)
# save_dict(X_filt, file_name="X_avg_new")



# plot_day(X_avg['2022-6-20'])





# from sklearn.decomposition import PCA
# from matplotlib.widgets import Cursor
# from matplotlib.gridspec import GridSpec

# def ravel_dict(data_dict):
#     return np.concatenate([data_dict[date]['r_param'] for date in data_dict], axis=1)

# def preprocess_data(X):
#     X = X.T
#     X = np.where(X > 0, X, np.nan)  # Replace negatives/zeros with NaN
#     X = np.log10(X)
#     X = np.nan_to_num(X, nan=0.0)  # Replace NaNs with 0
#     X[X < 6] == 8
#     # X = (X - np.min(X)) / (np.max(X) - np.min(X))  # Normalize to [0,1]
#     return X


# def apply_pca(data, reduce_dim=2):
    
#     pca_model = PCA(n_components=reduce_dim)
#     reduced_data = pca_model.fit_transform(data)
#     return reduced_data



# r_h = np.array([[ 91.5687711 ],[ 94.57444598],[ 97.57964223],[100.57010953],
#        [103.57141624],[106.57728701],[110.08393175],[114.60422289],
#        [120.1185208 ],[126.61221111],[134.1346149 ],[142.53945817],
#        [152.05174717],[162.57986185],[174.09833378],[186.65837945],
#        [200.15192581],[214.62769852],[230.12198695],[246.64398082],
#        [264.11728204],[282.62750673],[302.15668686],[322.70723831],
#        [344.19596481],[366.64409299],[390.113117  ]])

# X =  ravel_dict(load_dict("X_avg_all"))
# X_filt = preprocess_data(X)
# x = apply_pca(X_filt, reduce_dim=3)
# X = X.T

# # Create figure and grid layout
# fig = plt.figure(figsize=(10, 8))
# gs = GridSpec(3, 2, width_ratios=[1, 1], wspace=0.3, hspace=0.4)
# ax0 = fig.add_subplot(gs[0, 0])
# ax1 = fig.add_subplot(gs[1, 0])
# ax2 = fig.add_subplot(gs[2, 0])
# ax3 = fig.add_subplot(gs[:, 1])

# # Scatter plots for PC1 vs PC2, PC1 vs PC3, and PC2 vs PC3
# scatter0 = ax0.scatter(x[:, 0], x[:, 1], c='blue', alpha=0.6)
# ax0.set_xlim(-6, 6)
# ax0.set_ylim(-3, 5)
# ax0.set_xlabel('PC1')
# ax0.set_ylabel('PC2')

# scatter1 = ax1.scatter(x[:, 0], x[:, 2], c='blue', alpha=0.6)
# ax1.set_xlim(-6, 6)
# ax1.set_ylim(-4, 2.6)
# ax1.set_xlabel('PC1')
# ax1.set_ylabel('PC3')

# scatter2 = ax2.scatter(x[:, 1], x[:, 2], c='blue', alpha=0.6)
# ax2.set_xlim(-6, 6)
# ax2.set_ylim(-3, 5)
# ax2.set_xlabel('PC2')
# ax2.set_ylabel('PC3')

# # Initialize variables to store highlighted points
# highlighted_idx = None

# # Function to find the closest point to the click event
# def find_closest_point(event, scatter_data, ax):
#     if event.inaxes == ax:
#         dist = np.sqrt((scatter_data[:, 0] - event.xdata)**2 + (scatter_data[:, 1] - event.ydata)**2)
#         return np.argmin(dist)
#     return None

# # Mouse click event handler
# def on_click(event):
#     global highlighted_idx

#     # # Reset previous highlight
#     # if highlighted_idx is not None:
#     #     scatter0._facecolors[highlighted_idx] = 'blue'
#     #     scatter1._facecolors[highlighted_idx] = 'blue'
#     #     scatter2._facecolors[highlighted_idx] = 'blue'

#     # Find the closest point in each scatter plot
#     idx0 = find_closest_point(event, np.c_[x[:, 0], x[:, 1]], ax0)
#     idx1 = find_closest_point(event, np.c_[x[:, 0], x[:, 2]], ax1)
#     idx2 = find_closest_point(event, np.c_[x[:, 1], x[:, 2]], ax2)

#     # Determine the final index based on which subplot was clicked
#     if event.inaxes == ax0:
#         highlighted_idx = idx0
#     elif event.inaxes == ax1:
#         highlighted_idx = idx1
#     elif event.inaxes == ax2:
#         highlighted_idx = idx2

#     if highlighted_idx is not None:
#         # Highlight the selected point in all scatter plots
#         # scatter0._facecolors[highlighted_idx] = 'red'
#         # scatter1._facecolors[highlighted_idx] = 'red'
#         # scatter2._facecolors[highlighted_idx] = 'red'

#         # Update the original data plot on ax3
#         ax3.clear()
#         ax3.plot(X[highlighted_idx], r_h.flatten(), label=f'Sample {highlighted_idx}', color='green')
#         ax3.set_xscale("log")
#         ax3.set_title(f'Electron Density Profile (Sample {highlighted_idx})')
#         ax3.set_xlabel('Index')
#         ax3.set_ylabel('r_h')
#         ax3.legend()

#     # Redraw the figure
#     fig.canvas.draw_idle()

# # Add Cursor widgets to the scatter plots
# cursor0 = Cursor(ax0, useblit=True, color='red', linewidth=1)
# cursor1 = Cursor(ax1, useblit=True, color='red', linewidth=1)
# cursor2 = Cursor(ax2, useblit=True, color='red', linewidth=1)

# # Connect the click event to the callback function
# fig.canvas.mpl_connect('button_press_event', on_click)

# # Initial plot on ax3 (optional, you can choose a default sample)
# # idx = 3
# # ax3.plot(X[idx], r_h.flatten(), label=f'Sample {idx}', color='green')
# # ax3.set_title(f'Electron Density Profile (Sample {idx})')
# # ax3.set_xlabel('Index')
# # ax3.set_ylabel('r_h')
# # ax3.legend()

# # Show the plot
# plt.show()













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



