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


# # Use the local folder name containing data
# folder_name_in  = "EISCAT_Madrigal/2019"
folder_name_out = "EISCAT_MAT/2019"


# # Extract info from hdf5 files
# madrigal_processor = EISCATDataProcessor(folder_name_in, folder_name_out)
# madrigal_processor.process_all_files()



# Sorting data
Eiscat = EISCATDataSorter(folder_name_out)
Eiscat.sort_data()               # sort data
X_Eiscat = Eiscat.return_data()  # returning dict data




# Clipping range and Filtering data for nan
filt = EISCATDataFilter(X_Eiscat, filt_range=True, filt_nan=True) 
filt.batch_filtering()
X_filt = filt.return_data()


# Averaging data
AVG = EISCATAverager(X_filt)
AVG.batch_averaging(save_plot=False, weighted=False)
X_avg = AVG.return_data()


for key in X_avg:
    print(key, X_avg[key]["r_h"].shape)



# save_data(X_avg , file_name="X_avg")






# for key in X_filt:
#     detect_nan_in_arrays(X_filt[key])








# # Detecting outliers
# Outlier = EISCATOutlierDetection(X_filtered)
# Outlier.batch_detection(method_name="IQR", save_plot=False)
# X_outliers = Outlier.return_outliers()


# # Filtering outliers
# outlier_filter = EISCATDataFilter(X_filtered, filt_outlier=True)
# outlier_filter.batch_filtering(dataset_outliers=X_outliers, filter_size=3, plot_after_each_day=False)
# X_outliers_filtered = outlier_filter.return_data()










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



