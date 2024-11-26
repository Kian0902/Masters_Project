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
from data_outlier_detection import EISCATOutlierDetection
from matplotlib.dates import DateFormatter

from scipy.interpolate import interp1d
import numpy as np




def plot(data):
    """
    Plot a comparison of original and averaged data using pcolormesh.

    Input (type)                 | DESCRIPTION
    ------------------------------------------------
    original_data (dict)         | Dictionary containing the original data.
    """
    
    # Convert time arrays to datetime objects
    r_time = np.array([datetime(year, month, day, hour, minute, second) 
                            for year, month, day, hour, minute, second in data['r_time']])
    r_h = data['r_h']
    r_param = data['r_param']
    r_error = data['r_error']
    
    # Date
    date_str = r_time[0].strftime('%Y-%m-%d')
    
    # Creating the plots
    fig, ax = plt.subplots(1, 2, figsize=(10, 8), sharey=True)
    fig.suptitle(f'Date: {date_str}', fontsize=15)
    fig.tight_layout()
    
    
    # Plotting original data
    pcm_ne = ax[0].pcolormesh(r_time, r_h.flatten(), np.log10(r_param), shading='auto', cmap='turbo', vmin=9, vmax=12)
    ax[0].set_title('EISCAT UHF', fontsize=17)
    ax[0].set_xlabel('Time [hours]')
    ax[0].set_ylabel('Altitude [km]')
    ax[0].xaxis.set_major_formatter(DateFormatter('%H:%M'))
    fig.autofmt_xdate()
    
    # Add colorbar for the original data
    # cbar = fig.colorbar(pcm_ne, ax=ax[0], orientation='vertical', fraction=0.03, pad=0.04, aspect=20, shrink=1)
    # cbar.set_label('log10(n_e) [g/cm^3]')
    
    # fig.autofmt_xdate()
    
    # Plotting original data
    pcm_err = ax[1].pcolormesh(r_time, r_h.flatten(), np.log10(r_error), shading='auto', cmap='turbo', vmin=9, vmax=12)
    ax[1].set_title('Measurement Error', fontsize=17)
    ax[1].set_xlabel('Time [hours]')
    # ax[1].set_ylabel('Altitude [km]')
    ax[1].xaxis.set_major_formatter(DateFormatter('%H:%M'))
    # fig.autofmt_xdate()
    
    # Add colorbar for the original data
    cbar = fig.colorbar(pcm_err, ax=ax[1], orientation='vertical', fraction=0.03, pad=0.04, aspect=44, shrink=3)
    cbar.set_label(r'$log_{10}(n_e)$ [g/cm$^3$]', fontsize=17)
    
    
    # Display the plots
    plt.show()





def save_data(dataset: dict, file_name: str):
    with open(file_name, 'wb') as file:
        pickle.dump(dataset, file)



def detect_nan_in_arrays(data_dict):
    for key, array in data_dict.items():
        if np.isnan(array).any():
            print(f"NaN detected {key}: {array.shape}")


# # Use the local folder name containing data
# folder_name_in  = "EISCAT_Madrigal/2020"
folder_name_out = "EISCAT_MAT/2020"


# # Extract info from hdf5 files
# madrigal_processor = EISCATDataProcessor(folder_name_in, folder_name_out)
# madrigal_processor.process_all_files()



# Sorting data
Eiscat = EISCATDataSorter(folder_name_out)
Eiscat.sort_data()               # sort data
X_Eiscat = Eiscat.return_data()  # returning dict data


# x = X_Eiscat["2020-1-14"]

# plot(x)



# Clipping range and Filtering data for nan
filt = EISCATDataFilter(X_Eiscat, filt_range=True, filt_nan=True) 
filt.batch_filtering()
X_filt = filt.return_data()


# Detecting outliers
Outlier = EISCATOutlierDetection(X_filt)
Outlier.batch_detection(method_name="IQR", save_plot=False)
X_outliers = Outlier.return_outliers()


# Filtering outliers
outlier_filter = EISCATDataFilter(X_filt, filt_outlier=True)
outlier_filter.batch_filtering(dataset_outliers=X_outliers, filter_size=3, plot_after_each_day=False)
X_outliers_filtered = outlier_filter.return_data()




# Assuming X_avg contains your nested dictionary after filtering and averaging
# Extract the reference altitude array with length 27
reference_altitudes = None
for date, data in X_outliers_filtered.items():
    if len(data["r_h"]) == 27:
        reference_altitudes = data["r_h"].flatten()
        break

if reference_altitudes is None:
    raise ValueError("No day with 27 altitude measurements found")

# Create a new dictionary to store interpolated values
X_interp = {}

# Iterate over each day in X_avg
for date, data in X_outliers_filtered.items():
    r_h = data["r_h"].flatten()  # Original altitudes (shape: (N,))
    r_param = data["r_param"]     # Electron density measurements (shape: (N, M))
    r_error = data["r_error"]     # Error values (shape: (N, M))
    
    
    # Interpolating electron density (r_param) to reference_altitudes
    interpolated_r_param = np.zeros((len(reference_altitudes), r_param.shape[1]))
    interpolated_r_error = np.zeros((len(reference_altitudes), r_error.shape[1]))
    
    for i in range(r_param.shape[1]):
        # Interpolating each column of r_param and r_error
        interpolated_r_param[:, i] = np.interp(reference_altitudes, r_h, r_param[:, i])
        interpolated_r_error[:, i] = np.interp(reference_altitudes, r_h, r_error[:, i])

    # Storing the interpolated values in the new dictionary
    X_interp[date] = {
        "r_time": data["r_time"],  # Keep the original time data
        "r_h": reference_altitudes.reshape(-1, 1),  # Use the reference altitudes (shape: (27, 1))
        "r_param": interpolated_r_param,            # Interpolated electron densities (shape: (27, M))
        "r_error": interpolated_r_error             # Interpolated errors (shape: (27, M))
    }

# X_interp now contains the data with all days having 27 altitude levels

# Averaging data
AVG = EISCATAverager(X_interp)
AVG.batch_averaging(save_plot=False, weighted=False)
X_avg = AVG.return_data()


for day in X_avg:
    detect_nan_in_arrays(X_avg[day])




save_data(X_avg, file_name="X_averaged_2020")





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



