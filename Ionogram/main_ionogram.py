# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 12:37:28 2024

@author: Kian Sartipzadeh
"""



import os
import numpy as np
import matplotlib.pyplot as plt
from ionogram_sorting import IonogramSorting
from iono_utils import save_dict, load_dict, inspect_dict, merge_days, from_array_to_datetime

from matplotlib.gridspec import GridSpec
from process_ionograms import IonogramProcessing
from tqdm import tqdm




# X = merge_days("sorted_ionogram_dicts", save=True, save_filename="X_ionogram")

# X = load_dict("sorted_ionogram_dicts/2012-1-13.pkl")

# X = X['2012-1-13']['r_param'][0]
# B = IonogramProcessing()
# X_org = B.reconstruct_ionogram(X)
# B.plot_single_ionogram(X_org, flip_vertical=True)
# X_res = B.resample_ionogram(X_org)
# B.plot_single_ionogram(X_res)


# datapath_folder = "Temp"


# for file in tqdm(os.listdir(datapath_folder)):
    
    # print(f"Processing data from {file[6:12]}\n")
    
    # file_path = os.path.join(datapath_folder, file)
    
    # A = IonogramSorting()
    # print("-Sorting Mothly file into 15 min ionogram samples...")
    # A.import_data(file_path)
    # A.save_as_dict(folder_path="sorted_ionogram_dicts")
    # A.save_dataset()
    # data = A.return_dataset()
    # times, data = A.import_data(file_path)
    
    
    
    # print("-Sorting Complete!\n")
    # break
    

    
    # B = IonogramProcessing()
    # B.process_ionogram(data, times, plot=True, apply_amplitude_filter=True)
#     break
    
#     # print("-Making Ionogram images...")
#     # # B.process_ionogram(data, times, plot=False, result_path="Ionogram_Images")
#     # B.process_ionogram(data, times, plot=True)
#     # print("-Making Ionograms Complete!\n")
#     # print("==========================================================")
#     # break







data = load_dict("sorted_ionogram_dicts/2012-1-20.pkl")


# data = Data['2012-1-20']


import seaborn as sns
sns.set(style="dark", context=None, palette=None)

for i in range(len(data['r_time'])):
    X = data['r_param'][i]
    y = from_array_to_datetime(data['r_time'])[i]
    
    freq = X[:, 0]  # Frequency values
    rang = X[:, 1]  # Range values
    pol = np.round(X[:, 2])  # Polarization values
    amp = X[:, 4]  # Amplitude values
    ang = np.round(X[:, 7])  # Angle values
    
    # Initialize amplitude mask
    mask_amp = np.zeros_like(freq, dtype=bool)
    unique_freqs = np.unique(freq)
    
    for f in unique_freqs:
        indices = np.where(freq == f)[0]
        if len(indices) == 0:
            continue
        current_amps = amp[indices]
        max_amp_f = np.max(current_amps)
        threshold = 0.75 * max_amp_f
        mask_amp[indices] = current_amps >= threshold
        
    # Apply masks for polarization, angle, and amplitude
    mask_O = (pol == 90) & (ang == 0) & mask_amp
    mask_X = (pol == -90) & (ang == 0) & mask_amp
    
    # Apply masks for polarization, angle, and amplitude
    org_mask_O = (pol == 90) & (ang == 0)
    org_mask_X = (pol == -90) & (ang == 0)
    
    
    # # Plotting
    # fig, ax = plt.subplots()
    # fig.suptitle(y)
    
    # ax.scatter(freq[mask_O], rang[mask_O], s=1, c=amp[mask_O], cmap="turbo", zorder=2)
    # ax.scatter(freq[mask_X], rang[mask_X], s=1, c=amp[mask_X], cmap="turbo", zorder=2)
    # ax.scatter(freq[org_mask_O], rang[org_mask_O], s=1, c="black", zorder=0)
    # ax.scatter(freq[org_mask_X], rang[org_mask_X], s=1, c="black", zorder=0)
    # ax.set_xlabel("Frequency")
    # ax.set_ylabel("Range")
    # ax.set_xlim(0, 9.1)
    # ax.set_ylim(78, 645)
    
    # plt.show()

    
    # fig, ax = plt.subplots(1, 3, figsize=(16, 5))
    
    # fig.suptitle(y)
    
    # ax[0].scatter(freq[org_mask_O], rang[org_mask_O], s=1, c=amp[org_mask_O], cmap="turbo", zorder=2)
    # ax[0].scatter(freq[org_mask_X], rang[org_mask_X], s=1, c=amp[org_mask_X], cmap="turbo", zorder=2)
    # # ax[0].scatter(freq[org_mask_O], rang[org_mask_O], s=1, c="black", zorder=0)
    # # ax[0].scatter(freq[org_mask_X], rang[org_mask_X], s=1, c="black", zorder=0)
    # ax[0].set_xlabel("Frequency")
    # ax[0].set_ylabel("Range")
    # # ax[0].set_xlim(0, 9.6)
    # ax[0].set_ylim(78, 645)
    
    
    # ax[1].scatter(freq[mask_O], rang[mask_O], s=1, c=amp[mask_O], cmap="turbo", zorder=2)
    # ax[1].scatter(freq[mask_X], rang[mask_X], s=1, c=amp[mask_X], cmap="turbo", zorder=2)
    # ax[1].scatter(freq[org_mask_O], rang[org_mask_O], s=1, c="black", zorder=0)
    # ax[1].scatter(freq[org_mask_X], rang[org_mask_X], s=1, c="black", zorder=0)
    # ax[1].set_xlabel("Frequency")
    # ax[1].set_ylabel("Range")
    # # ax[1].set_xlim(0, 9.6)
    # ax[1].set_ylim(78, 645)
    
    
    # ax[2].scatter(freq[mask_O], rang[mask_O], s=1, c=amp[mask_O], cmap="turbo", zorder=2)
    # ax[2].scatter(freq[mask_X], rang[mask_X], s=1, c=amp[mask_X], cmap="turbo", zorder=2)
    # # ax[2].scatter(freq[org_mask_O], rang[org_mask_O], s=1, c="black", zorder=0)
    # # ax[2].scatter(freq[org_mask_X], rang[org_mask_X], s=1, c="black", zorder=0)
    # ax[2].set_xlabel("Frequency")
    # ax[2].set_ylabel("Range")
    # # ax[2].set_xlim(0, 9.6)
    # ax[2].set_ylim(78, 645)
    
    # plt.show()
    
    
    date_str = y.strftime('%Y-%m-%d %H:%M')

    # Create a grid layout
    fig = plt.figure(figsize=(11, 4))
    gs = GridSpec(1, 4, width_ratios=[1, 1, 1, 0.05], wspace=0.1)
    
    # Shared y-axis setup
    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1], sharey=ax0)
    ax2 = fig.add_subplot(gs[2], sharey=ax0)
    cax = fig.add_subplot(gs[3])
    
    fig.suptitle(f'{date_str}', fontsize=16, y=1.03)
    
    a=ax0.scatter(freq[org_mask_O], rang[org_mask_O], s=1, c=amp[org_mask_O], cmap="turbo", zorder=2)
    ax0.scatter(freq[org_mask_X], rang[org_mask_X], s=1, c=amp[org_mask_X], cmap="turbo", zorder=2)
    ax0.set_title("Original")
    ax0.set_xlabel("Frequency (MHz)")
    ax0.set_ylabel("Virtual Altitude (km)")
    ax0.set_xlim(0.9, 9.1)
    ax0.set_ylim(78, 645)
    ax0.grid(True)
    
    ax1.scatter(freq[mask_O], rang[mask_O], s=1, c=amp[mask_O], cmap="turbo", zorder=2)
    ax1.scatter(freq[mask_X], rang[mask_X], s=1, c=amp[mask_X], cmap="turbo", zorder=2)
    ax1.scatter(freq[org_mask_O], rang[org_mask_O], s=1, c="black", zorder=1)
    ax1.scatter(freq[org_mask_X], rang[org_mask_X], s=1, c="black", zorder=1)
    ax1.set_title("Outlined Noise")
    ax1.set_xlabel("Frequency (MHz)")
    ax1.set_xlim(0.9, 9.1)
    ax1.set_ylim(78, 645)
    ax1.tick_params(labelleft=False)
    ax1.grid(True)
    
    ax2.scatter(freq[mask_O], rang[mask_O], s=1, c=amp[mask_O], cmap="turbo", zorder=2)
    ax2.scatter(freq[mask_X], rang[mask_X], s=1, c=amp[mask_X], cmap="turbo", zorder=2)
    ax2.set_title("Filtered")
    ax2.set_xlabel("Frequency (MHz)")
    ax2.set_xlim(0.9, 9.1)
    ax2.set_ylim(78, 645)
    ax2.tick_params(labelleft=False)
    ax2.grid(True)
    
    # Add colorbar
    cbar = fig.colorbar(a, cax=cax, orientation='vertical')
    cbar.set_label('Amplitude', fontsize=13)
    
    plt.show()
    

# for i in range(0, len(data['r_time'])):

#     X = data['r_param'][i]
#     y = data['r_time'][i]
    
    
    
#     freq = X[:, 0]  # Frequency values
#     rang = X[:, 1]  # Range values
#     pol = np.round(X[:, 2])                # Polarization values
#     amp = X[:, 4]                          # Amplitude values
#     ang = np.round(X[:, 7])                # Angle values

    
    
    
#     mask_O = (pol == 90) & (ang == 0)
#     mask_X = (pol == -90) & (ang == 0)
#     Freq = freq[mask_O]
#     Rang = rang[mask_O]
    
#     fig, ax = plt.subplots()
#     fig.suptitle(y)
    
#     ax.scatter(Freq, Rang, s=1, c=amp[mask_O], cmap="turbo")
#     ax.scatter(freq[mask_X], rang[mask_X], s=1, c=amp[mask_X], cmap="turbo")
#     ax.set_xlabel("freq")
#     ax.set_ylabel("rang")
    
#     plt.show()











