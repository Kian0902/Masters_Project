# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 12:37:28 2024

@author: Kian Sartipzadeh
"""



import os
import numpy as np
import matplotlib.pyplot as plt
from ionogram_sorting import IonogramSorting
from iono_utils import save_dict, load_dict, inspect_dict, merge_days


from process_ionograms import IonogramProcessing
from tqdm import tqdm




# X = merge_days("sorted_ionogram_dicts")
# X = X['2012-1-13']['r_param'][0]
# B = IonogramProcessing()
# X_org = B.reconstruct_ionogram(X)
# B.plot_single_ionogram(X_org, flip_vertical=True)
# X_res = B.resample_ionogram(X_org)
# B.plot_single_ionogram(X_res)


# datapath_folder = "Iono_TXT"


# for file in tqdm(os.listdir(datapath_folder)):
    
#     # print(f"Processing data from {file[6:12]}\n")
    
#     file_path = os.path.join(datapath_folder, file)
    
#     A = IonogramSorting()
#     # print("-Sorting Mothly file into 15 min ionogram samples...")
#     # A.import_data(file_path)
#     # A.save_as_dict(folder_path="sorted_ionogram_dicts")
#     # A.save_dataset()
#     # data = A.return_dataset()
#     times, data = A.import_data(file_path)
#     # print("-Sorting Complete!\n")
#     # break
    
    
#     # print(data[0])
#     # print(data[0].shape)
#     # print(data.shape)
    
#     B = IonogramProcessing()
#     B.process_ionogram(data, times, plot=True)
#     break
    
#     # print("-Making Ionogram images...")
#     # # B.process_ionogram(data, times, plot=False, result_path="Ionogram_Images")
#     # B.process_ionogram(data, times, plot=True)
#     # print("-Making Ionograms Complete!\n")
#     # print("==========================================================")
#     # break





Data = merge_days("sorted_ionogram_dicts")


data = Data['2012-1-14']




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




for i in range(len(data['r_time'])):
    X = data['r_param'][i]
    y = data['r_time'][i]
    
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
    
    
    # Plotting
    fig, ax = plt.subplots()
    fig.suptitle(y)
    
    ax.scatter(freq[mask_O], rang[mask_O], s=1, c=amp[mask_O], cmap="turbo", zorder=2)
    ax.scatter(freq[mask_X], rang[mask_X], s=1, c=amp[mask_X], cmap="turbo", zorder=2)
    ax.scatter(freq[org_mask_O], rang[org_mask_O], s=1, c="black", zorder=0)
    ax.scatter(freq[org_mask_X], rang[org_mask_X], s=1, c="black", zorder=0)
    ax.set_xlabel("Frequency")
    ax.set_ylabel("Range")
    ax.set_xlim(0, 9.1)
    ax.set_ylim(78, 645)
    
    plt.show()












