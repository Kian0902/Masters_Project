# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 16:48:26 2025

@author: Kian Sartipzadeh
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


from torchvision import transforms
from storing_dataset import MatchingIonoPair, StoreIonoDataset


from Iono_CNN_utils import from_strings_to_array, filter_artist_times, load_dict, save_dict, convert_pred_to_dict, from_array_to_datetime, from_strings_to_datetime, from_csv_to_numpy, add_key_from_dict_to_dict, add_key_with_matching_times, inspect_dict, convert_geophys_to_dict
from iono_CNN_model import IonoCNN


from matplotlib.dates import DateFormatter
import matplotlib.colors as colors
from matplotlib.gridspec import GridSpec

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)




def merge_nested_dict(nested_dict):
    all_r_time = []
    all_r_param = []
    
    # Loop over keys in sorted order (optional, if order matters)
    for key in sorted(nested_dict):
        all_r_time.append(nested_dict[key]['r_time'])
        all_r_param.append(nested_dict[key]['r_param'])
    
    # Concatenate arrays along axis=0
    merged_r_time = np.concatenate(all_r_time, axis=0)
    merged_r_param = np.concatenate(all_r_param, axis=1)
    
    # Return the merged dictionary under the key "All"
    return {"All": {"r_time": merged_r_time, "r_param": merged_r_param}}



def plot_pred(data):
    r_time = from_array_to_datetime(data["r_time"])
    # r_h = data1["r_h"].flatten()
    r_h = np.array([[91.5687711 ],[ 94.57444598],[ 97.57964223],[100.57010953],
           [103.57141624],[106.57728701],[110.08393175],[114.60422289],
           [120.1185208 ],[126.61221111],[134.1346149 ],[142.53945817],
           [152.05174717],[162.57986185],[174.09833378],[186.65837945],
           [200.15192581],[214.62769852],[230.12198695],[246.64398082],
           [264.11728204],[282.62750673],[302.15668686],[322.70723831],
           [344.19596481],[366.64409299],[390.113117  ]]).flatten()
    
    ne_pred = 10**data["r_param"]

    
    # date_str = r_time[0].strftime('%Y-%m-%d')
    
    # Create a grid layout
    fig = plt.figure(figsize=(24, 6))
    gs = GridSpec(1, 2, width_ratios=[1, 0.05], wspace=0.1)
    
    # Shared y-axis setup
    ax0 = fig.add_subplot(gs[0])
    # ax1 = fig.add_subplot(gs[1], sharey=ax0)
    cax = fig.add_subplot(gs[1])
    
    # fig.suptitle(f'Date: {date_str}', fontsize=20, y=1.03)

    
    
    # Plotting EISCAT
    ne = ax0.pcolormesh(r_time, r_h, ne_pred, shading='auto', cmap='turbo', norm=colors.LogNorm(vmin=1e10, vmax=1e12))
    ax0.set_title('KIAN-Net', fontsize=17)
    ax0.set_xlabel('Time [hh:mm]', fontsize=13)
    ax0.set_ylabel('Altitude [km]', fontsize=15)
    ax0.xaxis.set_major_formatter(DateFormatter('%H:%M'))
    
    # # Plotting DL model
    # ax1.pcolormesh(r_time, r_h, ne_pred, shading='auto', cmap='turbo', norm=colors.LogNorm(vmin=1e10, vmax=1e12))
    # ax1.set_title('Iono-CNN', fontsize=17)
    # ax1.set_xlabel('Time [hh:mm]', fontsize=13)
    # ax1.xaxis.set_major_formatter(DateFormatter('%H:%M'))
    # ax1.tick_params(labelleft=False)
    
    
    # Rotate x-axis labels
    for ax in [ax0]:
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='center')
    
    
    
    # Add colorbar
    cbar = fig.colorbar(ne, cax=cax, orientation='vertical')
    cbar.set_label(r'$log_{10}(n_e)$ $[n\,m^{-3}]$', fontsize=17)
    
    plt.show()



def plot_compare(data1, data2):
    r_time = from_array_to_datetime(data1["r_time"])
    r_h = data1["r_h"].flatten()
    
    ne_eis = data1["r_param"]
    ne_pred = 10**data2["r_param"]

    
    date_str = r_time[0].strftime('%Y-%m-%d')
    
    # Create a grid layout
    fig = plt.figure(figsize=(16, 6))
    gs = GridSpec(1, 3, width_ratios=[1, 1, 0.05], wspace=0.1)
    
    # Shared y-axis setup
    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1], sharey=ax0)
    cax = fig.add_subplot(gs[2])
    
    fig.suptitle(f'Date: {date_str}', fontsize=20, y=1.03)

    
    
    # Plotting EISCAT
    ne_EISCAT = ax0.pcolormesh(r_time, r_h, ne_eis, shading='gouraud', cmap='turbo', norm=colors.LogNorm(vmin=1e10, vmax=1e12))
    ax0.set_title('EISCAT UHF', fontsize=17)
    ax0.set_xlabel('Time [hh:mm]', fontsize=13)
    ax0.set_ylabel('Altitude [km]', fontsize=15)
    ax0.xaxis.set_major_formatter(DateFormatter('%H:%M'))
    
    # Plotting DL model
    ax1.pcolormesh(r_time, r_h, ne_pred, shading='gouraud', cmap='turbo', norm=colors.LogNorm(vmin=1e10, vmax=1e12))
    ax1.set_title('Iono-CNN', fontsize=17)
    ax1.set_xlabel('Time [hh:mm]', fontsize=13)
    ax1.xaxis.set_major_formatter(DateFormatter('%H:%M'))
    ax1.tick_params(labelleft=False)
    
    
    # Rotate x-axis labels
    for ax in [ax0, ax1]:
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='center')
    
    
    
    # Add colorbar
    cbar = fig.colorbar(ne_EISCAT, cax=cax, orientation='vertical')
    cbar.set_label(r'$log_{10}(n_e)$ $[n\,m^{-3}]$', fontsize=17)
    
    plt.show()




def model_predict(stored_dataset, DL_model, model_weights):

    A = stored_dataset
    
    # Creating DataLoader
    batch_size = len(A)
    test_loader = DataLoader(A, batch_size=batch_size, shuffle=False)
    
    
    
    model = DL_model
    # criterion = nn.HuberLoss()
    
    # Loading the trained network weights
    weights_path = model_weights
    model.load_state_dict(torch.load(weights_path, weights_only=True))
    model.to(device)
    
    model.eval()
    
    predictions = []
    
    with torch.no_grad():
        for data in test_loader:
            
            data = data.to(device)
            
            outputs = model(data)
            
            # loss = criterion(outputs, targets)
            # print(loss)
            predictions.extend(outputs.cpu().numpy())
    
    
    model_ne = np.array(predictions)
    return model_ne




if __name__ == "__main__":
    
    # Test data folder names
    test_ionogram_folder = "testing_data1/ionogram_test_days_new"
    # test_radar_folder = "testing_data1/eiscat_test_days_new"
    # test_geophys_folder = "testing_data/test_geophys_folder"
    
    # Initializing class for matching pairs
    Pairs = MatchingIonoPair(test_ionogram_folder)
    
    
    # Returning matching sample pairs
    geo, radar_times = Pairs.find_pairs(return_date=True)
    r_t = from_strings_to_datetime(radar_times)
    r_times = from_strings_to_array(radar_times)
    
    
    
    # Storing the sample pairs
    A = StoreIonoDataset(geo)
    
    # Path to trained weights
    weights_path = 'Iono_CNNv1.pth'
    
    
    X_pred = model_predict(A, IonoCNN(), weights_path)
    X_kian = convert_pred_to_dict(r_t, r_times, X_pred)
    
    
    # X_Kian = merge_nested_dict(X_kian)
    
    
    save_dict(X_kian, "new_X_pred_ionocnn")
    
    # plot_pred(X_Kian['All'])
    
    # X_true = from_csv_to_numpy(test_radar_folder)[0]
    # X_eis = convert_pred_to_dict(r_t, r_times, X_true)
    
    
    # # Adding 'r_h' from eiscat to all dicts
    # Eiscat_support = load_dict("new_X_true_eiscat")
    # X_eis = add_key_from_dict_to_dict(Eiscat_support, X_eis, key="r_h")
    # X_eis = add_key_with_matching_times(Eiscat_support, X_eis, key="r_error")
    
    
    # X_kian = add_key_from_dict_to_dict(Eiscat_support, X_kian, key="r_h")

    
    # for day in X_eis:
    #     plot_compare(X_eis[day], X_kian[day])
    













# def plot_day(data):
#     r_time = from_array_to_datetime(np.array([[2012,    1,   19,    0,    0,    0],
#            [2012,    1,   19,    0,   15,    0],
#            [2012,    1,   19,    0,   30,    0],
#            [2012,    1,   19,    0,   45,    0],
#            [2012,    1,   19,    1,    0,    0],
#            [2012,    1,   19,    1,   15,    0],
#            [2012,    1,   19,    1,   30,    0],
#            [2012,    1,   19,    1,   45,    0],
#            [2012,    1,   19,    2,    0,    0],
#            [2012,    1,   19,    2,   15,    0],
#            [2012,    1,   19,    2,   30,    0],
#            [2012,    1,   19,    2,   45,    0],
#            [2012,    1,   19,    3,    0,    0],
#            [2012,    1,   19,    3,   15,    0],
#            [2012,    1,   19,    3,   30,    0],
#            [2012,    1,   19,    3,   45,    0],
#            [2012,    1,   19,    4,    0,    0],
#            [2012,    1,   19,    4,   15,    0],
#            [2012,    1,   19,    4,   30,    0],
#            [2012,    1,   19,    4,   45,    0],
#            [2012,    1,   19,    5,    0,    0],
#            [2012,    1,   19,    5,   15,    0],
#            [2012,    1,   19,    5,   30,    0],
#            [2012,    1,   19,    5,   45,    0],
#            [2012,    1,   19,    6,    0,    0],
#            [2012,    1,   19,    6,   15,    0],
#            [2012,    1,   19,    6,   30,    0],
#            [2012,    1,   19,    6,   45,    0],
#            [2012,    1,   19,    7,    0,    0],
#            [2012,    1,   19,    7,   15,    0],
#            [2012,    1,   19,    7,   30,    0],
#            [2012,    1,   19,    7,   45,    0],
#            [2012,    1,   19,    8,    0,    0],
#            [2012,    1,   19,    8,   15,    0],
#            [2012,    1,   19,    8,   30,    0],
#            [2012,    1,   19,    8,   45,    0],
#            [2012,    1,   19,    9,    0,    0],
#            [2012,    1,   19,    9,   15,    0],
#            [2012,    1,   19,    9,   30,    0],
#            [2012,    1,   19,    9,   45,    0],
#            [2012,    1,   19,   10,    0,    0],
#            [2012,    1,   19,   10,   15,    0],
#            [2012,    1,   19,   10,   30,    0],
#            [2012,    1,   19,   10,   45,    0],
#            [2012,    1,   19,   11,    0,    0],
#            [2012,    1,   19,   11,   15,    0],
#            [2012,    1,   19,   11,   30,    0],
#            [2012,    1,   19,   11,   45,    0],
#            [2012,    1,   19,   12,    0,    0],
#            [2012,    1,   19,   12,   15,    0],
#            [2012,    1,   19,   12,   30,    0],
#            [2012,    1,   19,   12,   45,    0],
#            [2012,    1,   19,   13,    0,    0],
#            [2012,    1,   19,   13,   15,    0],
#            [2012,    1,   19,   13,   30,    0],
#            [2012,    1,   19,   13,   45,    0],
#            [2012,    1,   19,   14,    0,    0],
#            [2012,    1,   19,   14,   15,    0],
#            [2012,    1,   19,   14,   30,    0],
#            [2012,    1,   19,   14,   45,    0],
#            [2012,    1,   19,   15,    0,    0],
#            [2012,    1,   19,   15,   15,    0],
#            [2012,    1,   19,   15,   30,    0],
#            [2012,    1,   19,   15,   45,    0],
#            [2012,    1,   19,   16,    0,    0],
#            [2012,    1,   19,   16,   15,    0],
#            [2012,    1,   19,   16,   30,    0],
#            [2012,    1,   19,   16,   45,    0],
#            [2012,    1,   19,   17,    0,    0],
#            [2012,    1,   19,   17,   15,    0],
#            [2012,    1,   19,   17,   30,    0],
#            [2012,    1,   19,   17,   45,    0],
#            [2012,    1,   19,   18,    0,    0],
#            [2012,    1,   19,   18,   15,    0],
#            [2012,    1,   19,   18,   30,    0],
#            [2012,    1,   19,   18,   45,    0],
#            [2012,    1,   19,   19,    0,    0],
#            [2012,    1,   19,   19,   15,    0],
#            [2012,    1,   19,   19,   30,    0],
#            [2012,    1,   19,   19,   45,    0],
#            [2012,    1,   19,   20,    0,    0],
#            [2012,    1,   19,   20,   15,    0],
#            [2012,    1,   19,   20,   30,    0],
#            [2012,    1,   19,   20,   45,    0],
#            [2012,    1,   19,   21,    0,    0],
#            [2012,    1,   19,   21,   15,    0],
#            [2012,    1,   19,   21,   30,    0],
#            [2012,    1,   19,   21,   45,    0],
#            [2012,    1,   19,   22,    0,    0],
#            [2012,    1,   19,   22,   15,    0],
#            [2012,    1,   19,   22,   30,    0],
#            [2012,    1,   19,   22,   45,    0],
#            [2012,    1,   19,   23,    0,    0],
#            [2012,    1,   19,   23,   15,    0],
#            [2012,    1,   19,   23,   30,    0],
#            [2012,    1,   19,   23,   45,    0]]))
#     # r_time = from_array_to_datetime(data['r_time'])
#     r_h = np.array([[ 91.46317376],
#            [ 94.45832965],
#            [ 97.58579014],
#            [100.70052912],
#            [103.70339597],
#            [106.70131251],
#            [109.94666419],
#            [113.57783654],
#            [118.09858718],
#            [123.86002718],
#            [130.60948667],
#            [138.3173866 ],
#            [147.00535981],
#            [156.73585448],
#            [167.59278336],
#            [179.57721244],
#            [192.43624664],
#            [206.30300179],
#            [221.17469246],
#            [237.20685488],
#            [254.39143579],
#            [272.34757393],
#            [291.47736465],
#            [311.68243288],
#            [332.82898428],
#            [354.92232018],
#            [377.86223818]]).flatten()
#     # r_h = data['r_h'].flatten()
#     r_param = data['r_param']
#     # r_error = data['r_error'] 
    
    
#     fig, ax = plt.subplots()
    
#     ne=ax.pcolormesh(r_time, r_h, r_param, shading="auto", cmap="turbo", norm=colors.LogNorm(vmin=1e10, vmax=1e12))
#     ax.set_xlabel("Time (UT)")
#     ax.set_ylabel("Altitudes (km)")
#     ax.xaxis.set_major_formatter(DateFormatter('%H:%M'))
#     fig.colorbar(ne, ax=ax, orientation='vertical')
#     plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='center')
#     plt.show()











