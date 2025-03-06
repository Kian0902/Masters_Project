# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 17:05:47 2024

@author: Kian Sartipzadeh
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


from torchvision import transforms
from storing_dataset import Matching3Pairs, Store3Dataset


from KIAN_Net_utils import from_strings_to_array, filter_artist_times, load_dict, save_dict, convert_pred_to_dict, from_array_to_datetime, from_strings_to_datetime, from_csv_to_numpy, add_key_from_dict_to_dict, add_key_with_matching_times, inspect_dict, convert_ionograms_to_dict, convert_geophys_to_dict

from Geo_DMLP_model import GeoDMLP
from Iono_CNN_model import IonoCNN
from KIAN_Net_model import FuDMLP, KIANNet


from matplotlib.dates import DateFormatter
import matplotlib.colors as colors
from matplotlib.gridspec import GridSpec

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)



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
    ne_EISCAT = ax0.pcolormesh(r_time, r_h, ne_eis, shading='auto', cmap='turbo', norm=colors.LogNorm(vmin=1e10, vmax=1e12))
    ax0.set_title('EISCAT UHF', fontsize=17)
    ax0.set_xlabel('Time [hh:mm]', fontsize=13)
    ax0.set_ylabel('Altitude [km]', fontsize=15)
    ax0.xaxis.set_major_formatter(DateFormatter('%H:%M'))
    
    # Plotting DL model
    ax1.pcolormesh(r_time, r_h, ne_pred, shading='auto', cmap='turbo', norm=colors.LogNorm(vmin=1e10, vmax=1e12))
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



def model_predict(stored_dataset, model_weights):

    A = stored_dataset
    
    # Creating DataLoader
    batch_size = len(A)
    test_loader = DataLoader(A, batch_size=batch_size, shuffle=False)
    
    iono_cnn = IonoCNN()
    iono_cnn.load_state_dict(torch.load("Iono_CNNv2.pth", weights_only=True))  # Pre-trained weights

    geo_dmlp = GeoDMLP()
    geo_dmlp.load_state_dict(torch.load("Geo_DMLPv7.pth", weights_only=True))

    fu_dmlp = FuDMLP(input_size=14848)  # Fusion 
    
    model = KIANNet(iono_cnn.conv, geo_dmlp.fc1, fu_dmlp).to(device)
    criterion = nn.HuberLoss()
    
    
    # Loading the trained network weights
    weights_path = model_weights
    model.load_state_dict(torch.load(weights_path, weights_only=True))
    model.to(device)
    
    model.eval()
    
    predictions = []
    
    with torch.no_grad():
        for data1, data2, targets in test_loader:
            
            data1 = data1.to(device)
            data2 = data2.to(device)
            targets = targets.to(device)
            
            outputs = model(data1, data2)
            
            loss = criterion(outputs, targets)
            print(loss)
            predictions.extend(outputs.cpu().numpy())
    
    
    model_ne = np.array(predictions)
    return model_ne




if __name__ == "__main__":
    
    # Test data folder names
    test_ionogram_folder = "testing_data/iono_test_flow_new"
    test_radar_folder = "testing_data/eis_test_flow_new"
    test_geophys_folder = "testing_data/geo_test_flow_new"
    
    # Initializing class for matching pairs
    Pairs = Matching3Pairs(test_ionogram_folder, test_radar_folder, test_geophys_folder)
    
    
    # Returning matching sample pairs
    rad, ion, geo, radar_times = Pairs.find_pairs(return_date=True)
    r_t = from_strings_to_datetime(radar_times)
    r_times = from_strings_to_array(radar_times)
    
    
    
    rad = np.abs(rad)
    rad[rad < 1e5] = 1e6
    
    
    # Storing the sample pairs
    A = Store3Dataset(ion, geo, np.log10(rad), transforms.Compose([transforms.ToTensor()]))
    
    # Path to trained weights
    weights_path = 'KIAN_Net_v1_NoPrevGrad.pth'
    
    
    X_pred = model_predict(A, weights_path)
    X_kian = convert_pred_to_dict(r_t, r_times, X_pred)
    
    
    
    X_true = from_csv_to_numpy(test_radar_folder)[0]
    X_eis = convert_pred_to_dict(r_t, r_times, X_true)
    
    
    # Adding 'r_h' from eiscat to all dicts
    Eiscat_support = load_dict("X_avg_test_data")
    X_eis = add_key_from_dict_to_dict(Eiscat_support, X_eis, key="r_h")
    X_eis = add_key_with_matching_times(Eiscat_support, X_eis, key="r_error")
    
    
    X_kian = add_key_from_dict_to_dict(Eiscat_support, X_kian, key="r_h")

    
    for day in X_eis:
        plot_compare(X_eis[day], X_kian[day])
    
    


