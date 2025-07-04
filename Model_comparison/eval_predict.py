# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 18:29:17 2024

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


from eval_utils import from_strings_to_array, filter_artist_times, load_dict, save_dict, convert_pred_to_dict, from_array_to_datetime, from_strings_to_datetime, from_csv_to_numpy, add_key_from_dict_to_dict, add_key_with_matching_times, inspect_dict, convert_ionograms_to_dict, convert_geophys_to_dict
from hnn_model import CombinedNetwork


from matplotlib.dates import DateFormatter
import matplotlib.colors as colors


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)




def model_predict(stored_dataset, DL_model, model_weights):

    A = stored_dataset
    
    # Creating DataLoader
    batch_size = len(A)
    test_loader = DataLoader(A, batch_size=batch_size, shuffle=False)
    
    
    
    model = DL_model
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


def plot_day(data):
    r_time = from_array_to_datetime(np.array([[2012,    1,   19,    0,    0,    0],
           [2012,    1,   19,    0,   15,    0],
           [2012,    1,   19,    0,   30,    0],
           [2012,    1,   19,    0,   45,    0],
           [2012,    1,   19,    1,    0,    0],
           [2012,    1,   19,    1,   15,    0],
           [2012,    1,   19,    1,   30,    0],
           [2012,    1,   19,    1,   45,    0],
           [2012,    1,   19,    2,    0,    0],
           [2012,    1,   19,    2,   15,    0],
           [2012,    1,   19,    2,   30,    0],
           [2012,    1,   19,    2,   45,    0],
           [2012,    1,   19,    3,    0,    0],
           [2012,    1,   19,    3,   15,    0],
           [2012,    1,   19,    3,   30,    0],
           [2012,    1,   19,    3,   45,    0],
           [2012,    1,   19,    4,    0,    0],
           [2012,    1,   19,    4,   15,    0],
           [2012,    1,   19,    4,   30,    0],
           [2012,    1,   19,    4,   45,    0],
           [2012,    1,   19,    5,    0,    0],
           [2012,    1,   19,    5,   15,    0],
           [2012,    1,   19,    5,   30,    0],
           [2012,    1,   19,    5,   45,    0],
           [2012,    1,   19,    6,    0,    0],
           [2012,    1,   19,    6,   15,    0],
           [2012,    1,   19,    6,   30,    0],
           [2012,    1,   19,    6,   45,    0],
           [2012,    1,   19,    7,    0,    0],
           [2012,    1,   19,    7,   15,    0],
           [2012,    1,   19,    7,   30,    0],
           [2012,    1,   19,    7,   45,    0],
           [2012,    1,   19,    8,    0,    0],
           [2012,    1,   19,    8,   15,    0],
           [2012,    1,   19,    8,   30,    0],
           [2012,    1,   19,    8,   45,    0],
           [2012,    1,   19,    9,    0,    0],
           [2012,    1,   19,    9,   15,    0],
           [2012,    1,   19,    9,   30,    0],
           [2012,    1,   19,    9,   45,    0],
           [2012,    1,   19,   10,    0,    0],
           [2012,    1,   19,   10,   15,    0],
           [2012,    1,   19,   10,   30,    0],
           [2012,    1,   19,   10,   45,    0],
           [2012,    1,   19,   11,    0,    0],
           [2012,    1,   19,   11,   15,    0],
           [2012,    1,   19,   11,   30,    0],
           [2012,    1,   19,   11,   45,    0],
           [2012,    1,   19,   12,    0,    0],
           [2012,    1,   19,   12,   15,    0],
           [2012,    1,   19,   12,   30,    0],
           [2012,    1,   19,   12,   45,    0],
           [2012,    1,   19,   13,    0,    0],
           [2012,    1,   19,   13,   15,    0],
           [2012,    1,   19,   13,   30,    0],
           [2012,    1,   19,   13,   45,    0],
           [2012,    1,   19,   14,    0,    0],
           [2012,    1,   19,   14,   15,    0],
           [2012,    1,   19,   14,   30,    0],
           [2012,    1,   19,   14,   45,    0],
           [2012,    1,   19,   15,    0,    0],
           [2012,    1,   19,   15,   15,    0],
           [2012,    1,   19,   15,   30,    0],
           [2012,    1,   19,   15,   45,    0],
           [2012,    1,   19,   16,    0,    0],
           [2012,    1,   19,   16,   15,    0],
           [2012,    1,   19,   16,   30,    0],
           [2012,    1,   19,   16,   45,    0],
           [2012,    1,   19,   17,    0,    0],
           [2012,    1,   19,   17,   15,    0],
           [2012,    1,   19,   17,   30,    0],
           [2012,    1,   19,   17,   45,    0],
           [2012,    1,   19,   18,    0,    0],
           [2012,    1,   19,   18,   15,    0],
           [2012,    1,   19,   18,   30,    0],
           [2012,    1,   19,   18,   45,    0],
           [2012,    1,   19,   19,    0,    0],
           [2012,    1,   19,   19,   15,    0],
           [2012,    1,   19,   19,   30,    0],
           [2012,    1,   19,   19,   45,    0],
           [2012,    1,   19,   20,    0,    0],
           [2012,    1,   19,   20,   15,    0],
           [2012,    1,   19,   20,   30,    0],
           [2012,    1,   19,   20,   45,    0],
           [2012,    1,   19,   21,    0,    0],
           [2012,    1,   19,   21,   15,    0],
           [2012,    1,   19,   21,   30,    0],
           [2012,    1,   19,   21,   45,    0],
           [2012,    1,   19,   22,    0,    0],
           [2012,    1,   19,   22,   15,    0],
           [2012,    1,   19,   22,   30,    0],
           [2012,    1,   19,   22,   45,    0],
           [2012,    1,   19,   23,    0,    0],
           [2012,    1,   19,   23,   15,    0],
           [2012,    1,   19,   23,   30,    0],
           [2012,    1,   19,   23,   45,    0]]))
    # r_time = from_array_to_datetime(data['r_time'])
    r_h = np.array([[ 91.46317376],
           [ 94.45832965],
           [ 97.58579014],
           [100.70052912],
           [103.70339597],
           [106.70131251],
           [109.94666419],
           [113.57783654],
           [118.09858718],
           [123.86002718],
           [130.60948667],
           [138.3173866 ],
           [147.00535981],
           [156.73585448],
           [167.59278336],
           [179.57721244],
           [192.43624664],
           [206.30300179],
           [221.17469246],
           [237.20685488],
           [254.39143579],
           [272.34757393],
           [291.47736465],
           [311.68243288],
           [332.82898428],
           [354.92232018],
           [377.86223818]]).flatten()
    # r_h = data['r_h'].flatten()
    r_param = data['r_param']
    # r_error = data['r_error'] 
    
    
    fig, ax = plt.subplots()
    
    ne=ax.pcolormesh(r_time, r_h, r_param, shading="auto", cmap="turbo", norm=colors.LogNorm(vmin=1e10, vmax=1e12))
    ax.set_xlabel("Time (UT)")
    ax.set_ylabel("Altitudes (km)")
    ax.xaxis.set_major_formatter(DateFormatter('%H:%M'))
    fig.colorbar(ne, ax=ax, orientation='vertical')
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='center')
    plt.show()


if __name__ == "__main__":
    
    
    # # Test data folder names
    # test_ionogram_folder = "testing_data/test_ionogram_folder"
    # test_radar_folder = "testing_data/test_eiscat_folder"        # These are the true data
    # test_sp19_folder = "testing_data/test_geophys_folder"
    
    test_ionogram_folder = "testing_data/test_ionogram_shutup_rusland_2"
    test_radar_folder = "testing_data/test_eiscat_shutup_rusland_2"        # These are the true data
    test_sp19_folder = "testing_data/test_geophys_shutup_rusland_2"
    
    
    # # Test data folder names
    # test_ionogram_folder = "hypothesis_data/hyp_ionogram_folder"
    # test_radar_folder = "hypothesis_data/hyp_eiscat_folder"        # These are the true data
    # test_sp19_folder = "hypothesis_data/hyp_geophys_folder"
    
    # test_ionogram_folder = "training_data/train_ionogram_folder"
    # test_radar_folder = "training_data/train_eiscat_folder"
    # test_sp19_folder = "training_data/train_geophys_folder"
    
    
    
    # Initializing class for matching pairs
    Pairs = Matching3Pairs(test_ionogram_folder, test_radar_folder, test_sp19_folder)
    
    
    # Returning matching sample pairs
    rad, ion, sp, radar_times = Pairs.find_pairs(return_date=True)
    r_t = from_strings_to_datetime(radar_times)
    r_times = from_strings_to_array(radar_times)
    
    
    
#     # print(type(sp), len(sp), sp[0])
    
    
    
    
    
    rad = np.abs(rad)
    rad[rad < 1e5] = 1e6
    
    
    # Storing the sample pairs
    A = Store3Dataset(ion, sp, np.log10(rad), transforms.Compose([transforms.ToTensor()]))
    
    # Path to trained weights
    # weights_path = 'Model_3333.pth'
    weights_path = 'HNN_v1_best_weights.pth'
    
    
    X_pred = model_predict(A, CombinedNetwork(), weights_path)
    X_kian = convert_pred_to_dict(r_t, r_times, X_pred)
    
    
    
    
    
    X_true = from_csv_to_numpy(test_radar_folder)[0]
    X_eis = convert_pred_to_dict(r_t, r_times, X_true)
    
    # plot_day(X_eis['2012-1-19'])
    # print(inspect_dict(X_eis))
    
    
    X_art = load_dict("processed_artist_test_days.pkl")
    X_art = filter_artist_times(X_eis, X_kian, X_art)
    
    
    # Adding 'r_h' from eiscat to all dicts
    Eiscat_support = load_dict("X_avg_test_data")
    X_eis = add_key_from_dict_to_dict(Eiscat_support, X_eis, key="r_h")
    X_eis = add_key_with_matching_times(Eiscat_support, X_eis, key="r_error")
    
    # print(inspect_dict(X_eis))
    
    X_kian = add_key_from_dict_to_dict(Eiscat_support, X_kian, key="r_h")
    X_art = add_key_from_dict_to_dict(Eiscat_support, X_art, key="r_h")
    X_ion = convert_ionograms_to_dict(ion, X_eis)
    X_geo = convert_geophys_to_dict(sp, X_eis)
    # inspect_dict(X_geo)
    # inspect_dict(X_eis)
    
    
    # plot_day(X_kian)
    
    
    save_dict(X_eis, "testing_data_shutup_rusland/X_eis.pkl")
    save_dict(X_kian, "testing_data_shutup_rusland/X_kian.pkl")
    save_dict(X_art, "testing_data_shutup_rusland/X_art.pkl")
    save_dict(X_ion, "testing_data_shutup_rusland/X_ion.pkl")
    save_dict(X_geo, "testing_data_shutup_rusland/X_geo.pkl")

































