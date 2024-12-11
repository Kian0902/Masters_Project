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


from eval_utils import add_key_with_matching_times,inspect_dict, save_dict, load_dict, add_key_from_dict_to_dict, convert_pred_to_dict, convert_ionograms_to_dict, from_csv_to_numpy, from_strings_to_array, from_strings_to_datetime, filter_artist_times
from hnn_model import CombinedNetwork


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




if __name__ == "__main__":

    # Test data folder names
    test_ionogram_folder = "testing_data/test_ionogram_folder_E+F"
    test_radar_folder = "testing_data/test_eiscat_folder_E+F"        # These are the true data
    test_sp19_folder = "testing_data/test_geophys_folder_E+F"


    # Initializing class for matching pairs
    Pairs = Matching3Pairs(test_ionogram_folder, test_radar_folder, test_sp19_folder)


    # Returning matching sample pairs
    rad, ion, sp, radar_times = Pairs.find_pairs(return_date=True)
    r_t = from_strings_to_datetime(radar_times)
    r_times = from_strings_to_array(radar_times)

    rad = np.abs(rad)
    rad[rad < 1e5] = 1e6


    # Storing the sample pairs
    A = Store3Dataset(ion, sp, np.log10(rad), transforms.Compose([transforms.ToTensor()]))

    # Path to trained weights
    # weights_path = 'Model_3333.pth'
    weights_path = 'HNN_v1_best_weights.pth'


    X_pred = model_predict(A, CombinedNetwork(), weights_path)
    # X_kian = convert_pred_to_dict(r_t, r_times, X_pred)
    
    
    # X_true = from_csv_to_numpy(test_radar_folder)[0]
    # X_eis = convert_pred_to_dict(r_t, r_times, X_true)
    
    
    # # print(inspect_dict(X_eis))
    
    
    # X_art = load_dict("processed_artist_test_days.pkl")
    # X_art = filter_artist_times(X_eis, X_kian, X_art)
    
    
    # # Adding 'r_h' from eiscat to all dicts
    # Eiscat_support = load_dict("X_avg_test_data")
    # X_eis = add_key_from_dict_to_dict(Eiscat_support, X_eis, key="r_h")
    # X_eis = add_key_with_matching_times(Eiscat_support, X_eis, key="r_error")
    
    
    # # print(inspect_dict(X_eis))
    
    # X_kian = add_key_from_dict_to_dict(Eiscat_support, X_kian, key="r_h")
    # X_art = add_key_from_dict_to_dict(Eiscat_support, X_art, key="r_h")
    # X_ion = convert_ionograms_to_dict(ion, X_eis)
    


    # save_dict(X_eis, "X_eis.pkl")
    # save_dict(X_kian, "X_kian.pkl")
    # save_dict(X_art, "X_art.pkl")
    # save_dict(X_ion, "X_ion.pkl")


































