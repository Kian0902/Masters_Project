# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 18:29:17 2024

@author: Kian Sartipzadeh
"""

import numpy as np

from torchvision import transforms
from storing_dataset import Matching3Pairs, Store3Dataset


from eval_utils import save_dict, load_dict, add_key_from_dict_to_dict, convert_pred_to_dict, convert_ionograms_to_dict, from_csv_to_numpy, from_strings_to_array, from_strings_to_datetime, filter_artist_times, inspect_dict
from eval_predict import apply_model
from hnn_model import CombinedNetwork



# Test data folder names
test_ionogram_folder = "testing_data/test_ionogram_folder"
test_radar_folder = "testing_data/test_eiscat_folder"        # These are the true data
test_sp19_folder = "testing_data/test_geophys_folder"


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


X_pred = apply_model(A, CombinedNetwork(), weights_path)
X_kian = convert_pred_to_dict(r_t, r_times, X_pred)


X_true = from_csv_to_numpy(test_radar_folder)[0]
X_eis = convert_pred_to_dict(r_t, r_times, X_true)


X_art = load_dict("processed_artist_test_days.pkl")
X_art = filter_artist_times(X_eis, X_kian, X_art)


# Adding 'r_h' from eiscat to all dicts
Eiscat_support = load_dict("X_avg_test_data")
X_eis = add_key_from_dict_to_dict(Eiscat_support, X_eis)
X_kian = add_key_from_dict_to_dict(Eiscat_support, X_kian)
X_art = add_key_from_dict_to_dict(Eiscat_support, X_art)
X_ion = convert_ionograms_to_dict(ion, X_eis)



save_dict(X_eis, "X_eis.pkl")
save_dict(X_kian, "X_kian.pkl")
save_dict(X_art, "X_art.pkl")
save_dict(X_ion, "X_ion.pkl")





