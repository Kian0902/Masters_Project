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


# from eval_utils import save_dict, convert_pred_to_dict, from_csv_to_numpy, from_strings_to_array, from_strings_to_datetime, plot_compare, plot_compare_r2, plot_results


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)




def apply_model(stored_dataset, DL_model, model_weights):

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
    true_targets = []
    
    
    with torch.no_grad():
        for data1, data2, targets in test_loader:
            
            data1 = data1.to(device)
            data2 = data2.to(device)
            targets = targets.to(device)
            
            
            outputs = model(data1, data2)
            
            loss = criterion(outputs, targets)
            print(loss)
            predictions.extend(outputs.cpu().numpy())
            true_targets.extend(targets.cpu().numpy())
    
    
    model_ne = np.array(predictions)
    eiscat_ne = np.array(true_targets)
    
    return model_ne

# # Calculate R² score for each feature
# r2_scores = [r2_score(eiscat_ne[:, i], model_ne[:, i]) for i in range(model_ne.shape[1])]

# # Print R² scores for all features
# for i, r2 in enumerate(r2_scores):
#     print(f"R² score for feature {i + 1}: {r2:.4f}")



# X_pred = convert_pred_to_dict(r_t, r_times, model_ne)

# save_dict(X_pred, "hnn_ne_pred")

# plot_compare_r2(eiscat_ne, model_ne, r2_scores, r_times)


# # for i in range(0, eiscat_ne.shape[0]):
# #     plot_results(model_ne[i], eiscat_ne[i])







