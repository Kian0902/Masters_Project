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
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from storing_dataset import Matching3Pairs, Store3Dataset
from hnn_model import CombinedNetwork





# import seaborn as sns
# sns.set(style="dark", context=None, palette=None)






device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)



def import_file(file_name):
    with open(file_name, 'rb') as f:
        dataset = pickle.load(f)
    
    return dataset


def plot_compare(ne_true, ne_pred):
    """
    Plot a comparison of original and averaged data using pcolormesh.

    Input (type)                 | DESCRIPTION
    ------------------------------------------------
    original_data (dict)         | Dictionary containing the original data.
    """
    

    r_h = np.array([[91.5687711 ],[ 94.57444598],[ 97.57964223],[100.57010953],
           [103.57141624],[106.57728701],[110.08393175],[114.60422289],
           [120.1185208 ],[126.61221111],[134.1346149 ],[142.53945817],
           [152.05174717],[162.57986185],[174.09833378],[186.65837945],
           [200.15192581],[214.62769852],[230.12198695],[246.64398082],
           [264.11728204],[282.62750673],[302.15668686],[322.70723831],
           [344.19596481],[366.64409299],[390.113117  ]])
    
    
    x = range(ne_pred.shape[1])
    

    # Creating the plots
    fig, ax = plt.subplots(1, 2, figsize=(10, 8), sharey=True)
    fig.tight_layout()


    # Plotting original data
    ne_EISCAT = ax[0].pcolormesh(x, r_h.flatten(), ne_true, shading='auto', cmap='turbo', vmin=10, vmax=12)
    ax[0].set_title('EISCAT UHF', fontsize=17)
    ax[0].set_xlabel('Num samples')
    ax[0].set_ylabel('Altitude [km]')
    
    
    # Plotting original data
    ne_Artist = ax[1].pcolormesh(x, r_h.flatten(), ne_pred, shading='auto', cmap='turbo', vmin=10, vmax=12)
    ax[1].set_title('HNN Ours', fontsize=17)
    ax[1].set_xlabel('Num samples')
    
    # Add colorbar for the original data
    cbar = fig.colorbar(ne_EISCAT, ax=ax[1], orientation='vertical', fraction=0.03, pad=0.04, aspect=44, shrink=3)
    cbar.set_label(r'$log_{10}(n_e)$ [g/cm$^3$]', fontsize=17)
    
    
    # Display the plots
    plt.show()

def plot_results(ne_pred, ne_true):
    
    
    r_h = np.array([[91.5687711 ],[ 94.57444598],[ 97.57964223],[100.57010953],
           [103.57141624],[106.57728701],[110.08393175],[114.60422289],
           [120.1185208 ],[126.61221111],[134.1346149 ],[142.53945817],
           [152.05174717],[162.57986185],[174.09833378],[186.65837945],
           [200.15192581],[214.62769852],[230.12198695],[246.64398082],
           [264.11728204],[282.62750673],[302.15668686],[322.70723831],
           [344.19596481],[366.64409299],[390.113117  ]])
    
    
    
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.suptitle('Model Results', fontsize=15)
    
    ax.set_title('EISCAT vs HNN', fontsize=15)
    ax.plot(ne_true, r_h.flatten(), color="C0", label="EISCAT_ne")
    ax.plot(ne_pred, r_h.flatten(), color="C1", label="Pred_ne")
    ax.set_xlabel("Electron Density  log10(ne)")
    ax.set_ylabel("Altitude (km)")
    ax.set_xlim(8.9, 12.1)
    ax.grid(True)
    ax.legend()
    
    plt.show()







# # Data Folder names
# test_ionogram_folder = "testing_data/test_ionogram_folder"
# test_radar_folder = "testing_data/test_radar_folder"        # These are the true data
# test_sp19_folder = "testing_data/test_sp19_folder"



# # Initializing class for matching pairs
# Pairs = Matching3Pairs(test_ionogram_folder, test_radar_folder, test_sp19_folder)

# # Returning matching sample pairs
# rad, ion, sp = Pairs.find_pairs()


# rad = np.abs(rad)
# rad[rad < 1e5] = 1e6


# # Storing the sample pairs
# A = Store3Dataset(ion, sp, np.log10(rad), transforms.Compose([transforms.ToTensor()]))


# # Creating DataLoader
# batch_size = len(A)
# test_loader = DataLoader(A, batch_size=batch_size, shuffle=False)



# model = CombinedNetwork()
# criterion = nn.HuberLoss()

# # Loading the trained network weights
# weights_path = 'best_model_weights.pth'
# model.load_state_dict(torch.load(weights_path, weights_only=True))
# model.to(device)



# model.eval()



# predictions = []
# true_targets = []


# with torch.no_grad():
#     for data1, data2, targets in test_loader:
        
#         data1 = data1.to(device)
#         data2 = data2.to(device)
#         targets = targets.to(device)
        
        
#         outputs = model(data1, data2)
        
#         loss = criterion(outputs, targets)
#         print(loss)
#         predictions.extend(outputs.cpu().numpy())
#         true_targets.extend(targets.cpu().numpy())


# model_ne = np.array(predictions).T
# eiscat_ne = np.array(true_targets).T



# plot_compare(eiscat_ne, model_ne)



# for i in range(0, len(model_ne)):
#     plot_results(model_ne[i], eiscat_ne[i])






 # You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
 #  model.load_state_dict(torch.load(weights_path))







