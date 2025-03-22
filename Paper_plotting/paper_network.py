# -*- coding: utf-8 -*-
"""
Created on Sat Jan  4 15:29:07 2025

@author: Kian Sartipzadeh
"""


import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from torchvision import transforms
from storing_dataset import Matching3Pairs, Store3Dataset


from eval_utils import (
    add_key_with_matching_times,
    inspect_dict,
    save_dict,
    load_dict,
    add_key_from_dict_to_dict,
    convert_pred_to_dict,
    convert_ionograms_to_dict,
    from_csv_to_numpy,
    from_strings_to_array,
    from_strings_to_datetime,
    filter_artist_times,
)
from hnn_model import CombinedNetwork
import seaborn as sns


class GradientAnalyzer:
    def __init__(self, model, loader, device, altitudes):
        """
        Initializes the GradientAnalyzer with a model, data loader, device, and altitudes.

        Args:
            model: The trained neural network model.
            loader: A DataLoader that returns (images, geophys, targets).
            device: 'cpu' or 'cuda'.
            altitudes: A numpy.ndarray containing the 27 altitude values.
        """
        self.model = model
        self.loader = loader
        self.device = device
        self.altitudes = altitudes

    def compute_gradients(self, target_index=None):
        """
        Computes gradients of the model output w.r.t. the geophysical features
        for a single batch from the loader.

        Args:
            target_index: 
                - If None, sums across all output neurons (model output shape [B, 27]).
                - If an integer, computes gradient only w.r.t. that output index.

        Returns:
            grads: A tensor of shape [B, 19], containing gradients of the
                   chosen output (sum or target_index) w.r.t. each geophysical feature.
        """
        # Grab one batch from the loader
        images, geophys, _ = next(iter(self.loader))

        # Send data to the correct device
        images = images.to(self.device)
        geophys = geophys.to(self.device)

        # Enable gradient tracking on geophysical features
        geophys = geophys.clone().detach().requires_grad_(True)

        # Disable gradients on images
        images = images.clone().detach().requires_grad_(False)

        # Forward pass
        output = self.model(images, geophys)  # shape [B, 27]

        # Compute loss
        if target_index is None:
            loss = output.sum()
        else:
            loss = output[:, target_index].sum()

        # Clear any old gradients
        self.model.zero_grad(set_to_none=True)

        # Backward pass to compute gradients
        loss.backward()

        # Extract gradients w.r.t. geophysical features
        grads = geophys.grad.detach().cpu()

        return grads

    def analyze_feature_importance(self, target_index=None, feature_labels=None):
        """
        Analyzes the average absolute gradients of geophysical features.

        Args:
            target_index: 
                - If None, sums across all output neurons (model output shape [B, 27]).
                - If an integer, computes gradient only w.r.t. that output index.
            feature_labels: List of feature labels for visualization.

        Displays:
            Bar chart of average absolute gradients for each feature.
        """
        grads = self.compute_gradients(target_index)

        # Compute average absolute gradients per feature
        avg_abs_grads = grads.abs().mean(dim=0).numpy()

        # Display results
        if feature_labels is None:
            feature_labels = [f"Feature {i}" for i in range(len(avg_abs_grads))]

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(feature_labels, avg_abs_grads, color='royalblue')
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='center')
        ax.set_xlabel("Geophysical Features", fontsize=15)
        ax.set_ylabel("Average |Gradient|", fontsize=15)
        ax.set_title("AAG Test Set F (Summed over altitudes)", fontsize=17)
        fig.tight_layout()
        plt.show()

    def analyze_all_altitudes(self, feature_labels=None):
        """
        Analyzes and visualizes gradients for all altitude levels.

        Args:
            feature_labels: List of feature labels for visualization.

        Displays:
            Color plot of average absolute gradients for each altitude.
        """
        # Initialize storage for gradients across all altitudes
        all_grads = []

        for target_index in range(len(self.altitudes)):
            grads = self.compute_gradients(target_index)
            avg_abs_grads = grads.abs().mean(dim=0).numpy()
            all_grads.append(avg_abs_grads)

        all_grads = np.array(all_grads)  # Shape: [27, 19]

        # Plot the color plot
        fig, ax = plt.subplots(figsize=(10, 5))
        cax=ax.pcolormesh(np.arange(len(feature_labels)), self.altitudes, all_grads,
                          shading='auto', cmap='inferno', vmax=0.5)
        
        ax.set_xlabel("Geophysical Features", fontsize=15)
        ax.set_ylabel("Altitude (km)", fontsize=15)
        ax.set_title("AAG Test Set F (At each altitude)", fontsize=17)
        
        if feature_labels is not None:
            ax.set_xticks(np.arange(len(feature_labels)))
            ax.set_xticklabels(feature_labels, rotation=45, ha='right')
        
        fig.colorbar(cax, label="|Gradient|")
        fig.tight_layout()
        plt.show()



# Test data folder names
test_ionogram_folder = "testing_data/test_ionogram_folder_F"
test_radar_folder = "testing_data/test_eiscat_folder_F"
test_sp19_folder = "testing_data/test_geophys_folder_F"

# test_ionogram_folder = "training_data/train_ionogram_folder"
# test_radar_folder = "training_data/train_eiscat_folder"
# test_sp19_folder = "training_data/train_geophys_folder"

# Initialize the class for matching pairs
Pairs = Matching3Pairs(test_ionogram_folder, test_radar_folder, test_sp19_folder)

print("Matching Pairs..")
# Find matching sample pairs
rad, ion, sp, radar_times = Pairs.find_pairs(return_date=True)
r_t = from_strings_to_datetime(radar_times)
r_times = from_strings_to_array(radar_times)

print("Pairs Matched!")

# Some preprocessing on radar data
rad = np.abs(rad)
rad[rad < 1e5] = 1e6

# Store the sample pairs
A = Store3Dataset(ion, sp, np.log10(rad), transforms.Compose([transforms.ToTensor()]))

# Path to your trained weights
weights_path = 'HNN_v1_best_weights.pth'



#############################
#   1) BUILD A DATALOADER   #
#############################

num=len(A)

# You can choose a batch_size as you prefer; here let's just pick 16
test_loader = DataLoader(A, batch_size=num, shuffle=True)

# Decide whether to use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#############################
#   2) LOAD THE MODEL       #
#############################
model = CombinedNetwork().to(device)
model.load_state_dict(torch.load(weights_path, map_location=device))
model.eval()




# Example usage
altitudes =  np.array([92.83, 95.96, 99.07, 102.07, 105.07, 108.07, 111.58, 116.23, 121.87,
                       128.51, 136.13, 144.56, 154.20, 164.97, 176.63, 189.28, 203.04, 217.64,
                       233.24, 250.04, 267.79, 286.55, 306.33, 327.06, 348.74, 371.51, 395.35])

analyzer = GradientAnalyzer(model, test_loader, device, altitudes)
analyzer.analyze_feature_importance(feature_labels=[
    'DoY/366', 'ToD/1440', 'SZ/44', 'Kp', 'R', 'Dst', 'ap', 'AE', 'AL', 'AU', 
    'PC_pot', 'F10_7', 'Ly_alp', 'Bx', 'By', 'Bz', 'dBx', 'dBy', 'dBz'
])
analyzer.analyze_all_altitudes(feature_labels=[
    'DoY/366', 'ToD/1440', 'SZ/44', 'Kp', 'R', 'Dst', 'ap', 'AE', 'AL', 'AU', 
    'PC_pot', 'F10_7', 'Ly_alp', 'Bx', 'By', 'Bz', 'dBx', 'dBy', 'dBz'
])




# mean_attr_geo = attr_geophys.mean(dim=0)  # shape [19]
# # Then pass that as a single "sample"
# explainer.plot_geophys_bar(mean_attr_geo.unsqueeze(0), feature_labels, sample_idx=0)








# #############################
# #   3) DEFINE A FUNCTION    #
# #      FOR GRAD ANALYSIS    #
# #############################
# def analyze_features_with_gradients(model, loader, device, target_index=None):
#     """
#     Computes standard gradients of the output w.r.t. the geophysical features
#     on a single batch from the provided loader.
    
#     Args:
#         model: Your trained CombinedNetwork.
#         loader: A DataLoader that returns (images, geophys, targets).
#         device: 'cpu' or 'cuda'.
#         target_index: 
#             - If None, sums across all output neurons (model output shape [B, 27]).
#             - If an integer, computes gradient only w.r.t. that output index.

#     Returns:
#         grads: A tensor of shape [B, 19], containing gradients of the
#                chosen output (sum or target_index) w.r.t. each geophys feature.
#     """
#     # Grab one batch from the loader
#     images, geophys, _ = next(iter(loader))
    
#     # Send to correct device
#     images = images.to(device)
#     geophys = geophys.to(device)
    
#     # Allow gradient tracking on geophys
#     geophys = geophys.clone().detach().requires_grad_(True)
    
#     # We typically do NOT need grads on images if we only care about geophys
#     images = images.clone().detach().requires_grad_(False)
    
#     # Forward pass
#     output = model(images, geophys)  # shape [B, 27]
    
#     # If target_index is None, sum across all output neurons
#     if target_index is None:
#         loss = output.sum()
#     else:
#         loss = output[:, target_index].sum()
    
#     # Clear any old gradients
#     model.zero_grad(set_to_none=True)
    
#     # Backward to compute gradients
#     loss.backward()
    
#     # Extract the gradient w.r.t. geophys (19 features)
#     grads = geophys.grad.detach().cpu()
    
#     return grads


# # Let's compute the standard gradients for a single batch:
# grads_geo = analyze_features_with_gradients(model, test_loader, device, target_index=3)

# print("Gradient shape (geophys features):", grads_geo.shape)  
# # Expect: [batch_size, 19]

# # For a quick sense of importance, look at average absolute gradient per feature
# avg_abs_grads_geo = grads_geo.abs().mean(dim=0)
# print("\nAverage absolute gradient per geophysical feature:")
# for i, val in enumerate(avg_abs_grads_geo):
#     print(f"Feature {i}: {val.item():.4f}")


# feature_labels = [
#     'DoY/366',
#     'ToD/1440',
#     'SZ/44',
#     'Kp',
#     'R',
#     'Dst',
#     'ap',
#     'AE',
#     'AL',
#     'AU',
#     'PC_pot',
#     'F10_7',
#     'Ly_alp',
#     'Bx',
#     'By',
#     'Bz',
#     'dBx',
#     'dBy',
#     'dBz'
# ]

# # Convert avg_abs_grads_geo to a NumPy array if needed
# # (if it's a PyTorch tensor)
# grad_values = avg_abs_grads_geo.cpu().numpy()  # shape: [19]


# fig, ax = plt.subplots(figsize=(10, 5))
# ax.bar(feature_labels, grad_values, color='royalblue')
# plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='center')
# ax.set_ylabel("Average |Gradient|", fontsize=13)
# ax.set_title("Average Absolute Gradients for Geophysical Features", fontsize=15)
# fig.tight_layout()  # Minimizes label clipping
# plt.show()
    



