# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 16:56:34 2024

@author: Kian Sartipzadeh
"""


import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn.cluster import DBSCAN
from sklearn.metrics import pairwise_distances
from matplotlib.lines import Line2D
import seaborn as sns
sns.set(style="dark", context=None, palette=None)


def import_file(file_name):
    with open(file_name, 'rb') as f:
        dataset = pickle.load(f)
    
    return dataset



def detect_nan_in_arrays(data_dict):
    for key, array in data_dict.items():
        if np.isnan(array).any():
            print(f"NaN detected {key}: {array.shape}")



def create_ml_dataset(data):
    samples = []
    
    for date, day_data in data.items():
        r_time = day_data["r_time"]  # Shape (M, 6)
        r_param = day_data["r_param"]  # Shape (N, M)
        r_h = day_data["r_h"]         # Shape (N, 1)
        
        # Transpose r_param to make it (M, N)
        r_param_transposed = r_param.T
        
        
        # Iterate over each measurement (M)
        for i in range(r_time.shape[0]):
            # Combine date-time information with features
            date_time_info = r_time[i]  # Shape (6,)
            features = r_param_transposed[i]  # Shape (N,)
            sample = np.concatenate([date_time_info, features])  # Concatenate date-time and features
            samples.append(sample)
    
    # Convert the list of samples to a numpy array
    dataset_array = np.array(samples)
    X = dataset_array[:,6:]
    r = r_h.flatten()
    times = dataset_array[:,:6]
    return X, r, times







def pca(data: np.ndarray, reduce_to_dim: int=2):
    """
    Reduces the data features into 2 dimensions.
    
    
    Input (type)    | DESCRIPTION
    ------------------------------------------------
    data (np.array) | Data to be reduced.
    
    
    Return (type)          | DESCRIPTION
    ------------------------------------------------
    pca_data (np.ndarray)  | Reduced data.
    """
    
    PCA_model = PCA(n_components = reduce_to_dim)
    pca_data = PCA_model.fit_transform(data.T)
    return pca_data.T








data = import_file("X_averaged")


X, r, t = create_ml_dataset(data)
X[X < 1e6] = 1e6





PCA_model = PCA(n_components = 3)
X_pca = PCA_model.fit_transform(np.log10(X))







# =============================================================================
#                        Data plots after PCA
# =============================================================================

n=X.shape[0]
#____________ PC 1 and 2 ____________
fig, ax = plt.subplots(figsize=(6, 6))
fig.tight_layout()

# Plot for Principal Components 1 and 2
sns.scatterplot(x=X_pca[:n, 0], y=X_pca[:n, 1], ax=ax, label="Reduced Electron Densities", color="C0", s=30, edgecolor="black", linewidth=0.5)
ax.set_title("Dataset after PCA feature reduction", fontsize=15)
ax.set_xlabel("PC 1", fontsize=15)
ax.set_ylabel("PC 2", fontsize=15)
ax.grid()
ax.legend()
plt.show()



#____________ PC 1 and 3 ____________
fig, ax = plt.subplots(figsize=(6, 6))
fig.tight_layout()

# Plot for Principal Components 1 and 2
sns.scatterplot(x=X_pca[:n, 0], y=X_pca[:n, 2], ax=ax, label="Reduced Electron Densities", color="C0", s=30, edgecolor="black", linewidth=0.5)
ax.set_title("Dataset after PCA feature reduction", fontsize=15)
ax.set_xlabel("PC 1", fontsize=15)
ax.set_ylabel("PC 3", fontsize=15)
ax.grid()
ax.legend()
plt.show()



#____________ PC 2 and 3 ____________
fig, ax = plt.subplots(figsize=(6, 6))
fig.tight_layout()

# Plot for Principal Components 1 and 2
sns.scatterplot(x=X_pca[:n, 1], y=X_pca[:n, 2], ax=ax, label="Reduced Electron Densities", color="C0", s=30, edgecolor="black", linewidth=0.5)
ax.set_title("Dataset after PCA feature reduction", fontsize=15)
ax.set_xlabel("PC 2", fontsize=15)
ax.set_ylabel("PC 3", fontsize=15)
ax.grid()
ax.legend()
plt.show()


















