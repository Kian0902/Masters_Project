# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 16:56:31 2025

@author: kian0
"""



import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
from sklearn.decomposition import PCA, KernelPCA
from data_utils import save_dict, load_dict, inspect_dict




def ravel_dict(data_dict):
    return np.concatenate([data_dict[date]['r_param'] for date in data_dict], axis=1)

def ravel_dict_days(data_dict, days):
    return np.concatenate([data_dict[date]['r_param'] for date in days], axis=1)


def preprocess_data(X):
    X = X.T
    X = np.where(X > 0, X, np.nan)  # Replace negatives/zeros with NaN
    X = np.log10(X)
    X = np.nan_to_num(X, nan=0.0)  # Replace NaNs with 0
    X[X < 6] == 8
    # X = (X - np.min(X)) / (np.max(X) - np.min(X))  # Normalize to [0,1]
    return X


def apply_pca(data, reduce_dim=2):
    pca_model = PCA(n_components=reduce_dim)
    reduced_data = pca_model.fit_transform(data)
    return reduced_data


def apply_kpca(data, reduce_dim=2):
    kpca = KernelPCA(n_components=reduce_dim, kernel='rbf', gamma=1)
    X_kpca = kpca.fit_transform(data)
    # pca_model = PCA(n_components=reduce_dim)
    # reduced_data = pca_model.fit_transform(data)
    return X_kpca

def apply_mds(data, reduce_dim=2):
    embedding = MDS(n_components=reduce_dim, normalized_stress='auto')
    X_t = embedding.fit_transform(data)
    return X_t


r_h = np.array([[ 91.5687711 ],[ 94.57444598],[ 97.57964223],[100.57010953],
       [103.57141624],[106.57728701],[110.08393175],[114.60422289],
       [120.1185208 ],[126.61221111],[134.1346149 ],[142.53945817],
       [152.05174717],[162.57986185],[174.09833378],[186.65837945],
       [200.15192581],[214.62769852],[230.12198695],[246.64398082],
       [264.11728204],[282.62750673],[302.15668686],[322.70723831],
       [344.19596481],[366.64409299],[390.113117  ]])



Data = load_dict('X_avg_all')



# days_list = ['2022-6-21', '2022-3-31', '2022-3-23', '2022-12-21', '2020-11-16', '2020-11-17', '2020-11-21', '2020-11-22', '2020-11-23', '2020-11-26', '2020-11-28', '2020-11-3', '2020-11-6', '2020-12-1', '2020-12-10', '2019-10-23', '2019-10-24', '2019-10-27', '2019-10-28', '2019-10-29', '2019-10-30', '2019-10-9', '2019-11-1', '2019-11-11', '2019-11-12', '2019-11-13', '2019-11-14', '2019-11-15']

import time
X =  ravel_dict(Data)
X_filt = preprocess_data(X)
start_time = time.time()
x = apply_mds(X_filt[:2000], reduce_dim=3)
end_time = time.time() 
execution_time = end_time - start_time

print(f'{execution_time:.5f}')

X = X.T



# fig = plt.figure(figsize=(12, 12))
# ax = fig.add_subplot(projection='3d')


# ax.scatter(x[:, 0], x[:, 1], x[:, 2])
# plt.show()

# plt.scatter(x[:, 0], x[:, 1])
# plt.xlabel("PC0")
# plt.ylabel("PC1")
# plt.show()


# plt.scatter(x[:, 0], x[:, 2])
# plt.xlabel("PC0")
# plt.ylabel("PC2")
# plt.show()


# plt.scatter(x[:, 1], x[:, 2])
# plt.xlabel("PC1")
# plt.ylabel("PC2")
# plt.show()

import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib.widgets import Cursor


# Create figure and grid layout
fig = plt.figure(figsize=(10, 8))
gs = GridSpec(3, 2, width_ratios=[1, 1], wspace=0.3, hspace=0.4)
ax0 = fig.add_subplot(gs[0, 0])
ax1 = fig.add_subplot(gs[1, 0])
ax2 = fig.add_subplot(gs[2, 0])
ax3 = fig.add_subplot(gs[:, 1])

# Scatter plots for PC1 vs PC2, PC1 vs PC3, and PC2 vs PC3
scatter0 = ax0.scatter(x[:, 0], x[:, 1], c='blue', alpha=0.6)
# ax0.set_xlim(-6, 6)
# ax0.set_ylim(-3, 5)
ax0.set_xlabel('PC1')
ax0.set_ylabel('PC2')

scatter1 = ax1.scatter(x[:, 0], x[:, 2], c='blue', alpha=0.6)
# ax1.set_xlim(-6, 6)
# ax1.set_ylim(-4, 2.6)
ax1.set_xlabel('PC1')
ax1.set_ylabel('PC3')

scatter2 = ax2.scatter(x[:, 1], x[:, 2], c='blue', alpha=0.6)
# ax2.set_xlim(-6, 6)
# ax2.set_ylim(-3, 5)
ax2.set_xlabel('PC2')
ax2.set_ylabel('PC3')

# Initialize variables to store highlighted points
highlighted_idx = None

# Function to find the closest point to the click event
def find_closest_point(event, scatter_data, ax):
    if event.inaxes == ax:
        dist = np.sqrt((scatter_data[:, 0] - event.xdata)**2 + (scatter_data[:, 1] - event.ydata)**2)
        return np.argmin(dist)
    return None

# Mouse click event handler
def on_click(event):
    global highlighted_idx

    # # Reset previous highlight
    # if highlighted_idx is not None:
    #     scatter0._facecolors[highlighted_idx] = 'blue'
    #     scatter1._facecolors[highlighted_idx] = 'blue'
    #     scatter2._facecolors[highlighted_idx] = 'blue'

    # Find the closest point in each scatter plot
    idx0 = find_closest_point(event, np.c_[x[:, 0], x[:, 1]], ax0)
    idx1 = find_closest_point(event, np.c_[x[:, 0], x[:, 2]], ax1)
    idx2 = find_closest_point(event, np.c_[x[:, 1], x[:, 2]], ax2)

    # Determine the final index based on which subplot was clicked
    if event.inaxes == ax0:
        highlighted_idx = idx0
    elif event.inaxes == ax1:
        highlighted_idx = idx1
    elif event.inaxes == ax2:
        highlighted_idx = idx2

    if highlighted_idx is not None:
        # Highlight the selected point in all scatter plots
        # scatter0._facecolors[highlighted_idx] = 'red'
        # scatter1._facecolors[highlighted_idx] = 'red'
        # scatter2._facecolors[highlighted_idx] = 'red'

        # Update the original data plot on ax3
        ax3.clear()
        ax3.plot(X[highlighted_idx], r_h.flatten(), label=f'Sample {highlighted_idx}', color='green')
        ax3.set_xscale("log")
        ax3.set_title(f'Electron Density Profile (Sample {highlighted_idx})')
        ax3.set_xlabel('Index')
        ax3.set_ylabel('r_h')
        ax3.legend()

    # Redraw the figure
    fig.canvas.draw_idle()

# Add Cursor widgets to the scatter plots
cursor0 = Cursor(ax0, useblit=True, color='red', linewidth=1)
cursor1 = Cursor(ax1, useblit=True, color='red', linewidth=1)
cursor2 = Cursor(ax2, useblit=True, color='red', linewidth=1)

# Connect the click event to the callback function
fig.canvas.mpl_connect('button_press_event', on_click)

# Initial plot on ax3 (optional, you can choose a default sample)
# idx = 3
# ax3.plot(X[idx], r_h.flatten(), label=f'Sample {idx}', color='green')
# ax3.set_title(f'Electron Density Profile (Sample {idx})')
# ax3.set_xlabel('Index')
# ax3.set_ylabel('r_h')
# ax3.legend()

# Show the plot
plt.show()
