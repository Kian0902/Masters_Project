# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 13:12:37 2025

@author: Kian Sartipzadeh
"""


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.dates import DateFormatter
from sklearn.cluster import HDBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from data_utils import load_dict, inspect_dict, from_array_to_datetime



def plot_day(data):
    r_time = from_array_to_datetime(data['r_time'])
    r_h = data['r_h'].flatten()
    r_param = data['r_param']
    r_error = data['r_error'] 
    
    
    fig, ax = plt.subplots()
    
    ne=ax.pcolormesh(r_time, r_h, r_param, shading="auto", cmap="turbo", norm=colors.LogNorm(vmin=1e10, vmax=1e11))
    ax.set_xlabel("Time (UT)")
    ax.set_ylabel("Altitudes (km)")
    ax.xaxis.set_major_formatter(DateFormatter('%H:%M'))
    fig.colorbar(ne, ax=ax, orientation='vertical')
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='center')
    plt.show()





d = '2019-1-10'


X = load_dict("X_filt_few")
x = X[d]['r_param'].T
r = X[d]['r_h'].flatten()

hdb = HDBSCAN(min_cluster_size=12)
hdb.fit(np.log10(x))


labels = hdb.labels_
unique, frequency = np.unique(hdb.labels_, return_counts = True)

print(unique)
print(frequency)

principal=PCA(n_components=3)
principal.fit(np.log10(x))
z=principal.transform(np.log10(x))





COL = {}
for i, label in enumerate(unique):
    COL.update({int(label):"C"+ f"{i}"})
    

for key, val in COL.items():
    print(key, val)
    plt.scatter(0, 0, color=COL[key], label=f'{key}')
    


for i, label in enumerate(labels):
    plt.scatter(z[i, 0], z[i, 1], color=COL[label])



plt.legend()
plt.show()

print(unique)

for key, val in COL.items():
    print(key, val)
    plt.scatter(0, 0, color=COL[key], label=f'{key}')

for i, label in enumerate(labels):
    plt.scatter(z[i, 0], z[i, 2], color=COL[label])


plt.legend()
plt.show()

print(unique)
for key, val in COL.items():
    print(key, val)
    plt.scatter(0, 0, color=COL[key], label=f'{key}')

for i, label in enumerate(labels):
    plt.scatter(z[i, 1], z[i, 2], color=COL[label])



plt.legend()
plt.show()





for i, label in enumerate(labels):
    # print(x[i, :].shape)
    plt.plot(x[label==1][i, :], r)
    plt.xlabel("Ne")
    plt.ylabel("Alt")
    # plt.xscale("log")
    break
plt.show()

















# import os
# import numpy as np
# import matplotlib.pyplot as plt


# import torch
# from torch.utils.data import Dataset



# # Store training pairs
# class StoreDataset(Dataset):
#     def __init__(self, data, targets):
#         self.targets = torch.tensor(np.array(targets), dtype=torch.float32)
#         self.targets = torch.tensor(np.array(targets), dtype=torch.float32)
        
#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
        
#         return self.data[idx], self.targets[idx]






# data_folder = "EISCAT_MAT/UHF_All"











