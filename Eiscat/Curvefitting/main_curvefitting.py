# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 14:53:48 2024

@author: Kian Sartipzadeh
"""



import os
import pickle

from curvefitting_models import CurvefittingChapman



# Importing processed dataset
custom_file_path = "Ne_vhf_avg"
with open(custom_file_path, 'rb') as f:
    dataset = pickle.load(f)



X = dataset['2018-11-10']

for key in X:
    print(type(X[key]), f'{key}: {X[key].shape}')



m = 'scipy'
# m = 'lmfit'
# m = 'NN'

A = CurvefittingChapman(X)
A.get_curvefits(X, m)







# import os
# import pickle

# import numpy as np
# import matplotlib.pyplot as plt

# from curvefitting_NN import Curvefit_NN

# import torch
# import torch.nn as nn



# # Importing processed dataset
# custom_file_path = "Ne_vhf_avg"
# with open(custom_file_path, 'rb') as f:
#     dataset = pickle.load(f)



# # Loop through day
# for day in dataset.keys():
    
#     # Dict for day
#     data_day = dataset[day]
    

#     # True EISCAT data
#     Z_true = np.tile(np.array(data_day["r_h"]), 32).T        # altitude [km] 
#     Ne_true = np.array(data_day["r_param"]).T   # electron density []
#     print(Z_true.shape)
#     print(Ne_true.shape)
    
#     # Loop though time of day
#     for i in np.arange(0, len(Z_true)):
        
        
#         # Indexing time of measurement (averaged over 15 min)
#         z_true = Z_true[i]
#         ne_true = Ne_true[i]
        
#         # Finding peak electron densities
#         neE_peak = np.max(ne_true[(150 >= z_true) & (z_true > 100)])
#         neF_peak = np.max(ne_true[(300 >= z_true) & (z_true > 150)])
        
#         # Finding altitude at peak
#         zE_peak = z_true[ne_true==neE_peak]
#         zF_peak = z_true[ne_true==neF_peak]
        
        
#         # Model, criterion, and optimizer
#         model = Curvefit_NN()
#         criterion = nn.MSELoss()
#         optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        
#         # Converting data to PyTorch Tensors
#         Z = torch.from_numpy(z_true).double().view(-1, 1)
#         Ne = torch.from_numpy(ne_true).double().view(-1, 1)
        
#         for epoch in range(2001):
            
#             output = model(Z, neE_peak, neF_peak, zE_peak, zF_peak)
#             loss   = criterion(output, Ne)
            
#             optimizer.zero_grad()
            
#             loss.backward()
#             optimizer.step()

#             # if epoch%100==0:
#                 # print(f"Epoch {epoch} Loss: {loss.item()}")


#             if epoch == 1000:
#                 print(f"Epoch {epoch} Loss: {loss.item()}")
#                 with torch.no_grad():
#                     pred = model(Z, neE_peak, neF_peak, zE_peak, zF_peak)
#                     plt.plot(pred.numpy(), Z.numpy(), label='Predicted')
#                     plt.plot(ne_true, z_true, label='Actual')
#                     plt.xscale("log")
#                     plt.legend()
#                     plt.show()
        
#     break































































