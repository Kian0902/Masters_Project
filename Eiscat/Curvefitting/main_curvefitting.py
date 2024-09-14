# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 14:53:48 2024

@author: Kian Sartipzadeh
"""



import os
import pickle
from curvefitting_models import CurvefittingChapman



# Importing processed dataset
custom_file_path = "Ne_uhf_avg"
with open(custom_file_path, 'rb') as f:
    dataset = pickle.load(f)




key_choise = list(dataset.keys())[:]
X = {key: dataset[key] for key in key_choise}


print(len(X))

y=0
for day in X:
    size=X[day]['r_param'].shape[1]
    y+=size

print(y)

# m = 'scipy'
m = 'lmfit'
# m = 'NN'

A = CurvefittingChapman(X)
A.batch_detection(model_name=m, H_initial=[5, 10, 25, 40], save_plot=False)
A.save_curvefits(custom_file_path + "_" + m + "_curvefits")

x = A.return_curvefits()







































































