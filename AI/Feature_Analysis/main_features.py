# -*- coding: utf-8 -*-
"""
Created on Sat Sep 14 13:46:08 2024

@author: Kian Sartipzadeh
"""

import time
start_time = time.time()


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from feature_analysis import FeatureAnalysis



# Extracting the header
with open("spacephysics_19features.csv", "r") as f:
    header = f.readline().strip().split(",")

# Reading 19 space physics features
data = np.genfromtxt("spacephysics_19features.csv", dtype=float, delimiter=",", skip_header=0)

col_means = np.nanmean(data, axis=0)
data = np.where(np.isnan(data), col_means, data)




A = FeatureAnalysis(data[:, :], header[:])

A.correlation_matrix(plot_correlation_matrix=True, plot_correlogram=False)
# A.exclude_feature(['Bx', 'By', 'dBx', 'dBy'])
A.merge_feature(['Bx', 'By', 'Bz'], 1, ["B_pca"])
A.merge_feature(['dBx', 'dBy', 'dBz'], 1, ["dB_pca"])
A.merge_feature(['AU', 'AL', 'AE',], 1, ["A_pca"])


A.correlation_matrix(plot_correlation_matrix=True, plot_correlogram=False)







































end_time = time.time()
print(f"Time taken to run the line: {end_time - start_time:.6f} seconds")






















