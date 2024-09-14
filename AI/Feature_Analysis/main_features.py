# -*- coding: utf-8 -*-
"""
Created on Sat Sep 14 13:46:08 2024

@author: Kian Sartipzadeh
"""

import time
start_time = time.time()


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt




# Extracting the header
with open("spacephysics_19features.csv", "r") as f:
    header = f.readline().strip().split(",")

# Reading 19 space physics features
data = np.genfromtxt("spacephysics_19features.csv", delimiter=",", skip_header=0)













































end_time = time.time()
print(f"Time taken to run the line: {end_time - start_time:.6f} seconds")






















