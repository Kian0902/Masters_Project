# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 11:45:37 2024

@author: Kian Sartipzadeh
"""



import numpy as np
import matplotlib.pyplot as plt
from lmfit import Model






def gaussian(x, amp, mean, sig):
    return amp * np.exp(-(x - mean) ** 2 / (2 * sig ** 2))







































