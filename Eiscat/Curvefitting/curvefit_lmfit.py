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


true_amp  = 5.0
true_mean = 0.0
true_sig  = 1.0

x = np.linspace(-5, 5, 1000)

noise = np.random.normal(scale=0.2, size=x.size)

y = gaussian(x, true_amp, true_mean, true_sig) + noise


# Plot the synthetic data
plt.scatter(x, y, label='Noisy data', color='blue')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()

























