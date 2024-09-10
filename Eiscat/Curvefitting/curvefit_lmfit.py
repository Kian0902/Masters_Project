# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 11:45:37 2024

@author: Kian Sartipzadeh
"""



import numpy as np
import matplotlib.pyplot as plt
from lmfit import Model



# Defining function to be curvefitted
def gaussian(x, amp, mean, sig):
    return amp * np.exp(-(x - mean) ** 2 / (2 * sig ** 2))


true_amp  = 5.0
true_mean = 0.0
true_sig  = 1.0

x = np.linspace(-5, 5, 1000)

noise = np.random.normal(scale=0.2, size=x.size)

np.random.seed(42)
y = gaussian(x, true_amp, true_mean, true_sig) + noise


# Plot the synthetic data
plt.scatter(x, y, label='Noisy data', color='blue')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()




# LMFIT model

#Create fitting model
gaussian_model = Model(gaussian)


# Init params
params = gaussian_model.make_params(amp=4, mean=0, sig=2)


# Firt model to data
result = gaussian_model.fit(y, params, x=x)


print(result.fit_report())



# Plot the fitted curve
plt.scatter(x, y, label='Noisy data', color='blue')
plt.plot(x, result.best_fit, label='Best fit', color='red')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()





















