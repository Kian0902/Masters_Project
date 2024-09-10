# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 11:45:37 2024

@author: Kian Sartipzadeh
"""



import numpy as np
import matplotlib.pyplot as plt
from lmfit import Model
from scipy.optimize import curve_fit

# Defining the double Chapman model function
def double_chapman(z, HEd, HEu, HFd, HFu, zE_peak, zF_peak, neE_peak, neF_peak):
    HE = np.where(z <= zE_peak, HEd, HEu)
    HF = np.where(z <= zF_peak, HFd, HFu)
    
    neE = neE_peak * np.exp(1 - ((z - zE_peak)/HE) - np.exp(-((z - zE_peak)/HE)))
    neF = neF_peak * np.exp(1 - ((z - zF_peak)/HF) - np.exp(-((z - zF_peak)/HF)))
    return neE + neF

# Wrapper function for curve fitting
def fit_double_chapman(z, HEd, HEu, HFd, HFu, zE_peak, zF_peak, neE_peak, neF_peak):
    return double_chapman(z, HEd, HEu, HFd, HFu, zE_peak, zF_peak, neE_peak, neF_peak)

# Example data
x = np.linspace(90, 450, 38)
y = double_chapman(x, HEd=20, HEu=55, HFd=45, HFu=70, zE_peak=130, zF_peak=290, neE_peak=5e8, neF_peak=1e9) + np.random.normal(scale=1e8, size=x.size)

# Constants
zE_peak = 130
zF_peak = 290
neE_peak = 5e8
neF_peak = 1e9

# Curve fitting
popt, pcov = curve_fit(lambda z, HEd, HEu, HFd, HFu: fit_double_chapman(z, HEd, HEu, HFd, HFu, zE_peak, zF_peak, neE_peak, neF_peak), x, y, p0=[20, 55, 45, 70])

# Extracting fitted parameters
HEd_fitted, HEu_fitted, HFd_fitted, HFu_fitted = popt

y_fit = fit_double_chapman(x, HEd_fitted, HEu_fitted, HFd_fitted, HFu_fitted, zE_peak, zF_peak, neE_peak, neF_peak)

# Plot the synthetic data
plt.plot(y, x, label='Chapman', color='C0')
plt.plot(y_fit, x, label='Curvefit', color='C1')
plt.xlabel('Ne')
plt.ylabel('z')
plt.legend()
plt.show()






curvefit_model = Model(double_chapman)
params = curvefit_model.make_params(HEd=10, HEu=35, HFd=25, HFu=40)

params.add('zE_peak', value=150, vary=True)
params.add('neE_peak', value=5e8, vary=True)
params.add('zF_peak', value=290, vary=True)
params.add('neF_peak', value=1e9, vary=True)


result = curvefit_model.fit(y, params, z=x)

y_fit = result.best_fit

print(result.fit_report())




# Plot the synthetic data
plt.plot(y, x, label='Chapman', color='C0')
plt.plot(y_fit, x, label='Curvefit', color='C1')
plt.xlabel('Ne')
plt.ylabel('z')
plt.legend()
plt.show()










