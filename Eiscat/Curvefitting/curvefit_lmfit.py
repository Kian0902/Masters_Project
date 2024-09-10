# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 11:45:37 2024

@author: Kian Sartipzadeh
"""



import numpy as np
import matplotlib.pyplot as plt
from lmfit import Model



# Defining simple chapman model function
def chapman(z, z_peak, ne_peak, H):
    
    y = (z - z_peak)/H
    ne = ne_peak * np.exp(1 - y - np.exp(-y))
    return ne



x = np.linspace(90, 300, 22)
y = chapman(x, z_peak=130, ne_peak=1e8, H=25)


# Plot the synthetic data
plt.plot(y, x, label='Chapman', color='C0')
plt.xlabel('Ne')
plt.ylabel('z')
plt.legend()
plt.show()






curvefit_model = Model(chapman)
params = curvefit_model.make_params(H=10)

params.add('z_peak', value=130, vary=False)
params.add('ne_peak', value=1e8, vary=False)

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










