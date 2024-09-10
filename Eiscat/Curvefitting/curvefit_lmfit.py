# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 11:45:37 2024

@author: Kian Sartipzadeh
"""



import numpy as np
import matplotlib.pyplot as plt
from lmfit import Model



# Defining simple chapman model function
def chapman(z, z_peak, ne_peak, Hd, Hu):
    
    H = np.where(z <= z_peak, Hd, Hu)
    
    
    ne = ne_peak * np.exp(1 - ((z - z_peak)/H) - np.exp(-((z - z_peak)/H)))
    return ne



x = np.linspace(90, 400, 32)
y = chapman(x, z_peak=150, ne_peak=1e8, Hd=20, Hu=55)


# Plot the synthetic data
plt.plot(y, x, label='Chapman', color='C0')
plt.xlabel('Ne')
plt.ylabel('z')
plt.legend()
plt.show()






curvefit_model = Model(chapman)
params = curvefit_model.make_params(Hd=10, Hu=35)

params.add('z_peak', value=150, vary=False)
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










