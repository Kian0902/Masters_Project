# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 11:45:00 2024

@author: Kian Sartipzadeh
"""



import numpy as np
import matplotlib.pyplot as plt

from SALib.sample import saltelli
from SALib.analyze import sobol






def double_chapman(z, HEd, HEu, HFd, HFu, zE_peak, zF_peak, neE_peak, neF_peak):
    """
    Function for the double chapman electron density model.
    
    Input (type) | DESCRIPTION
    ------------------------------------------------
    z            | Altitude values.
    
    HEd          | E-reg lower scale-height.
    HEu          | E-reg upper scale-height.
    HFd          | F-reg lower scale-height.
    HFu          | F-reg upper scale-height.
    
    neE_peak     | E-region Peak electron density.
    neF_peak     | F-region Peak electron density.
    zE_peak      | E-region Peak altitude.
    zF_peak      | F-region Peak altitude.
    
    Return (type) | DESCRIPTION
    ------------------------------------------------
    neE + neF     | Double Chapman electron density.
    """
    HE = np.where(z <= zE_peak, HEd, HEu)   # E-reg scale-height
    HF = np.where(z <= zF_peak, HFd, HFu)   # F-reg scale-height
    
    neE = neE_peak * np.exp(1 - ((z - zE_peak)/HE) - np.exp(-((z - zE_peak)/HE)))
    neF = neF_peak * np.exp(1 - ((z - zF_peak)/HF) - np.exp(-((z - zF_peak)/HF)))
    
    ne = neE + neF
    
    return ne




def eval_model(z, parm):
    HEd, HEu, HFd, HFu, neE_peak, neF_peak, zE_peak, zF_peak = parm
    return double_chapman(z, HEd, HEu, HFd, HFu, zE_peak, zF_peak, neE_peak, neF_peak)



problem = {'num_vars': 8,
           'names': ['HEd','HEu','HFd', 'HFu',
                     'log_neE_peak', 'log_neF_peak',
                     'zE_peak','zF_peak'],
           
           'bounds': [[1, 100], [1, 100], [1, 100], [1, 100],
                      [1e5, 1e15], [1e5, 1e15],
                      [80, 600], [80, 600]]
           }





# generate parameter samples
param_values = saltelli.sample(problem, 2**8)


z = np.linspace(90, 400, 150)



# Evaluate the model for each sample
Y = np.array([eval_model(z, param) for param in param_values])


Si = [sobol.analyze(problem, Y[:, i]) for i in range(len(z))]

Si_T = np.array([Si[i]['ST'] for i in range(len(z))])
Si_T_coef = np.array([Si[i]['ST_conf'] for i in range(len(z))])


pars = ['HEd','HEu','HFd', 'HFu', 'neE_peak', 'neF_peak', 'zE_peak','zF_peak']




fig, ax = plt.subplots(figsize=(8, 6))
c = ax.pcolormesh(pars, z, Si_T, shading='auto', cmap="hot")

fig.colorbar(c, ax=ax)
ax.set_title('Total-Order Sensitivity Indices (ST)')
ax.set_xlabel('Parameters')
ax.set_ylabel('Altitude (km)')


plt.xticks(rotation=45, ha='center')
plt.show()
















