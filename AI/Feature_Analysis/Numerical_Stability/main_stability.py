# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 11:45:00 2024

@author: Kian Sartipzadeh
"""



from SALib.sample import saltelli
from SALib.analyze import sobol
import numpy as np





def double_chapman(self, z, HEd, HEu, HFd, HFu, zE_peak, zF_peak, neE_peak, neF_peak):
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




problem = {'num_vars': 8,
           'names': ['HEd',
                     'HEu',
                     'HFd', 
                     'HFu',
                     'log_neE_peak', 
                     'log_neF_peak',
                     'zE_peak',
                     'zF_peak'],
           
           'bounds': [[1, 100],
                      [1, 100],
                      [1, 100],
                      [1, 100],
                      [5, 16],
                      [5, 16],
                      [80, 600],
                      [80, 600]]
           }






















