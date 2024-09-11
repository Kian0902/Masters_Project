# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 13:02:05 2024

@author: Kian Sartipzadeh
"""


import numpy as np
import matplotlib.pyplot as plt
from lmfit import Model



class Curvefitting:
    """
    Class for choosing between 3 different curvefitting models:
    - SciPy
    - lmfit
    - torch
    """
    def __init__(self):
        
        self.curvefitting_model = {'scipy': self.curvefit_scipy,
                                   'lmfit': self.curvefit_lmfit,
                                   'NN': self.curvefit_neural_network}
        
    
    def double_chapman(self, z, HEd, HEu, HFd, HFu, zE_peak, neE_peak, zF_peak, neF_peak):
        HE = np.where(z <= zE_peak, HEd, HEu)
        HF = np.where(z <= zF_peak, HFd, HFu)
        
        neE = neE_peak * np.exp(1 - ((z - zE_peak)/HE) - np.exp(-((z - zE_peak)/HE)))
        neF = neF_peak * np.exp(1 - ((z - zF_peak)/HF) - np.exp(-((z - zF_peak)/HF)))
        return neE + neF
    
    
    def curvefit_scipy(self):
        ...
        
    def curvefit_lmfit(self):
        ...
    
    def curvefit_neural_network(self):
        ...
        
