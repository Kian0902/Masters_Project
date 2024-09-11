# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 13:02:05 2024

@author: Kian Sartipzadeh
"""


import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from lmfit import Model



class CurvefittingChapman:
    """
    Class for choosing between 3 different curvefitting models:
    - SciPy
    - lmfit
    - torch
    """
    def __init__(self, dataset):
        """
        Initialize with dataset containing processed EISCAT data of one or
        multiple days.
        
        
        Attributes (type) | DESCRIPTION
        ------------------------------------------------
        dataset (dict)    | Global dictionary containing the EISCAT data.
        """
        
        self.dataset = dataset
        self.curvefitting_model = {'scipy': self.curvefit_scipy,
                                   'lmfit': self.curvefit_lmfit,
                                   'NN': self.curvefit_neural_network}
    
    
    
    def _double_chapman_wrapper(self, z, HEd, HEu, HFd, HFu, zE_peak, zF_peak, neE_peak, neF_peak):
        """
        Wrapper function for the double chapman.
        """
        return self.double_chapman(z, HEd, HEu, HFd, HFu, zE_peak, zF_peak, neE_peak, neF_peak)
    
    
    def double_chapman(self, z, HEd, HEu, HFd, HFu, zE_peak, neE_peak, zF_peak, neF_peak):
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
        return neE + neF
    
    
    def curvefit_scipy(self):
        
        print("scipy")
        
        
    def curvefit_lmfit(self):
        
        print("lmfit")
        
    
    def curvefit_neural_network(self):
        
        print("NN")
    
    
    
    def get_curvefits(self, data: dict, model_name: str, save_plot: bool=False):
        
        
        # Value error 
        if model_name not in self.curvefitting_model:
            raise ValueError(f"Curvefitting model '{model_name}' not recognized.")
        
        
        
        y_fit = self.curvefitting_model[model_name]()
        





A = CurvefittingChapman(1)
A.get_curvefits(2, "NN")







































