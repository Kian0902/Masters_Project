# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 13:02:05 2024

@author: Kian Sartipzadeh
"""


import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from lmfit import Model



class CurvefittingChapman:
    """
    Class for choosing between 3 different curvefitting models from:
    - SciPy
    - lmfit
    - PyTorch
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
    
    
    
    
    def get_curvefits(self, data: dict, model_name: str, save_plot: bool=False):
        
        # Value error 
        if model_name not in self.curvefitting_model:
            raise ValueError(f"Curvefitting model '{model_name}' not recognized.")
        
        
        
        y_peaks = self.get_peaks(data)
        
        y_fit = self.curvefitting_model[model_name]()  # Calling curvefitting model
        
        
        
    
    def get_peaks(self, data: dict):
        
        print("\nget_peaks")
        print("---------------------")
        
        print(data.keys())
        
        z = data['r_h'].flatten()
        print(type(z), z.shape)
        
        Ne = data['r_param'].T
        print(type(Ne), Ne.shape)
        print("\n")
        
        print("Inside Loop")
        print("---------------------")
        for i in range(0, len(Ne.T)):
            
            # Indexing time of measurement (averaged over 15 min)
            ne = Ne[i+11]
            
            print(ne.shape)
            
            
            plt.plot(ne, z)
            plt.xscale('log')
            plt.show()
            
            
            # Finding peak electron densities
            neE_peak = np.max(ne[(150 >= z) & (z > 100)])
            neF_peak = np.max(ne[(300 >= z) & (z > 150)])
            
            print(neE_peak)
            print(neF_peak)
            break
            
        # # True EISCAT data
        # Z_true = np.tile(np.array(data_day["r_h"]), 32).T        # altitude [km] 
        # Ne_true = np.array(data_day["r_param"]).T   # electron density []
        # print(Z_true.shape)
        # print(Ne_true.shape)
        
        # # Loop though time of day
        # for i in np.arange(0, len(Z_true)):
            
            
        #     # Indexing time of measurement (averaged over 15 min)
        #     z_true = Z_true[i]
        #     ne_true = Ne_true[i]
            
        #     # Finding peak electron densities
        #     neE_peak = np.max(ne_true[(150 >= z_true) & (z_true > 100)])
        #     neF_peak = np.max(ne_true[(300 >= z_true) & (z_true > 150)])
            
        #     # Finding altitude at peak
        #     zE_peak = z_true[ne_true==neE_peak]
        #     zF_peak = z_true[ne_true==neF_peak]
    
        
    
    
    def curvefit_scipy(self):
        
        print("scipy")
        
        
    def curvefit_lmfit(self):
        
        print("lmfit")
        
    
    def curvefit_neural_network(self):
        
        print("NN")
    


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




import pickle

# Importing processed dataset
custom_file_path = "Ne_vhf_avg"
with open(custom_file_path, 'rb') as f:
    dataset = pickle.load(f)



X = dataset['2018-11-10']


m = 'scipy'
# m = 'lmfit'
# m = 'NN'

A = CurvefittingChapman(X)
A.get_curvefits(X, m)



































