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
        
        z = data['r_h'].flatten()
        Ne = data['r_param'].T
        
        # print(Ne.shape)
        # print(Ne.T)
        
        for i in range(0, len(Ne)):
            
            # Indexing time of measurement (averaged over 15 min)
            ne = Ne[i]
            print(i)
            
            neE_peak = np.max(ne[(150 >= z) & (z > 100)])
            neF_peak = np.max(ne[(300 >= z) & (z > 150)])
            
            
            
            zE_peak = z[ne==neE_peak]
            zF_peak = z[ne==neF_peak]
            
            
            # Defining E and F-region altitudes
            e_reg = (150 >= z) & (90 < z)
            f_reg = (300 >= z) & (150 < z)
            
            # Finding E and F-region peaks
            e_peaks, e_properties = find_peaks(ne[e_reg], prominence=True)
            f_peaks, f_properties = find_peaks(ne[f_reg], prominence=True)
            
            # Handling E-region
            if e_peaks.size > 0:
                e_peak_index = e_properties['prominences'].argmax()
                e_peaks_z = z[e_reg][e_peaks][e_peak_index]
                e_peaks_ne = ne[e_reg][e_peaks][e_peak_index]
            else:
                e_peaks_z = z[e_reg][ne[e_reg].argmax()]
                e_peaks_ne = ne[e_reg].max()
            
            # Handling F-region
            if f_peaks.size > 0:
                f_peak_index = f_properties['prominences'].argmax()
                f_peaks_z = z[f_reg][f_peaks][f_peak_index]
                f_peaks_ne = ne[f_reg][f_peaks][f_peak_index]
            else:
                f_peaks_z = z[f_reg][ne[f_reg].argmax()]
                f_peaks_ne = ne[f_reg].max()
                    
                
            # print(len(e_peaks), len(f_peaks))
            # print(e_peaks_z, f_peaks_z)
            # print(e_properties['prominences'], f_properties['prominences'])
            # # print(e_peak_index, f_peak_index)
            # print("\n")
            
            plt.plot(ne, z, color='green', label="Ne")
            plt.scatter([e_peaks_ne, f_peaks_ne], [e_peaks_z, f_peaks_z], color="red", label="scipy")
            plt.scatter([neE_peak, neF_peak], [zE_peak, zF_peak], color="C0", label="normal", marker="x")
            plt.xscale('log')
            plt.legend()
            plt.show()
            
            
            
            
            
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



X = dataset['2018-11-9']


m = 'scipy'
# m = 'lmfit'
# m = 'NN'

A = CurvefittingChapman(X)
A.get_curvefits(X, m)



































