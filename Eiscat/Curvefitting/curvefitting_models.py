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
        
        
        time = data['r_time']
        z    = data['r_h'].flatten()
        Ne   = data['r_param'].T
        
        
        for i in range(0, len(Ne)):
            
            # Indexing time of measurement (averaged over 15 min)
            ne = Ne[i]
            
            # Getting peaks
            e_peaks_z, e_peaks_ne, f_peaks_z, f_peaks_ne = self.get_peaks(z, ne)
            
            # Calling curvefitting model
            y_fit = self.curvefitting_model[model_name](z, ne, e_peaks_z, f_peaks_z, e_peaks_ne, f_peaks_ne)
            
            
            
            
            
            plt.plot(ne, z, color='C0', label="Ne")
            plt.plot(y_fit, z, color='C1', label="fitted")
            plt.scatter([e_peaks_ne, f_peaks_ne], [e_peaks_z, f_peaks_z], color="red", label="Peaks")
            plt.xscale('log')
            plt.legend()
            plt.show()
            
    
    
    
    def get_peaks(self, z, ne):
        """
        Function for finding E and F-region electron density peaks and their
        corresponding altitudes.
        
        Input (type)  | DESCRIPTION
        ------------------------------------------------
        z  (np.array) | Altitude values.
        ne (np.array) | Electron density values.
        
        Return (type) | DESCRIPTION
        ------------------------------------------------
        e_peaks_z     | E-region altitude peak.
        e_peaks_ne    | E-region electron density peak.
        f_peaks_z     | F-region altitude peak.
        f_peaks_ne    | F-region electron density peak.
        """
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
        
        return e_peaks_z, e_peaks_ne, f_peaks_z, f_peaks_ne
            
    
    
    def curvefit_scipy(self, z, ne, zE_peak, zF_peak, neE_peak, neF_peak):
        
        # Curve fitting
        popt, pcov = curve_fit(lambda z, HEd, HEu, HFd, HFu: self.double_chapman_wrapper(z, HEd, HEu, HFd, HFu, zE_peak, zF_peak, neE_peak, neF_peak), z, ne, p0=[20, 55, 45, 70], bounds=(5, [200, 200, 200, 200]), maxfev=10000)
        
        # Extracting fitted parameters
        HEd_fitted, HEu_fitted, HFd_fitted, HFu_fitted = popt

        y_fit = self.double_chapman_wrapper(z, HEd_fitted, HEu_fitted, HFd_fitted, HFu_fitted, zE_peak, zF_peak, neE_peak, neF_peak)
        return y_fit
    
    
    
    
    def curvefit_lmfit(self):
        curvefit_model = Model(self.double_chapman)
        # params = curvefit_model.make_params(HEd=10, HEu=35, HFd=25, HFu=40)

        # params.add('zE_peak', value=150, vary=True)
        # params.add('neE_peak', value=5e8, vary=True)
        # params.add('zF_peak', value=290, vary=True)
        # params.add('neF_peak', value=1e9, vary=True)


        # result = curvefit_model.fit(y, params, z=x)

        # y_fit = result.best_fit
        
        print("lmfit")
        
    
    def curvefit_neural_network(self):
        
        print("NN")
    


    def double_chapman_wrapper(self, z, HEd, HEu, HFd, HFu, zE_peak, zF_peak, neE_peak, neF_peak):
        """
        Wrapper function for the double chapman.
        """
        return self.double_chapman(z, HEd, HEu, HFd, HFu, zE_peak, zF_peak, neE_peak, neF_peak)
    
    
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
        return neE + neF




import pickle

# Importing processed dataset
custom_file_path = "Ne_vhf_avg"
with open(custom_file_path, 'rb') as f:
    dataset = pickle.load(f)



X = dataset['2018-11-10']


# m = 'scipy'
m = 'lmfit'
# m = 'NN'

A = CurvefittingChapman(X)
A.get_curvefits(X, m)



































