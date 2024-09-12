# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 13:02:05 2024

@author: Kian Sartipzadeh
"""


import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

from scipy.optimize import curve_fit
from lmfit import Model
from curvefitting_NN import CurvefitNN

import torch
import torch.nn as nn





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
    
    
    def batch_processing(self, method_name: str, save_plot=False):
        
        for key in list(self.dataset.keys()):
            print(key)
            
            # self.dataset_outliers[key] = self.detect_outliers(self.dataset[key], method_name=method_name, save_plot=save_plot)
            
            
    
    
    def get_curvefits(self, data: dict, model_name: str, H_initial: list, save_plot: bool=False):
        
        
        time = data['r_time']
        z    = data['r_h'].flatten()
        Ne   = data['r_param'].T
        
        
        for i in range(0, len(Ne)):
            
            # Indexing time of measurement (averaged over 15 min)
            ne = Ne[i]
            
            # Getting peaks
            e_peaks_z, e_peaks_ne, f_peaks_z, f_peaks_ne = self.get_peaks(z, ne)
            
            # Calling curvefitting model
            y_fit = self.curvefitting_model[model_name](z, ne, e_peaks_z, f_peaks_z, e_peaks_ne, f_peaks_ne, H_initial)
            
            
            
            
            if save_plot is True:
                plt.plot(ne, z, color='C0', label="Ne")
                plt.plot(y_fit, z, color='C1', label="fitted")
                plt.scatter([e_peaks_ne, f_peaks_ne], [e_peaks_z, f_peaks_z], color="red", label="Peaks")
                plt.xscale('log')
                plt.legend()
                plt.show()
            
            # if i == 10:
            #     break
    
    
    def get_peaks(self, z, ne):
        """
        Function for finding E and F-region electron density peaks and their
        corresponding altitudes. This method uses SciPy's signal library to
        find the most common peaks.
        
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
            
    
    
    def curvefit_scipy(self, z, ne, zE_peak, zF_peak, neE_peak, neF_peak, H_initial):
        """
        Function that use SciPy's curvefitting model.
        
        Input (type)  | DESCRIPTION
        ------------------------------------------------
        z  (np.array) | Altitude values.
        ne (np.array) | Electron density values.
        neE_peak      | E-region Peak electron density.
        neF_peak      | F-region Peak electron density.
        zE_peak       | E-region Peak altitude.
        zF_peak       | F-region Peak altitude.
        H_initial     | List containing initial guesses
        
        Return (type) | DESCRIPTION
        ------------------------------------------------
        ne_fit        | Curvefitted Double Chapman electron density.
        """
        
        # Initial guesses of scale-heights
        HEd_initial = H_initial[0]
        HEu_initial = H_initial[1]
        HFd_initial = H_initial[2]
        HFu_initial = H_initial[3]
        
        
        
        # Calling wrapper function
        func = lambda z, HEd, HEu, HFd, HFu: self.double_chapman_wrapper(z, HEd, HEu, HFd, HFu, zE_peak, zF_peak, neE_peak, neF_peak)
        
        # Curve fitting
        popt, pcov = curve_fit(func, z, ne, p0=[HEd_initial, HEu_initial, HFd_initial, HFu_initial], bounds=(5, [200, 200, 200, 200]), maxfev=10000)
        
        # Extracting fitted parameters
        HEd_fitted, HEu_fitted, HFd_fitted, HFu_fitted = popt
        
        # Calculating fitted double chapman curve
        ne_fit = self.double_chapman_wrapper(z, HEd_fitted, HEu_fitted, HFd_fitted, HFu_fitted, zE_peak, zF_peak, neE_peak, neF_peak)
        return ne_fit
    
    
    
    
    def curvefit_lmfit(self, z, ne, zE_peak, zF_peak, neE_peak, neF_peak, H_initial):
        """
        Function that use SciPy's curvefitting model.
        
        Input (type)  | DESCRIPTION
        ------------------------------------------------
        z  (np.array) | Altitude values.
        ne (np.array) | Electron density values.
        neE_peak      | E-region Peak electron density.
        neF_peak      | F-region Peak electron density.
        zE_peak       | E-region Peak altitude.
        zF_peak       | F-region Peak altitude.
        H_initial     | List containing initial guesses
        
        Return (type) | DESCRIPTION
        ------------------------------------------------
        ne_fit        | Curvefitted Double Chapman electron density.
        """
        
        # Initial guesses of scale-heights
        HEd_initial = H_initial[0]
        HEu_initial = H_initial[1]
        HFd_initial = H_initial[2]
        HFu_initial = H_initial[3]
        
        # Defining curvefitting model
        curvefit_model = Model(self.double_chapman)
        params = curvefit_model.make_params(HEd=HEd_initial, HEu=HEu_initial, HFd=HFd_initial, HFu=HFu_initial)
        
        # Adding peak values as constant values
        params.add('zE_peak', value=zE_peak, vary=True, min=50, max=600)
        params.add('zF_peak', value=zF_peak, vary=True, min=50, max=600)
        params.add('neE_peak', value=neE_peak, vary=True, min=1e3, max=1e16)
        params.add('neF_peak', value=neF_peak, vary=True, min=1e3, max=1e16)
        
        # Performing curvefitting
        result = curvefit_model.fit(ne, params, z=z)
        
        # Getting best results
        ne_fit = result.best_fit
        return ne_fit
        
    
    def curvefit_neural_network(self, z, ne, zE_peak, zF_peak, neE_peak, neF_peak, H_initial):
        
        
        # Model, criterion, and optimizer
        model = CurvefitNN(H_initial)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        
        # Converting data to PyTorch Tensors
        z = torch.from_numpy(z).double().view(-1, 1)
        ne = torch.from_numpy(ne).double().view(-1, 1)
        
        # Training loop
        for epoch in range(500):
            
            # Prediction
            output = model(z, zE_peak, zF_peak, neE_peak, neF_peak)
            
            # Calculate loss
            loss   = criterion(output, ne)
            optimizer.zero_grad()
            
            # Backpropagation
            loss.backward()
            
            # Updating weights (Scale-Heights)
            optimizer.step()


        with torch.no_grad():
            pred = model(z, zE_peak, zF_peak, neE_peak, neF_peak)
        
        
        # Converting torch.tensor to numpy.ndarray
        ne_fit = pred.numpy().flatten()
        return ne_fit
        

    


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
# m = 'lmfit'
m = 'NN'

A = CurvefittingChapman(X)
A.get_curvefits(X, m, H_initial=[20, 30, 35, 40], save_plot=True)
# A.batch_processing(m)


































