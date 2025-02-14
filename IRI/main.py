# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 14:59:20 2025

@author: Kian Sartipzadeh
"""


import numpy as np
import matplotlib.pyplot as plt
from utils import save_dict, load_dict, inspect_dict
from IRI_processing import filter_range, interpolate_data
from IRI_plotting import IRIPlotter




# day = '2019-1-5'
# day = '2019-12-15'
# day = '2020-2-27'




X_EISCAT = load_dict("X_avg_test_data")
X_IRI = load_dict("X_IRI")


IRI_processed = {}

for day in X_EISCAT:
    X_iri = X_IRI[day]
    X_uhf = X_EISCAT[day]
    
    plotter = IRIPlotter(X_iri)
    
    
    
    # plotter.plot_profile()
    plotter.plot_profiles()
    # plotter.plot_day()
    
    
    X_filt = filter_range(X_iri, 'r_h', 90, 400)
    
    r_uhf = X_uhf["r_h"]
    X_inter = interpolate_data(X_filt, r_uhf)
    
    
    plotter.plot_before_vs_after(X_inter)
    
    IRI_processed[day] = X_inter
        

save_dict(IRI_processed, "X_IRI_interp")
inspect_dict(load_dict("X_IRI_interp"))
