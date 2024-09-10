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
        
    
    def curvefit_scipy(self):
        ...
        
    def curvefit_lmfit(self):
        ...
    
    def curvefit_neural_network(self):
        ...
        
