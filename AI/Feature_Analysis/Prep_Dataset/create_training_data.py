# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 13:21:16 2024

@author: Kian Sartipzadeh
"""




import os
import pandas as pd



def list_csv_files(folder_path):
    
    L = []
    for f in os.listdir(folder_path):
        if f.endswith('.csv'):
            L.append(f)
        else:
            pass
    
    return L















