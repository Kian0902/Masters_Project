# -*- coding: utf-8 -*-
"""
Created on Sat Sep 14 14:04:36 2024

@author: Kian Sartipzadeh
"""




import os
import glob
import pandas as pd




folder_path = "ionograms_1D"

all_files = glob.glob(os.path.join(folder_path, "*.csv"))


def sort():
    with open('ionograms1D.csv', 'w') as outfile:
        for i, file in enumerate(all_files):
            
            if i % 100 == 0:
                print(f'{i}/{len(all_files)}')
            
            df = pd.read_csv(file)
            if i==0:
                df.to_csv(outfile, index=False)
            else:
                df.to_csv(outfile, index=False, header=False)

















