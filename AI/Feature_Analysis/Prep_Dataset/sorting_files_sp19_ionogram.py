# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 16:30:46 2024

@author: Kian Sartipzadeh
"""




import os
import glob
import pandas as pd



# custom_header=['DoY/366', 'ToD/1440', 'Solar_Zenith/44', 'Kp', 'R', 'Dst',
#                'ap', 'F10_7', 'AE', 'AL', 'AU', 'PC_potential', 'Lyman_alpha',
#                'Bx', 'By', 'Bz', 'dBx', 'dBy', 'dBz']


folder_path = "ionograms_1D"


df_list = []



for i, filename in enumerate(os.listdir(folder_path)):
    if filename.endswith('.csv'):
        if i % 1000 == 0:
            print(f'{i}/{len(os.listdir(folder_path))}')
        
        file_path = os.path.join(folder_path, filename)
        
        
        df = pd.read_csv(file_path, header=None, skiprows=1)
        
        df = df.T
        
        
        df.to_csv(f'SP19_Ionogram_samples/{filename}', index=False, header=False)
        
        
        
        
        # df_list.append(df)

# combined_df = pd.concat(df_list, axis=0)
# combined_df.to_csv('spacephysics_19features.csv', index=False, header=custom_header)

































































