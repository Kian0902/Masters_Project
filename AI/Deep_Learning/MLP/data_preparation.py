# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 13:21:45 2024

@author: Kian Sartipzadeh
"""




import numpy as np


data_sp19 = np.load('sp19_data.npy')
data_eiscat = np.load('eiscat_data.npy')





print(np.amin(data_sp19), np.amax(data_sp19))
print(np.amin(data_eiscat), np.amax(data_eiscat))



data_eiscat[data_eiscat < 1e5] = 10**5



print(np.amin(data_sp19), np.amax(data_sp19))
print(np.amin(data_eiscat), np.amax(data_eiscat))



































