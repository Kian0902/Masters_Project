# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 14:13:45 2024

@author: Kian Sartipzadeh
"""


from sorting_data import EISCATDataSorter
import matplotlib.pyplot as plt

# Use the local folder name instead of the full path
folder_name = "Ne"
A = EISCATDataSorter(folder_name)

A.sort_data()

data = A.return_data()

a = data['2018-11-10']

z  =  a['r_h']
ne =  a['r_param']


plt.plot(ne[:, 0:6], z[:, 0:6])
plt.show()



# A.test_dataflow()