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

# A.sort_data()

# data = A.return_data()


# print(type(data))
# print(data.keys())
# print(data['2018-11-10'].keys())

# for key in data['2018-11-10']:
#     print(key)
#     print(type(data['2018-11-10'][key]))
#     print(data['2018-11-10'][key].shape)

# a = data['2018-11-10']

# z  =  a['r_h']
# ne =  a['r_param']


A.test_dataflow()





# plt.plot(ne[:, 0:12], z[:, 0:12])
# plt.show()






