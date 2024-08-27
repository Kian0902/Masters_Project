# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 14:13:45 2024

@author: Kian Sartipzadeh
"""


from sorting_data import EISCATDataSorter
from average_data import EISCATAverager
import matplotlib.pyplot as plt
import numpy as np
# Use the local folder name instead of the full path
folder_name = "Ne_vhf"
A = EISCATDataSorter(folder_name)

# A.sort_data(save_data=True, save_filename="Ne_vhf_ready_curvefit")

# data = A.return_data()


# print(type(data))
# print(data.keys())
# print(data['2018-11-10'].keys())

# for key in data['2018-11-10']:
#     print(key)
#     print(type(data['2018-11-10'][key]))
#     print(data['2018-11-10'][key].shape)

# a = data['2018-11-10']



data = A.test_dataflow(return_data=True)

t  = data['2018-11-10']['r_time']
z  = data['2018-11-10']['r_h'].flatten()
ne = data['2018-11-10']['r_param']

print(t.shape, z.shape, ne.shape)

tt = np.arange(len(t))
print(tt.shape)




plt.pcolormesh(tt, z, ne, shading='nearest',cmap='turbo')
plt.clim(1e9, 5e11)
plt.colorbar()
plt.show()


B = EISCATAverager(data['2018-11-10'])
data_avg = B.average_over_period()


t_avg  =  data_avg['r_time']
z_avg  =  data_avg['r_h']
ne_avg =  data_avg['r_param']


tt_avg = np.arange(len(t_avg))

plt.pcolormesh(tt_avg, z_avg, ne_avg, shading='nearest',cmap='turbo')
plt.clim(1e9, 5e11)
plt.colorbar()
plt.show()

# plt.plot(ne[:, 10:13], z)
# plt.xscale('log')
# plt.show()






