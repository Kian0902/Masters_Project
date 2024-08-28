# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 14:13:45 2024

@author: Kian Sartipzadeh
"""


from data_sorting import EISCATDataSorter
from data_averaging import EISCATAverager
from data_outlier_detection import OutlierDetection
import matplotlib.pyplot as plt
import numpy as np




# Use the local folder name instead of the full path
folder_name = "Ne_vhf"
A = EISCATDataSorter(folder_name, filter_nan=True, filter_outliers=False, average_data=True)

A.sort_data(save_data=False)

data1 = A.return_data()




# for key in data1:
#     print(key)
#     data1 = A.return_data()[key]
#     for row in data1['r_param']:
#         # Check for NaN using numpy
#         if np.isnan(row.any()):  # This works for numerical data types (e.g., float)
#             print(f"NaN detected in {key}: {row}")





# data2 = A.return_data()['2018-11-11']
# data3 = A.return_data()['2018-12-1']
# data4 = A.return_data()['2022-12-19']
# data = A.test_dataflow(return_data=True)['2018-11-10']


# B = OutlierDetection(data1)
# B.detect_outliers('z-score', plot_outliers=True)
# B.detect_outliers('IQR', plot_outliers=True)





























# t  = data['2018-11-10']['r_time']
# z  = data['2018-11-10']['r_h']
# ne = data['2018-11-10']['r_param']




# plt.plot(ne[:, 164:165], z)
# plt.xscale('log')
# plt.show()








# print(t.shape, z.shape, ne.shape)

# tt = np.arange(len(t))
# print(tt.shape)




# plt.pcolormesh(tt, z, ne, shading='nearest',cmap='turbo')
# plt.clim(1e9, 5e11)
# plt.colorbar()
# plt.show()


# B = EISCATAverager(data['2018-11-10'])
# data_avg = B.average_over_period()


# t_avg  =  data_avg['r_time']
# z_avg  =  data_avg['r_h']
# ne_avg =  data_avg['r_param']


# tt_avg = np.arange(len(t_avg))

# plt.pcolormesh(tt_avg, z_avg, ne_avg, shading='nearest',cmap='turbo')
# plt.clim(1e9, 5e11)
# plt.colorbar()
# plt.show()








