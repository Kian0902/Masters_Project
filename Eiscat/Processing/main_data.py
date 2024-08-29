# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 14:13:45 2024

@author: Kian Sartipzadeh
"""


import numpy as np
import matplotlib.pyplot as plt

from read_EISCAT_data import EISCATDataProcessor
from data_sorting import EISCATDataSorter
from data_averaging import EISCATAverager




def detect_nan_in_arrays(data_dict):
    for key, array in data_dict.items():
        if np.isnan(array).any():
            print(f"NaN detected {key}: {array.shape}")


# Use the local folder name containing data
folder_name = "Ne_vhf"


# Initialize data sorting for VHF
VHF = EISCATDataSorter(folder_name)
# VHF.sort_data()  # sort data

# X_vhf = VHF.return_data()  # returning dict data

VHF.test_dataflow()


# for day in X_vhf:
    # detect_nan_in_arrays(X_vhf[day])





















# guisdap_data_folder_name = "beata_vhf"
# result_folder_name = "Ne_vhf"


# process_data = EISCATDataProcessor(guisdap_data_folder_name, result_folder_name)
# process_data.process_all_files()












































# A = EISCATDataSorter(folder_name, filter_nan=True, filter_outliers=False, average_data=False)
# B = EISCATDataSorter(folder_name, filter_nan=True, filter_outliers=False, average_data=True)



# A.sort_data(save_data=False)
# B.sort_data(save_data=False)


# dataA = A.return_data()
# dataB = B.return_data()






# for i, day in enumerate(dataA.keys()):
#     # print(day, i)
#     # print(dataA[day].keys())
#     # print(dataB[day].keys())
#     print('\n')
#     print(day, i)
    
#     tA  = dataA[day]['r_time']
#     zA  = dataA[day]['r_h']
#     neA = dataA[day]['r_param']
#     ttA = np.arange(len(tA))
    
#     print(neA.shape)
    
#     tB = dataB[day]['r_time']
#     zB  = dataB[day]['r_h']
#     neB = dataB[day]['r_param']
#     ttB = np.arange(len(tB))
#     print(neB.shape)
    
#     fig, ax = plt.subplots(1,2)
    
#     a = ax[0].pcolormesh(ttA, zA, neA, shading='nearest', cmap='turbo')
#     a.set_clim(1e9, 5e11)
    
#     b = ax[1].pcolormesh(ttB, zB, neB, shading='nearest', cmap='turbo')
#     b.set_clim(1e9, 5e11)
    
#     plt.show()

    
    # t  = data['2018-11-10']['r_time']
    
    
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








