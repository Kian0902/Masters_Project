# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 14:17:03 2024

@authors: Kian Sartipzadeh, Andreas Kvammen

This code was originally written by Andreas in matlab but converted into
python code by Kian.
"""







# import os
# import h5py
# import numpy as np
# import matplotlib.pyplot as plt
# from datetime import datetime
# import scipy.io as sio



# # 1 - Define paths
# # Define the path to the data folder (where the input HDF5 files are stored)
# datapath = "C:\\Users\\kian0\\OneDrive\\Desktop\\UiT Courses\\FYS_3931\\Scripts\\Masters_Project\\Eiscat\\Processing\\beata_vhf"
# # Define the path to the results folder (where the output will be saved)
# resultpath = "C:\\Users\\kian0\\OneDrive\\Desktop\\UiT Courses\\FYS_3931\\Scripts\\Masters_Project\\Eiscat\\Processing\\Ne_new"

# # 2 - Find datafiles
# # Change the current directory to the data folder
# os.chdir(datapath)
# # Get a list of all HDF5 files in the directory
# datafiles = [f for f in os.listdir(datapath) if f.endswith('.hdf5')]

# # 3 - Read files

# # Get the total number of experiments/files
# nE = len(datafiles)

# # Loop through each experiment (file)
# for iE in range(0, 1):
#     print(f'File {iE+1}/{nE}')

#     os.chdir(datapath)
#     file = datafiles[iE]
    
#     with h5py.File(file, 'r') as f:
#         data = f['/Data/Table Layout']



#         # Extract year, month, and day from the file name
#         year = int(file[8:12])
#         yy = np.full(len(data['ne']), year)

#         month = int(file[13:15])
#         mm = np.full(len(data['ne']), month)

#         day = int(file[16:18])
#         dd = np.full(len(data['ne']), 2)  # Placeholder for day

#         # Convert time data (year, month, day, hour, min, sec) into datetime format
#         t = [datetime(year, month, day, int(h), int(m), int(s))
#              for h, m, s in zip(data['hour'], data['min'], data['sec'])]

#         # Extract the range information
#         range_data = data['range'][:]
        
#         # Find indices where range data shows significant drops (indicating a new time step)
#         ind = np.concatenate(([0], np.where(np.diff(range_data) < -500)[0] + 1))
        
        
#         # Get the time steps corresponding to these indices
#         t_i = np.array(t)[ind]
        
        
        
        
#         # Combine electron and ion temperatures
#         te = data['tr'][:] * data['ti'][:]
        
        
        
        
#         # Number of time-steps
#         nT = len(ind)
#         unique_values, counts = np.unique(np.diff(ind), return_counts=True)
#         max_index = np.argmax(counts)
        
#         most_repeated_value = unique_values[max_index]
        
        
#         nZ = int(most_repeated_value)
        
        
        
        
#         # Initialize matrices to store interpolated data
#         Ne_i = np.full((nT, nZ), np.nan)  # Electron density
#         DNe_i = np.full((nT, nZ), np.nan)  # Electron density error
#         range_i = np.full((nT, nZ), np.nan)  # Range values
#         Te_i = np.full((nT, nZ), np.nan)  # Electron temperature
#         Ti_i = np.full((nT, nZ), np.nan)  # Ion temperature
#         Vi_i = np.full((nT, nZ), np.nan)  # Ion velocity
#         El_i = np.full((nT, nZ), np.nan)  # Elevation angle

#         # Loop over time steps to process the data
#         for iT in range(nT - 1):
#             # Define the start and end indices for each time step
#             ind_s = ind[iT]
#             ind_f = ind[iT + 1]
#             # Extract elevation angle for this time range
#             El_iT = data['elm'][ind_s:ind_f]

#             # Check if the mean elevation is close to 90 degrees (indicating data is usable)
#             if round(abs(np.mean(El_iT) - 90)) < 6:
#                 # Store data for this time step
#                 Ne_i[iT, :len(data['ne'][ind_s:ind_f])] = data['ne'][ind_s:ind_f]
#                 DNe_i[iT, :len(data['dne'][ind_s:ind_f])] = data['dne'][ind_s:ind_f]
#                 Vi_i[iT, :len(data['vo'][ind_s:ind_f])] = data['vo'][ind_s:ind_f]
#                 Te_i[iT, :len(te[ind_s:ind_f])] = te[ind_s:ind_f]
#                 Ti_i[iT, :len(data['ti'][ind_s:ind_f])] = data['ti'][ind_s:ind_f]
#                 range_i[iT, :len(data['ne'][ind_s:ind_f])] = range_data[ind_s:ind_f]

#         # If the elevation angle is still valid after the loop
#         if round(abs(np.mean(El_iT) - 90)) < 6:
#             # Plot the results for visualization
#             plt.figure()

#             # Electron density plot
#             ax1 = plt.subplot(4, 1, 1)
#             plt.pcolor(t_i, np.nanmean(range_i, axis=0), Ne_i.T, shading='auto')
#             plt.colorbar()
#             plt.clim(1e9, 5e11)  # Set color axis limits

#             # Electron temperature plot
#             ax2 = plt.subplot(4, 1, 2)
#             plt.pcolor(t_i, np.nanmean(range_i, axis=0), Te_i.T, shading='auto')
#             plt.colorbar()
#             plt.clim(500, 4000)  # Set color axis limits

#             # Ion temperature plot
#             ax3 = plt.subplot(4, 1, 3)
#             plt.pcolor(t_i, np.nanmean(range_i, axis=0), Ti_i.T, shading='auto')
#             plt.colorbar()
#             plt.clim(500, 3000)  # Set color axis limits

#             # Ion velocity plot
#             ax4 = plt.subplot(4, 1, 4)
#             plt.pcolor(t_i, np.nanmean(range_i, axis=0), Vi_i.T, shading='auto')
#             plt.colorbar()
#             plt.clim(-400, 400)  # Set color axis limits

#             # Set the colormap and link the axes of all plots
#             plt.set_cmap('turbo')
#             plt.subplots_adjust(hspace=0.5)
#             # plt.show()

#             # # Save the figure as a .png file in the result folder
#             os.chdir(resultpath)
#             name = f"{year}-{month}-{day}.png"
#             plt.savefig(name)
#             plt.close()
            
#             # Convert the datetime array into a matrix of date/time components
#             datetime_matrix = np.array([[dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second] for dt in t_i])

#             # Assign the final values for saving
#             r_time = datetime_matrix  # Time values
#             r_h = np.nanmean(range_i, axis=0)  # Range values
#             r_param = Ne_i  # Electron density
#             r_error = DNe_i  # Error in electron density

#             # Save the results as a .mat file
#             name = f"{year}-{month}-{day}.mat"
#             sio.savemat(name, {'r_time': r_time, 'r_h': r_h, 'r_param': r_param, 'r_error': r_error})







import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import scipy.io as sio

class EiscatDataProcessor:
    def __init__(self, datapath, resultpath):
        self.datapath = datapath
        self.resultpath = resultpath
        self.datafiles = self.find_data_files()
        self.nE = len(self.datafiles)

    def find_data_files(self):
        os.chdir(self.datapath)
        return [f for f in os.listdir(self.datapath) if f.endswith('.hdf5')]

    def process_file(self, file_index):
        if file_index < 0 or file_index >= self.nE:
            raise ValueError("File index out of range")
        
        file = self.datafiles[file_index]
        print(f'Processing file {file_index + 1}/{self.nE}: {file}')

        os.chdir(self.datapath)
        with h5py.File(file, 'r') as f:
            data = f['/Data/Table Layout']
            
            year = int(file[8:12])
            month = int(file[13:15])
            day = int(file[16:18])

            t = [datetime(year, month, day, int(h), int(m), int(s))
                 for h, m, s in zip(data['hour'], data['min'], data['sec'])]
            range_data = data['range'][:]
            ind = np.concatenate(([0], np.where(np.diff(range_data) < -500)[0] + 1))
            t_i = np.array(t)[ind]
            te = data['tr'][:] * data['ti'][:]
            nT = len(ind)
            unique_values, counts = np.unique(np.diff(ind), return_counts=True)
            most_repeated_value = unique_values[np.argmax(counts)]
            nZ = int(most_repeated_value)

            Ne_i = np.full((nT, nZ), np.nan)
            DNe_i = np.full((nT, nZ), np.nan)
            range_i = np.full((nT, nZ), np.nan)
            Te_i = np.full((nT, nZ), np.nan)
            Ti_i = np.full((nT, nZ), np.nan)
            Vi_i = np.full((nT, nZ), np.nan)
            El_i = np.full((nT, nZ), np.nan)

            for iT in range(nT - 1):
                ind_s = ind[iT]
                ind_f = ind[iT + 1]
                El_iT = data['elm'][ind_s:ind_f]

                if round(abs(np.mean(El_iT) - 90)) < 6:
                    Ne_i[iT, :len(data['ne'][ind_s:ind_f])] = data['ne'][ind_s:ind_f]
                    DNe_i[iT, :len(data['dne'][ind_s:ind_f])] = data['dne'][ind_s:ind_f]
                    Vi_i[iT, :len(data['vo'][ind_s:ind_f])] = data['vo'][ind_s:ind_f]
                    Te_i[iT, :len(te[ind_s:ind_f])] = te[ind_s:ind_f]
                    Ti_i[iT, :len(data['ti'][ind_s:ind_f])] = data['ti'][ind_s:ind_f]
                    range_i[iT, :len(data['ne'][ind_s:ind_f])] = range_data[ind_s:ind_f]

            if round(abs(np.mean(El_iT) - 90)) < 6:
                self.plot_and_save_results(t_i, range_i, Ne_i, Te_i, Ti_i, Vi_i, year, month, day)
                self.save_mat_file(t_i, range_i, Ne_i, DNe_i, year, month, day)

    def plot_and_save_results(self, t_i, range_i, Ne_i, Te_i, Ti_i, Vi_i, year, month, day):
        plt.figure()

        ax1 = plt.subplot(4, 1, 1)
        plt.pcolor(t_i, np.nanmean(range_i, axis=0), Ne_i.T, shading='auto')
        plt.colorbar()
        plt.clim(1e9, 5e11)

        ax2 = plt.subplot(4, 1, 2)
        plt.pcolor(t_i, np.nanmean(range_i, axis=0), Te_i.T, shading='auto')
        plt.colorbar()
        plt.clim(500, 4000)

        ax3 = plt.subplot(4, 1, 3)
        plt.pcolor(t_i, np.nanmean(range_i, axis=0), Ti_i.T, shading='auto')
        plt.colorbar()
        plt.clim(500, 3000)

        ax4 = plt.subplot(4, 1, 4)
        plt.pcolor(t_i, np.nanmean(range_i, axis=0), Vi_i.T, shading='auto')
        plt.colorbar()
        plt.clim(-400, 400)

        plt.set_cmap('turbo')
        plt.subplots_adjust(hspace=0.5)

        os.chdir(self.resultpath)
        name = f"{year}-{month}-{day}.png"
        plt.savefig(name)
        plt.close()

    def save_mat_file(self, t_i, range_i, Ne_i, DNe_i, year, month, day):
        datetime_matrix = np.array([[dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second] for dt in t_i])
        r_time = datetime_matrix
        r_h = np.nanmean(range_i, axis=0)
        r_param = Ne_i
        r_error = DNe_i

        name = f"{year}-{month}-{day}.mat"
        sio.savemat(os.path.join(self.resultpath, name), {'r_time': r_time, 'r_h': r_h, 'r_param': r_param, 'r_error': r_error})

    def process_all_files(self):
        for iE in range(1):
            self.process_file(iE)

# Usage example:
datapath = "C:\\Users\\kian0\\OneDrive\\Desktop\\UiT Courses\\FYS_3931\\Scripts\\Masters_Project\\Eiscat\\Processing\\beata_vhf"
resultpath = "C:\\Users\\kian0\\OneDrive\\Desktop\\UiT Courses\\FYS_3931\\Scripts\\Masters_Project\\Eiscat\\Processing\\Ne_new"

processor = EiscatDataProcessor(datapath, resultpath)
processor.process_all_files()


















