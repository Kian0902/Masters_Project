# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 14:17:03 2024

@authors: Kian Sartipzadeh, Andreas Kvammen

This code was originally written by Andreas in matlab but converted into
python code by Kian.
"""





import os
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import scipy.io as sio




class EISCATDataProcessor:
    """
    Class to process semi_processed EISCAT radar data from .hdf5 files and save the
    processed data as .mat files for further analysis. The class handles data
    extraction, processing, visualization, and saving the results.
    """
    def __init__(self, folder_name_in: str, folder_name_out: str):
        """
        Initializes the Class with the specified input and output local folder
        dir names.
        
        Input (type)          | DESCRIPTION
        ------------------------------------------------
        folder_name_in  (str) | Name of local folder containing .hdf5 data files.
        folder_name_out (str) | Name of local folder to processed data and images will be stored.
        
        
        Attributes (type)   | DESCRIPTION
        ------------------------------------------------
        datapath   (str)    | Full path to the local folder containing input .hdf5 data files.
        resultpath (str)    | Full path to the local folder for storing output data (images and .mat files).
        datafiles  (list)   | List of .hdf5 file names in the input data directory.
        num_datafiles (int) | Number of .hdf5 files found in the input directory.
        """
        self.datapath   = os.path.abspath(os.path.join(os.getcwd(), folder_name_in))
        self.resultpath = os.path.abspath(os.path.join(os.getcwd(), folder_name_out))
        self.datafiles = self._find_data_files()
        self.num_datafiles = len(self.datafiles)
    
    
    
    def _find_data_files(self):
        """
        Finds and returns a list of all hdf5 files in the input directory.
        
        Return (type)      | DESCRIPTION
        ------------------------------------------------
        file_names (list)  | A list of filenames (str) of all .hfd5 files found in the input directory.
        """
        file_names = [f for f in os.listdir(self.datapath) if f.endswith('.hdf5')]
        return file_names
    
    
    
    def interpolate_nan(self, data):
        
        data = pd.DataFrame(data)
        
        data_interp = data.interpolate(method="linear", axis=0).interpolate(method="linear", axis=1)
        
        data_out = data_interp.to_numpy()
        
        return data_out
        
        

    def process_file(self, file_index):
        """
        Processes a specific .hdf5 file by its index, extracting relevant data,
        generating plots, and saving the processed data.
        
        Input (type)     | DESCRIPTION
        ------------------------------------------------
        file_index (int) | The index of the file to be processed in the `datafiles` list.
        
        
        Raises       | Condition
        ------------------------------------------------
        ValueError   | If 'file_index' is out of the range of available files.
        """
        
        
        if file_index < 0 or file_index >= self.num_datafiles:
            raise ValueError("File index out of range")
        
        file = self.datafiles[file_index]
        print(f'Processing file {file_index + 1}/{self.num_datafiles}: {file}\n')



        file_path = os.path.join(self.datapath, file)
        with h5py.File(file_path, 'r') as f:
            data = f['/Data/Table Layout']

            year = int(file[8:12])
            month = int(file[13:15])
            day = int(file[16:18])
            
            # Convert time data into datetime format
            t = [datetime(year, month, day, int(h), int(m), int(s))
                  for h, m, s in zip(data['hour'], data['min'], data['sec'])]
            
            
            range_data = data['range'][:]
            
            # Find indices where range data shows significant drops (indicating a new time step)
            ind = np.concatenate(([0], np.where(np.diff(range_data) < -500)[0] + 1))
            t_i = np.array(t)[ind]
            
            
            te = data['tr'][:] * data['ti'][:]  # combining electron and ion temperatures
            
            nT = len(ind)  # number of time-steps
            
            
            # Prepare indices for slicing
            ind_extended = np.concatenate((ind, [len(range_data)]))
            measure_lengths = np.diff(ind_extended)
            unique_values, counts = np.unique(measure_lengths, return_counts=True)  # find num of unique values  
            most_repeated_value = unique_values[np.argmax(counts)]  # get the most repeated value
            nZ = int(most_repeated_value)  # convert to int
            
            
                
            
            shape = (nT, nZ)
            if shape[0]*shape[1] != range_data.size:
                
                # Generate a mask for valid entries
                offsets = np.arange(nZ)
                valid_mask = offsets < measure_lengths[:, None]
                
                # print(valid_mask.shape)
                
                data_indices = ind[:, None] + offsets
                
                # Mask out indices that are beyond the actual data length
                valid_data_indices = data_indices[valid_mask]
                
                # print(valid_data_indices.shape)
                
                # Initialize matrices to store interpolated data
                Ne_i = np.full((nT, nZ), np.nan)     # Electron density
                DNe_i = np.full((nT, nZ), np.nan)    # Electron density error
                range_i = np.full((nT, nZ), np.nan)  # Range values
                Te_i = np.full((nT, nZ), np.nan)     # Electron temperature
                Ti_i = np.full((nT, nZ), np.nan)     # Ion temperature
                Vi_i = np.full((nT, nZ), np.nan)     # Ion velocity
                El_i = np.full((nT, nZ), np.nan)     # Elevation angle
                
                # Store data for this time step, truncating if necessary
                Ne_i[valid_mask] = data['ne'][valid_data_indices]
                DNe_i[valid_mask] = data['dne'][valid_data_indices]
                Vi_i[valid_mask] = data['vo'][valid_data_indices]
                Te_i[valid_mask] = te[valid_data_indices]
                Ti_i[valid_mask] = data['ti'][valid_data_indices]
                range_i[valid_mask] = range_data[valid_data_indices]
                
                # Store data for this time step, truncating if necessary
                Ne_i  = self.interpolate_nan(Ne_i)
                DNe_i = self.interpolate_nan(DNe_i)
                Vi_i  = self.interpolate_nan(Vi_i)
                Te_i  = self.interpolate_nan(Te_i)
                Ti_i  = self.interpolate_nan(Ti_i)
                range_i = self.interpolate_nan(range_i)
                
                
                
                self.plot_and_save_results(t_i, range_i, Ne_i, Te_i, Ti_i, Vi_i, year, month, day)
                self.save_mat_file(t_i, range_i, Ne_i, DNe_i, year, month, day)
                
            
            else:
                
                El_it = data['elm'].reshape(shape)
                elevation_mask = np.abs(np.mean(El_it, axis=1) - 90) < 20
                
                Ne_i = np.where(elevation_mask[:, None], data['ne'].reshape(shape), 0)
                DNe_i = np.where(elevation_mask[:, None], data['dne'].reshape(shape), 0)
                Vi_i = np.where(elevation_mask[:, None], data['vo'].reshape(shape), 0)
                Te_i = np.where(elevation_mask[:, None], te.reshape(shape), 0)
                Ti_i = np.where(elevation_mask[:, None], data['ti'].reshape(shape), 0)
                range_i = np.where(elevation_mask[:, None], range_data.reshape(shape), 0)
            
                self.plot_and_save_results(t_i, range_i, Ne_i, Te_i, Ti_i, Vi_i, year, month, day)
                self.save_mat_file(t_i, range_i, Ne_i, DNe_i, year, month, day)
            
            # else:
            #     El_it  = data['elm'].reshape(shape)
            #     Ne_i  = data['ne'].reshape(shape)
            #     DNe_i = data['dne'].reshape(shape)
            #     Vi_i  = data['vo'].reshape(shape)
            #     Te_i  = te.reshape(shape)
            #     Ti_i  = data['ti'].reshape(shape)
            #     range_i = range_data.reshape(shape)
                
            #     self.plot_and_save_results(t_i, range_i, Ne_i, Te_i, Ti_i, Vi_i, year, month, day)
            #     self.save_mat_file(t_i, range_i, Ne_i, DNe_i, year, month, day)
    
    def plot_and_save_results(self, t_i, range_i, Ne_i, Te_i, Ti_i, Vi_i, year, month, day):
        
        
        print(t_i.shape, range_i.shape, Ne_i.shape)
        
        plt.figure()
        
        # Electron density plot
        ax1 = plt.subplot(4, 1, 1)
        plt.pcolormesh(t_i, np.nanmean(range_i, axis=0), Ne_i.T, shading='auto')
        plt.colorbar()
        plt.clim(1e9, 5e11)
        
        # Electron temperature plot
        ax2 = plt.subplot(4, 1, 2)
        plt.pcolormesh(t_i, np.nanmean(range_i, axis=0), Te_i.T, shading='auto')
        plt.colorbar()
        plt.clim(500, 4000)
        
        # Ion temperature plot
        ax3 = plt.subplot(4, 1, 3)
        plt.pcolormesh(t_i, np.nanmean(range_i, axis=0), Ti_i.T, shading='auto')
        plt.colorbar()
        plt.clim(500, 3000)
        
        # Ion velocity plot
        ax4 = plt.subplot(4, 1, 4)
        plt.pcolormesh(t_i, np.nanmean(range_i, axis=0), Vi_i.T, shading='auto')
        plt.colorbar()
        plt.clim(-400, 400)
        
        # Set the colormap and link the axes of all plots
        plt.set_cmap('turbo')
        plt.subplots_adjust(hspace=0.5)
        plt.show()
        
        # # Save the figure as a .png file in the result folder
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
        for iE in range(self.num_datafiles):
            self.process_file(iE)
        print(f"Processing complete. Processed {self.num_datafiles} data files.")







# # Usage example:
# datapath = "Processing_inputs/beata_uhf_madrigal"
# resultpath = "Processing_outputs/Ne_uhf_madrigal"

# processor = EISCATDataProcessor(datapath, resultpath)

# # processor.process_file(17)

# processor.process_all_files()











