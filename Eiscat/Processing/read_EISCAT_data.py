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
import matplotlib.pyplot as plt
from datetime import datetime
import scipy.io as sio






class EISCATDataProcessor:
    """
    Class to process raw EISCAT radar data from .hdf5 files and save the
    processed data as .mat files for further analysis. The class handles data
    extraction, processing, visualization, and saving the results.
    
    Attributes (type)  | DESCRIPTION
    ------------------------------------------------
    datapath   (str)    | Full path to the local folder containing input .hdf5 data files.
    resultpath (str)    | Full path to the local folder for storing output data (images and .mat files).
    datafiles  (list)   | List of .hdf5 file names in the input data directory.
    num_datafiles (int) | Number of .hdf5 files found in the input directory.
    """
    def __init__(self, folder_name_in: str, folder_name_out: str):
        """
        Initializes the Class with the specified input and output local folder
        dir names.
        
        
        Input (type)         | DESCRIPTION
        ------------------------------------------------
        folder_name_in  (str) | Name of local folder containing .hdf5 data files.
        folder_name_out (str) | Name of local folder where the processed data and images will be stored.
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
        # os.chdir(self.datapath)
        file_names = [f for f in os.listdir(self.datapath) if f.endswith('.hdf5')]
        return file_names
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
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
        print(f'Processing file {file_index + 1}/{self.num_datafiles}: {file}')



        file_path = os.path.join(self.datapath, file)
        with h5py.File(file_path, 'r') as f:
            data = f['/Data/Table Layout']
            # print("Shape:", data.shape)
            # print("Data Type:", data.dtype)
            
            
            # Extract year, month, and day from the file name
            year = int(file[8:12])
            month = int(file[13:15])
            day = int(file[16:18])
            
            
            # Convert time data (year, month, day, hour, min, sec) into datetime format
            t = [datetime(year, month, day, int(h), int(m), int(s))
                 for h, m, s in zip(data['hour'], data['min'], data['sec'])]
            
            
            # Extract the range information
            range_data = data['range'][:]
            
            # print(range_data[-36:-1])
            
            # Find indices where range data shows significant drops (indicating a new time step)
            ind = np.concatenate(([0], np.where(np.diff(range_data) < -500)[0] + 1))
            
            
            # Get the time steps corresponding to these indices
            t_i = np.array(t)[ind]
            
            # Combine electron and ion temperatures
            te = data['tr'][:] * data['ti'][:]
            
            
            # Number of time-steps
            nT = len(ind)
            

            unique_values, counts = np.unique(np.diff(ind), return_counts=True)  # find num of unique values  
            most_repeated_value = unique_values[np.argmax(counts)]  # find the largest unique value
            nZ = int(most_repeated_value)  # convert to int
            
            
            
            shape = (nT, nZ)
            
            if shape[0]*shape[1] != range_data.size:
                
                n_rows = nT
                n_cols = data['range'].size // n_rows
                
                El_i_newshape = data['elm'][:n_rows * n_cols].reshape(n_rows, n_cols)
                El_it  = np.full((n_rows, nZ), np.nan)
                El_it[:, :n_cols] = El_i_newshape
                
                
                Ne_i_newshape = data['ne'][:n_rows * n_cols].reshape(n_rows, n_cols)
                Ne_i  = np.full((n_rows, nZ), np.nan)
                Ne_i[:, :n_cols] = Ne_i_newshape
                
                
                DNe_i_newshape = data['dne'][:n_rows * n_cols].reshape(n_rows, n_cols)
                DNe_i  = np.full((n_rows, nZ), np.nan)
                DNe_i[:, :n_cols] = DNe_i_newshape
                
                Vi_i_newshape = data['vo'][:n_rows * n_cols].reshape(n_rows, n_cols)
                Vi_i  = np.full((n_rows, nZ), np.nan)
                Vi_i[:, :n_cols] = Vi_i_newshape
                
                Te_i_newshape = te[:n_rows * n_cols].reshape(n_rows, n_cols)
                Te_i  = np.full((n_rows, nZ), np.nan)
                Te_i[:, :n_cols] = Te_i_newshape
                
                Ti_i_newshape = data['ti'][:n_rows * n_cols].reshape(n_rows, n_cols)
                Ti_i  = np.full((n_rows, nZ), np.nan)
                Ti_i[:, :n_cols] = Ti_i_newshape
                
                range_i_newshape = range_data[:n_rows * n_cols].reshape(n_rows, n_cols)
                range_i  = np.full((n_rows, nZ), np.nan)
                range_i[:, :n_cols] = range_i_newshape
                
                
                
                self.plot_and_save_results(t_i, range_i, Ne_i, Te_i, Ti_i, Vi_i, year, month, day)
                self.save_mat_file(t_i, range_i, Ne_i, DNe_i, year, month, day)

            else:
                El_it  = data['elm'].reshape(shape)
                Ne_i  = data['ne'].reshape(shape)
                DNe_i = data['dne'].reshape(shape)
                Vi_i  = data['vo'].reshape(shape)
                Te_i  = te.reshape(shape)
                Ti_i  = data['ti'].reshape(shape)
                range_i = range_data.reshape(shape)
                
                self.plot_and_save_results(t_i, range_i, Ne_i, Te_i, Ti_i, Vi_i, year, month, day)
                self.save_mat_file(t_i, range_i, Ne_i, DNe_i, year, month, day)

    def plot_and_save_results(self, t_i, range_i, Ne_i, Te_i, Ti_i, Vi_i, year, month, day):
        plt.figure(dpi=50, figsize=(6, 8))
        
        t_i = np.nan_to_num(t_i, nan=0.0)
        range_i_mean = np.nan_to_num(np.nanmean(range_i, axis=0), nan=0.0)
        
        
        ax1 = plt.subplot(4, 1, 1)
        plt.pcolormesh(t_i, range_i_mean, np.ma.masked_invalid(Ne_i.T), shading='auto')
        plt.colorbar()
        plt.clim(1e9, 5e11)

        ax2 = plt.subplot(4, 1, 2)
        plt.pcolormesh(t_i, range_i_mean, np.ma.masked_invalid(Te_i.T), shading='auto')
        plt.colorbar()
        plt.clim(500, 4000)

        ax3 = plt.subplot(4, 1, 3)
        plt.pcolormesh(t_i, range_i_mean, np.ma.masked_invalid(Ti_i.T), shading='auto')
        plt.colorbar()
        plt.clim(500, 3000)

        ax4 = plt.subplot(4, 1, 4)
        plt.pcolormesh(t_i, range_i_mean,np.ma.masked_invalid(Vi_i.T), shading='auto')
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
        for iE in range(self.num_datafiles):
            self.process_file(iE)

# Usage example:
datapath = "beata_vhf"
resultpath = "Ne_new"

processor = EISCATDataProcessor(datapath, resultpath)
processor.process_all_files()


















