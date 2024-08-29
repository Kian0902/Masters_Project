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
    
    

    def process_file(self, file_index: int):
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

            # Extract year, month, and day from the file name
            year = int(file[8:12])
            month = int(file[13:15])
            day = int(file[16:18])
            
            # Convert time data (year, month, day, hour, min, sec) into datetime format
            t = [datetime(year, month, day, int(h), int(m), int(s))
                  for h, m, s in zip(data['hour'], data['min'], data['sec'])]
            
            # Extract the range information
            range_data = data['range'][:]

            # Find indices where range data shows significant drops (indicating a new time step)
            ind = np.concatenate(([0], np.where(np.diff(range_data) < -500)[0] + 1))
            
            # Get the time steps corresponding to these indices
            t_i = np.array(t)[ind]
            
            # Combine electron and ion temperatures
            te = data['tr'][:] * data['ti'][:]
            
            
            nT = len(ind)  # number of time-steps
            
            # Find most common value
            unique_values, counts = np.unique(np.diff(ind), return_counts=True)  # find num of unique values  
            most_repeated_value = unique_values[np.argmax(counts)]  # get the most repeated value
            nZ = int(most_repeated_value)  # convert to int
            

            shape = (nT, nZ)
            if shape[0]*shape[1] != range_data.size:
                print("Warning! Detected unmatching shapes.\nSwitching from vectorized to manual processing.\n")

                # Initialize matrices to store interpolated data
                Ne_i = np.full((nT, nZ), np.nan)  # Electron density
                DNe_i = np.full((nT, nZ), np.nan)  # Electron density error
                range_i = np.full((nT, nZ), np.nan)  # Range values
                Te_i = np.full((nT, nZ), np.nan)  # Electron temperature
                Ti_i = np.full((nT, nZ), np.nan)  # Ion temperature
                Vi_i = np.full((nT, nZ), np.nan)  # Ion velocity
                El_i = np.full((nT, nZ), np.nan)  # Elevation angle
                
                # Loop over time steps to process the data
                for iT in range(nT - 1):
                    # Define the start and end indices for each time step
                    ind_s = ind[iT]
                    ind_f = ind[iT + 1]
                    
                    # Extract elevation angle for this time range
                    El_iT = data['elm'][ind_s:ind_f]
                    
                    # Check if the mean elevation is close to 90 degrees (indicating data is usable)
                    if round(abs(np.mean(El_iT) - 90)) < 6:
                        # Determine the number of elements that will fit in the current row of the matrices
                        num_elements = min(ind_f - ind_s, nZ)  # Truncate to fit within the matrix dimensions
                        
                        # Store data for this time step, truncating if necessary
                        Ne_i[iT, :num_elements] = data['ne'][ind_s:ind_f][:num_elements]
                        DNe_i[iT, :num_elements] = data['dne'][ind_s:ind_f][:num_elements]
                        Vi_i[iT, :num_elements] = data['vo'][ind_s:ind_f][:num_elements]
                        Te_i[iT, :num_elements] = te[ind_s:ind_f][:num_elements]
                        Ti_i[iT, :num_elements] = data['ti'][ind_s:ind_f][:num_elements]
                        range_i[iT, :num_elements] = range_data[ind_s:ind_f][:num_elements]
                                                
                                                
                
                print("Manual processing complete.\nSwitching back to vectorized operations...\n")
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
    
    
    
    def process_all_files(self):
        for i in range(self.num_datafiles):
            self.process_file(i)
        print(f"Processing complete. Processed {self.num_datafiles} data files.")
    
    
    
    def save_mat_file(self, t_i, range_i, Ne_i, DNe_i, year, month, day):
        datetime_matrix = np.array([[dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second] for dt in t_i])
        r_time = datetime_matrix
        r_h = range_i
        r_param = Ne_i
        r_error = DNe_i

        name = f"{year}-{month}-{day}.mat"
        sio.savemat(os.path.join(self.resultpath, name), {'r_time': r_time, 'r_h': r_h, 'r_param': r_param, 'r_error': r_error})
    
    
    def plot_and_save_results(self, t_i, range_i, Ne_i, Te_i, Ti_i, Vi_i, year, month, day):
        # Plot the results for visualization
        plt.figure()

        # Electron density plot
        ax1 = plt.subplot(4, 1, 1)
        plt.pcolormesh(t_i, np.nanmean(range_i, axis=0), Ne_i.T, shading='auto')
        plt.colorbar()
        plt.clim(1e9, 5e11)  # Set color axis limits

        # Electron temperature plot
        ax2 = plt.subplot(4, 1, 2)
        plt.pcolormesh(t_i, np.nanmean(range_i, axis=0), Te_i.T, shading='auto')
        plt.colorbar()
        plt.clim(500, 4000)  # Set color axis limits

        # Ion temperature plot
        ax3 = plt.subplot(4, 1, 3)
        plt.pcolormesh(t_i, np.nanmean(range_i, axis=0), Ti_i.T, shading='auto')
        plt.colorbar()
        plt.clim(500, 3000)  # Set color axis limits

        # Ion velocity plot
        ax4 = plt.subplot(4, 1, 4)
        plt.pcolormesh(t_i, np.nanmean(range_i, axis=0), Vi_i.T, shading='auto')
        plt.colorbar()
        plt.clim(-400, 400)  # Set color axis limits

        # Set the colormap and link the axes of all plots
        plt.set_cmap('turbo')
        plt.subplots_adjust(hspace=0.5)
        # plt.show()

        # # Save the figure as a .png file in the result folder
        os.chdir(self.resultpath)
        name = f"{year}-{month}-{day}.png"
        plt.savefig(name)
        plt.close()


    

# # Usage example:
# datapath = "beata_zenith_data_uhf"
# resultpath = "Ne_uhf"

# processor = EISCATDataProcessor(datapath, resultpath)
# processor.process_all_files()


















