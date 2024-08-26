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
    Class for processing EISCAT data taken from guisdap.
    """
    def __init__(self, folder_name_in: str, folder_name_out: str):
        """
        Input (type)         | DESCRIPTION
        ------------------------------------------------
        folder_name_in  (str) | Name of local folder containing data.
        folder_name_out (str) | Name of local folder for storing the processed data.
        
        
        Attributes (type)  | DESCRIPTION
        ------------------------------------------------
        datapath   (str)    | Full path to local folder containing input data.
        resultpath (str)    | Full path to local folder for storing output data.
        datafiles  (list)   | List containing names of -hdf5 datafiles.
        num_datafiles (int) | Number of files to process.
        """
        self.datapath   = os.path.abspath(os.path.join(os.getcwd(), folder_name_in))
        self.resultpath = os.path.abspath(os.path.join(os.getcwd(), folder_name_out))
        self.datafiles = self._find_data_files()
        self.num_datafiles = len(self.datafiles)
    
    
    
    def _find_data_files(self):
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
        for iE in range(1):#self.num_datafiles):
            self.process_file(iE)

# Usage example:
datapath = "beata_vhf"
resultpath = "Ne_new"

processor = EISCATDataProcessor(datapath, resultpath)
processor.process_all_files()


















