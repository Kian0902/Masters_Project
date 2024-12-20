# -*- coding: utf-8 -*-
"""
Created:
    Sun Dec 10 14:32:00 2023
    @author: Kian Sartipzadeh

Updated:
    Sat Sep 28 13:36:00 2024
    @author: Kian Sartipzadeh
"""




import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy.interpolate import RegularGridInterpolator
from datetime import datetime





class IonogramProcessing:
    def __init__(self):
        self.a = 1
    

    def resample_ionogram(self, iono_org, freq_org, rang_org, f, r, output_size):
        """
        Resampling onto an 81x81 grid using RegularGridInterpolator
        """

        # Create interpolator for each mode separately
        iono_resampled = np.zeros((output_size, output_size, 3))
        for mode in range(2):  # Mode 0 is O-mode, Mode 1 is X-mode
            interpolator = RegularGridInterpolator(
                (rang_org, freq_org), iono_org[:, :, mode], method='linear', bounds_error=False, fill_value=0
            )
            # Interpolate on the new grid
            grid_coords = np.array([r.ravel(), f.ravel()]).T
            iono_resampled[:, :, mode] = interpolator(grid_coords).reshape((output_size, output_size))

        # Normalize and convert to uint8
        iono_resampled = (iono_resampled / np.max(iono_resampled)) * 255
        iono_resampled = iono_resampled.astype(np.uint8)
        iono_resampled = np.rot90(iono_resampled, k=1)
        
        return iono_resampled
    
    
    
    def process_ionogram(self, data, times, plot=False, result_path=None):
        """
        Function for reconstructing ionograms to their original dimensions, then
        resampling onto a 81x81 grid.
        
        This function takes in data that has been processed by the "import_data"
        function which contains ionograms over a whole day.
        
        Input (type)        | DESCRIPTION
        ------------------------------------------------
        data  (np.ndarray)  | ndarray with original ionograms 
        times (np.ndarray)  | Timestamps of the original ionograms
        plot  (bool)        | Argument for plotting ionograms 
        result_path (str)   | Path for saving processed ionograms
        """
        
        # Defining ionogram axes
        freq_org = np.arange(1, 16 + 0.05, 0.05)  # original ionogram freq: [1, 16] MHz
        rang_org = np.arange(80, 640 + 5, 5)      # original ionogram range: [80, 640] km 
        
        # Max and min allowed amplitudes
        I_max = 75
        I_min = 20
        
        # Prepare output grid for resampling
        output_size = 81
        Frange = [1, 9]  # Frequency range of interest
        Zrange = [80, 480]  # Range of interest
        frequency_axis = np.linspace(Frange[0], Frange[1], output_size)
        range_axis = np.linspace(Zrange[0], Zrange[1], output_size)
        r, f = np.meshgrid(range_axis, frequency_axis)
        
    
        # Process each ionogram
        for i in np.arange(0, len(data)):
            time = times[i]
            data_i = data[i]
            
            # print(f'  - Processing ionogram at time {time}')
            
            """ Reconstructing ionograms to original dimensions"""
            freq = np.around(data_i[:, 0], decimals=2)     # frequencies [MHz]
            rang = np.around(data_i[:, 1], decimals=2)     # ionosonde vertual range [km]
            pol  = np.round(data_i[:, 2])                  # polarization (either 90 or -90 degrees)
            amp  = data_i[:, 4]                            # backscatter amplitude
            dop  = np.around(data_i[:, 5], decimals=2)     # doppler shift
            ang  = np.round(data_i[:, 7])                  # received angle [deg]
    
            """ Recreate ionogram """
            
            iono_org = np.zeros((len(rang_org), len(freq_org), 3))
            
            # Finding indices and ensuring they stay within valid bounds
            F_idx = np.clip(np.searchsorted(freq_org, freq), 0, len(freq_org) - 1)
            Z_idx = np.clip(np.searchsorted(rang_org, rang), 0, len(rang_org) - 1)
            I_idx = np.clip(amp, I_min, I_max)  # only interested in amp: [21, 75]
            
            mask_O = (pol == 90) & (ang == 0)   # mask for positive 90 deg pol values (O-mode) and 0 deg ang values
            mask_X = (pol == -90) & (ang == 0)  # mask for positive -90 deg pol values (X-mode) and 0 deg ang values
    
            # O and X-mode without doppler shift
            iono_org[Z_idx[mask_O], F_idx[mask_O], 0] = (I_idx[mask_O] - I_min) / (I_max - I_min)  # Scale amplitude 0 to 1
            iono_org[Z_idx[mask_X], F_idx[mask_X], 1] = (I_idx[mask_X] - I_min) / (I_max - I_min)  # Scale amplitude 0 to 1
            
            
            # Normalize the ionogram data
            iono_org = (iono_org / np.max(iono_org)) * 255  # multiplying by 255 for image purposes
            iono_org = iono_org.astype(np.uint8)
            
            
            iono_resampled = self.resample_ionogram(iono_org, freq_org, rang_org, f, r, output_size)
            
            
            
            
            iono_org = Image.fromarray(iono_org).transpose(Image.FLIP_TOP_BOTTOM)
            iono_image = Image.fromarray(iono_resampled)
            
            
            # Save the resampled image if a result path is provided
            if result_path:
                iono_resampled_image = Image.fromarray(iono_resampled)
                
                # Format the timestamp into yyyyMMdd_hhmm
                time_str = datetime.strptime(time, "%Y.%m.%d_%H-%M-%S").strftime("%Y%m%d_%H%M")
                save_filename = os.path.join(result_path, f"{time_str}.png")
                
                iono_resampled_image.save(save_filename)
                # print(f"  - Saved ionogram image to {save_filename}")
            
            if plot:
                fig, ax = plt.subplots(1, 2, width_ratios=[1, 0.6], figsize=(12, 5))
                
                # Original ionogram
                ax[0].imshow(iono_org, extent=[freq_org[0], freq_org[-1], rang_org[0], rang_org[-1]],
                                    aspect='auto')
                ax[0].set_title("Original Ionogram", fontsize=20)
                ax[0].set_xlabel("Frequency (MHz)", fontsize=15)
                ax[0].set_ylabel("Virtual Altitude (km)", fontsize=15)
                
                
                # Resampled ionogram
                ax[1].imshow(iono_resampled, extent=[Frange[0], Frange[1], Zrange[0], Zrange[1]],
                                    aspect='auto')
                ax[1].set_title("Resampled Ionogram", fontsize=20)
                ax[1].set_xlabel("Frequency (MHz)", fontsize=15)
                ax[1].set_ylabel("Virtual Altitude (km)", fontsize=15)
                
                
                # Custom labels with filled squares
                green_patch = Patch(color='green', label='X-mode')
                red_patch = Patch(color='red', label='O-mode')
                
                # Add legends
                for axis in ax:
                    axis.legend(handles=[red_patch, green_patch], loc='upper right', title="Modes", frameon=True)
                
                
                plt.tight_layout()
                plt.show()
            
            
            # if plot:
            #     fig, ax = plt.subplots(1, 2, width_ratios=[1, 0.5], frameon=False)
                
            #     ax[0].imshow(iono_org)
            #     ax[1].imshow(iono_image)
            #     plt.show()
    
        print("Processing complete.")














