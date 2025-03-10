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
from scipy.interpolate import RegularGridInterpolator
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.gridspec import GridSpec
import seaborn as sns
from datetime import datetime



# class IonogramProcessing:
#     def __init__(self):
#         # Original ionogram axes and parameters
#         self.freq_org = np.arange(1, 16 + 0.05, 0.05)  # Frequency axis: 1-16 MHz, step 0.05
#         self.rang_org = np.arange(80, 640 + 5, 5)       # Range axis: 80-640 km, step 5
#         self.I_max = 120  # Maximum amplitude for scaling
#         self.I_min = 20  # Minimum amplitude for scaling

#     def amplitude_filter(self, freq, amp, threshold_fraction=0.75):
#         mask_amp = np.zeros_like(freq, dtype=bool)
#         unique_freqs = np.unique(freq)
#         for f in unique_freqs:
#             indices = np.where(freq == f)[0]
#             if len(indices) == 0:
#                 continue
#             current_amps = amp[indices]
#             max_amp_f = np.max(current_amps)
#             threshold = threshold_fraction * max_amp_f
#             mask_amp[indices] = current_amps >= threshold
#         return mask_amp

#     def reconstruct_ionogram(self, data_i, apply_amplitude_filter=False):
#         # Extract data components
#         freq = np.around(data_i[:, 0], decimals=2)  # Frequency values
#         rang = np.around(data_i[:, 1], decimals=2)  # Range values
#         pol  = np.round(data_i[:, 2])               # Polarization values
#         amp  = data_i[:, 4]                         # Amplitude values
#         ang  = np.round(data_i[:, 7])               # Angle values

#         # Initialize ionogram structure: dimensions (range, frequency, channels)
#         iono_org = np.zeros((len(self.rang_org), len(self.freq_org), 3))
        
#         # Calculate indices for frequency and range
#         F_idx = np.clip(np.searchsorted(self.freq_org, freq), 0, len(self.freq_org)-1)
#         Z_idx = np.clip(np.searchsorted(self.rang_org, rang), 0, len(self.rang_org)-1)
#         I_idx = np.clip(amp, self.I_min, self.I_max)  # Clip amplitudes
        
#         # Apply amplitude filter (optional)
#         if apply_amplitude_filter:
#             mask_amp = self.amplitude_filter(freq, amp, threshold_fraction=0.75)
#         else:
#             mask_amp = np.ones_like(freq, dtype=bool)
        
#         # Create masks for O-mode and X-mode
#         mask_O = (pol == 90) & (ang == 0) & mask_amp
#         mask_X = (pol == -90) & (ang == 0) & mask_amp
        
#         # Populate the image channels
#         iono_org[Z_idx[mask_O], F_idx[mask_O], 0] = (I_idx[mask_O] - self.I_min) / (self.I_max - self.I_min)
#         iono_org[Z_idx[mask_X], F_idx[mask_X], 1] = (I_idx[mask_X] - self.I_min) / (self.I_max - self.I_min)
        
#         # Normalize to 255 and convert to uint8
#         if np.max(iono_org) > 0:
#             iono_org = (iono_org / np.max(iono_org)) * 255
#         else:
#             iono_org = np.zeros_like(iono_org)
#         return iono_org.astype(np.uint8)

#     def resample_ionogram(self, iono_org, Frange=[1, 9], Zrange=[80, 480], output_size=81):
#         # New grid coordinates
#         frequency_axis = np.linspace(Frange[0], Frange[1], output_size)
#         range_axis = np.linspace(Zrange[0], Zrange[1], output_size)
#         r, f = np.meshgrid(range_axis, frequency_axis)

#         # Prepare the output for O-mode and X-mode (channels 0 and 1)
#         iono_resampled = np.zeros((output_size, output_size, 3))
#         for mode in range(2):
#             interpolator = RegularGridInterpolator(
#                 (self.rang_org, self.freq_org), iono_org[:, :, mode],
#                 method='linear', bounds_error=False, fill_value=0
#             )
#             grid_coords = np.array([r.ravel(), f.ravel()]).T
#             iono_resampled[:, :, mode] = interpolator(grid_coords).reshape((output_size, output_size))
        
#         # Normalize and convert to uint8
#         if np.max(iono_resampled) > 0:
#             iono_resampled = (iono_resampled / np.max(iono_resampled)) * 255
#         else:
#             iono_resampled = np.zeros_like(iono_resampled)
#         iono_resampled = iono_resampled.astype(np.uint8)
#         # Rotate for proper display orientation
#         return np.rot90(iono_resampled, k=1)


class IonogramProcessing:
    def __init__(self):
        # Original ionogram axes and parameters
        self.freq_org = np.arange(1, 16 + 0.05, 0.05)  # Frequency axis: 1-16 MHz, step 0.05
        self.rang_org = np.arange(80, 640 + 5, 5)       # Range axis: 80-640 km, step 5

    def amplitude_filter(self, freq, amp, threshold_fraction=0.75):
        mask_amp = np.zeros_like(freq, dtype=bool)
        unique_freqs = np.unique(freq)
        for f in unique_freqs:
            indices = np.where(freq == f)[0]
            if indices.size == 0:
                continue
            current_amps = amp[indices]
            max_amp_f = np.max(current_amps)
            threshold = threshold_fraction * max_amp_f
            mask_amp[indices] = current_amps >= threshold
        return mask_amp

    def reconstruct_ionogram(self, data_i, apply_amplitude_filter=False):
        # Extract data components
        freq = np.around(data_i[:, 0], decimals=2)  # Frequency values
        rang = np.around(data_i[:, 1], decimals=2)  # Range values
        pol  = np.round(data_i[:, 2])               # Polarization values
        amp  = data_i[:, 4]                         # Amplitude values
        ang  = np.round(data_i[:, 7])               # Angle values

        # Initialize ionogram array: dimensions (range, frequency, channels)
        iono_org = np.zeros((len(self.rang_org), len(self.freq_org), 3))
        
        # Calculate indices for frequency and range on the grid
        F_idx = np.clip(np.searchsorted(self.freq_org, freq), 0, len(self.freq_org)-1)
        Z_idx = np.clip(np.searchsorted(self.rang_org, rang), 0, len(self.rang_org)-1)
        
        # Apply amplitude filter if requested
        if apply_amplitude_filter:
            mask_amp = self.amplitude_filter(freq, amp, threshold_fraction=0.75)
        else:
            mask_amp = np.ones_like(freq, dtype=bool)
        
        # Create masks for O-mode and X-mode
        mask_O = (pol == 90) & (ang == 0) & mask_amp
        mask_X = (pol == -90) & (ang == 0) & mask_amp

        # Process O-mode: normalize amplitude using the dynamic range found in the data
        if np.any(mask_O):
            amp_O = amp[mask_O]
            min_O, max_O = np.min(amp_O), np.max(amp_O)
            # Avoid division by zero
            if max_O - min_O > 0:
                norm_O = (amp_O - min_O) / (max_O - min_O)
            else:
                norm_O = np.zeros_like(amp_O)
            iono_org[Z_idx[mask_O], F_idx[mask_O], 0] = norm_O

        # Process X-mode similarly
        if np.any(mask_X):
            amp_X = amp[mask_X]
            min_X, max_X = np.min(amp_X), np.max(amp_X)
            if max_X - min_X > 0:
                norm_X = (amp_X - min_X) / (max_X - min_X)
            else:
                norm_X = np.zeros_like(amp_X)
            iono_org[Z_idx[mask_X], F_idx[mask_X], 1] = norm_X

        # Finally, scale the entire ionogram to 0-255 and convert to uint8.
        if np.max(iono_org) > 0:
            iono_org = (iono_org / np.max(iono_org)) * 255
        else:
            iono_org = np.zeros_like(iono_org)
        return iono_org.astype(np.uint8)

    def resample_ionogram(self, iono_org, Frange=[1, 9], Zrange=[80, 480], output_size=81):
        # Create new grid coordinates for resampling
        frequency_axis = np.linspace(Frange[0], Frange[1], output_size)
        range_axis = np.linspace(Zrange[0], Zrange[1], output_size)
        r, f = np.meshgrid(range_axis, frequency_axis)

        # Prepare output ionogram for O-mode and X-mode (channels 0 and 1)
        iono_resampled = np.zeros((output_size, output_size, 3))
        for mode in range(2):
            interpolator = RegularGridInterpolator(
                (self.rang_org, self.freq_org), iono_org[:, :, mode],
                method='linear', bounds_error=False, fill_value=0
            )
            grid_coords = np.array([r.ravel(), f.ravel()]).T
            iono_resampled[:, :, mode] = interpolator(grid_coords).reshape((output_size, output_size))
        
        # Normalize to 0-255 and convert to uint8
        if np.max(iono_resampled) > 0:
            iono_resampled = (iono_resampled / np.max(iono_resampled)) * 255
        else:
            iono_resampled = np.zeros_like(iono_resampled)
        iono_resampled = iono_resampled.astype(np.uint8)
        # Rotate for proper display orientation
        return np.rot90(iono_resampled, k=1)


    def process_and_plot(self, data, times, result_path=None, apply_amplitude_filter=False):
        """
        Processes a list of ionogram data arrays and corresponding timestamps.
        Generates a 2x3 figure for each ionogram:
          - Top row: Original ionogram (spanning columns 1-2) and Resampled ionogram (column 3)
          - Bottom row: Three filtering plots.
        Optionally saves the figure if a result_path is provided.
        """
        output_size = 81
        Frange = [1, 9]
        Zrange = [80, 480]
        
        for i in range(len(data)):
            time_str_raw = times[i]
            data_i = data[i]
            
            # Reconstruct and resample ionogram
            iono_org = self.reconstruct_ionogram(data_i, apply_amplitude_filter=apply_amplitude_filter)
            iono_resampled = self.resample_ionogram(iono_org, Frange, Zrange, output_size)
            
            # --- Prepare filtering data ---
            freq = np.around(data_i[:, 0], decimals=2)
            rang = np.around(data_i[:, 1], decimals=2)
            pol = np.round(data_i[:, 2])
            amp = data_i[:, 4]
            ang = np.round(data_i[:, 7])
            
            # Create amplitude filter mask
            mask_amp = np.zeros_like(freq, dtype=bool)
            unique_freqs = np.unique(freq)
            for f in unique_freqs:
                indices = np.where(freq == f)[0]
                if len(indices) == 0:
                    continue
                current_amps = amp[indices]
                max_amp_f = np.max(current_amps)
                threshold = 0.75 * max_amp_f
                mask_amp[indices] = current_amps >= threshold
            
            # Masks for scatter plots
            mask_O = (pol == 90) & (ang == 0) & mask_amp
            mask_X = (pol == -90) & (ang == 0) & mask_amp
            org_mask_O = (pol == 90) & (ang == 0)
            org_mask_X = (pol == -90) & (ang == 0)
            
            
            
            # --- Create figure ---
            fig = plt.figure(figsize=(15, 10))
            gs = GridSpec(2, 5, height_ratios=[1.2, 1], width_ratios=[1, 1, 1, -0.35, 0.05], hspace=0.3, wspace=0.4)
            
            # Top row plots (using default style)
            ax_top_org = fig.add_subplot(gs[0, :2])  # Original ionogram spans columns 0 and 1
            ax_top_res = fig.add_subplot(gs[0, 2])     # Resampled ionogram in column 2
            
            # Plot the original ionogram (flip vertically for proper display)
            iono_org_disp = Image.fromarray(iono_org).transpose(Image.FLIP_TOP_BOTTOM)
            ax_top_org.imshow(
                iono_org_disp,
                extent=[self.freq_org[0], self.freq_org[-1],
                        self.rang_org[0], self.rang_org[-1]],
                aspect='auto'
            )
            ax_top_org.set_title("Filtered Ionogram", fontsize=17)
            ax_top_org.set_xlabel("Frequency (MHz)", fontsize=13)
            ax_top_org.set_ylabel("Virtual Altitude (km)", fontsize=13)
            ax_top_org.legend(handles=[Patch(color='red', label='O-mode'),
                                       Patch(color='green', label='X-mode')],
                              loc='upper right')
            
            # Plot the resampled ionogram
            ax_top_res.imshow(
                iono_resampled,
                extent=[Frange[0], Frange[1], Zrange[0], Zrange[1]],
                aspect='auto'
            )
            ax_top_res.set_title("Resampled Ionogram", fontsize=17)
            ax_top_res.set_xlabel("Frequency (MHz)", fontsize=13)
            ax_top_res.set_ylabel("Virtual Altitude (km)", fontsize=13)
            ax_top_res.legend(handles=[Patch(color='red', label='O-mode'),
                                       Patch(color='green', label='X-mode')],
                              loc='upper right')
            
            # --- Bottom row: Filtering scatter plots with Seaborn dark style ---
            # Use a context manager so that only these axes get the dark style.
            with sns.axes_style("dark"):
                ax_filt0 = fig.add_subplot(gs[1, 0])
                ax_filt1 = fig.add_subplot(gs[1, 1])
                ax_filt2 = fig.add_subplot(gs[1, 2])
                ax_space = fig.add_subplot(gs[1, 3])
                ax_space.set_visible(False)
                
                cax      = fig.add_subplot(gs[1, 4])
                
                
                # Plot 1: Original (non-filtered) data
                scatter0 = ax_filt0.scatter(freq[org_mask_O], rang[org_mask_O], s=1, 
                                            c=amp[org_mask_O], cmap="turbo", zorder=2)
                ax_filt0.scatter(freq[org_mask_X], rang[org_mask_X], s=1, 
                                 c=amp[org_mask_X], cmap="turbo", zorder=2)
                ax_filt0.set_title("Original", fontsize=15)
                ax_filt0.set_xlabel("Frequency (MHz)", fontsize=13)
                ax_filt0.set_ylabel("Virtual Altitude (km)", fontsize=13)
                ax_filt0.set_xlim(0.9, 9.1)
                ax_filt0.set_ylim(78, 645)
                ax_filt0.grid(True)
                
                # Plot 2: Outlined Noise
                scatter1 = ax_filt1.scatter(freq[mask_O], rang[mask_O], s=1, 
                                            c=amp[mask_O], cmap="turbo", zorder=2)
                ax_filt1.scatter(freq[mask_X], rang[mask_X], s=1, 
                                 c=amp[mask_X], cmap="turbo", zorder=2)
                # Overlay original (non-filtered) points in black for context
                ax_filt1.scatter(freq[org_mask_O], rang[org_mask_O], s=1, c="black", zorder=1)
                ax_filt1.scatter(freq[org_mask_X], rang[org_mask_X], s=1, c="black", zorder=1)
                ax_filt1.set_title("Outlined Noise", fontsize=15)
                ax_filt1.set_xlabel("Frequency (MHz)", fontsize=13)
                ax_filt1.set_xlim(0.9, 9.1)
                ax_filt1.set_ylim(78, 645)
                ax_filt1.grid(True)
                
                # Plot 3: Filtered data
                scatter2 = ax_filt2.scatter(freq[mask_O], rang[mask_O], s=1, 
                                            c=amp[mask_O], cmap="turbo", zorder=2)
                ax_filt2.scatter(freq[mask_X], rang[mask_X], s=1, 
                                 c=amp[mask_X], cmap="turbo", zorder=2)
                ax_filt2.set_title("Filtered", fontsize=15)
                ax_filt2.set_xlabel("Frequency (MHz)", fontsize=13)
                ax_filt2.set_xlim(0.9, 9.1)
                ax_filt2.set_ylim(78, 645)
                ax_filt2.grid(True)
                
                cbar = fig.colorbar(scatter0, cax=cax, orientation='vertical')
                cbar.set_label('Amplitude', fontsize=13)
                
            
            # Set the overall title using the timestamp
            time_display = datetime.strptime(time_str_raw, "%Y.%m.%d_%H-%M-%S").strftime("%Y-%m-%d %H:%M")
            fig.suptitle(time_display, fontsize=18)
            
            plt.tight_layout()
            plt.show()
            
            # Optionally, save the figure if a result path is provided
            if result_path:
                time_file = datetime.strptime(time_str_raw, "%Y.%m.%d_%H-%M-%S").strftime("%Y%m%d_%H%M")
                fig.savefig(os.path.join(result_path, f"{time_file}_combined.png"))



# class IonogramProcessing:
#     def __init__(self):
#         # Original ionogram axes and parameters
#         self.freq_org = np.arange(1, 16 + 0.05, 0.05)  # Frequency axis: 1-16 MHz, step 0.05
#         self.rang_org = np.arange(80, 640 + 5, 5)      # Range axis: 80-640 km, step 5
#         self.I_max = 75  # Maximum amplitude for scaling
#         self.I_min = 20    # Minimum amplitude for scaling

#     def amplitude_filter(self, freq, amp, threshold_fraction=0.75):
#         """
#         Returns a boolean mask indicating which data points
#         pass the amplitude filter based on `threshold_fraction`.
        
#         :param freq: 1D array of frequency values
#         :param amp:  1D array of amplitude values
#         :param threshold_fraction: Fraction of the maximum amplitude
#                                    used as the cutoff threshold
#         :return: Boolean mask (True where data is kept)
#         """
#         mask_amp = np.zeros_like(freq, dtype=bool)
#         unique_freqs = np.unique(freq)
        
#         for f in unique_freqs:
#             indices = np.where(freq == f)[0]
#             if len(indices) == 0:
#                 continue
#             current_amps = amp[indices]
#             max_amp_f = np.max(current_amps)
#             threshold = threshold_fraction * max_amp_f
#             mask_amp[indices] = current_amps >= threshold
        
#         return mask_amp

#     def reconstruct_ionogram(self, data_i, apply_amplitude_filter=False):
#         """
#         Reconstructs the ionogram to its original dimensions from raw data.
#         Returns the reconstructed ionogram as a uint8 array.
        
#         :param data_i: Raw ionogram data (N x M array-like)
#         :param apply_amplitude_filter: Whether to apply amplitude-based filtering
#         """
#         # Extract data components
#         freq = np.around(data_i[:, 0], decimals=2)  # Frequency values
#         rang = np.around(data_i[:, 1], decimals=2)  # Range values
#         pol  = np.round(data_i[:, 2])               # Polarization values
#         amp  = data_i[:, 4]                         # Amplitude values
#         ang  = np.round(data_i[:, 7])               # Angle values
        
#         # Initialize ionogram structure
#         iono_org = np.zeros((len(self.rang_org), len(self.freq_org), 3))
        
#         # Calculate indices for frequency and range
#         F_idx = np.clip(np.searchsorted(self.freq_org, freq), 0, len(self.freq_org)-1)
#         Z_idx = np.clip(np.searchsorted(self.rang_org, rang), 0, len(self.rang_org)-1)
#         I_idx = np.clip(amp, self.I_min, self.I_max)  # Clip amplitudes
        
#         # Apply amplitude filter (optional)
#         if apply_amplitude_filter:
#             mask_amp = self.amplitude_filter(freq, amp, threshold_fraction=0.75)
#         else:
#             # If not applying the filter, just use a mask that keeps all points
#             mask_amp = np.ones_like(freq, dtype=bool)
        
#         # Apply masks for polarization, angle, and amplitude
#         mask_O = (pol == 90) & (ang == 0) & mask_amp
#         mask_X = (pol == -90) & (ang == 0) & mask_amp
        
#         # Populate O-mode and X-mode data
#         iono_org[Z_idx[mask_O], F_idx[mask_O], 0] = \
#             (I_idx[mask_O] - self.I_min) / (self.I_max - self.I_min)
#         iono_org[Z_idx[mask_X], F_idx[mask_X], 1] = \
#             (I_idx[mask_X] - self.I_min) / (self.I_max - self.I_min)
        
#         # Normalize and convert to uint8
#         if np.max(iono_org) > 0:
#             iono_org = (iono_org / np.max(iono_org)) * 255
#         else:
#             iono_org = np.zeros_like(iono_org)
#         iono_org = iono_org.astype(np.uint8)
        
#         return iono_org

#     def resample_ionogram(self, iono_org, Frange=[1, 9], Zrange=[80, 480], output_size=81):
#         """
#         Resamples the ionogram onto a new grid of specified size.
#         Returns the resampled ionogram as a uint8 array.
#         """
#         # Create new grid coordinates
#         frequency_axis = np.linspace(Frange[0], Frange[1], output_size)
#         range_axis = np.linspace(Zrange[0], Zrange[1], output_size)
#         r, f = np.meshgrid(range_axis, frequency_axis)

#         # Interpolate each mode
#         iono_resampled = np.zeros((output_size, output_size, 3))
#         for mode in range(2):  # Process O-mode and X-mode
#             interpolator = RegularGridInterpolator(
#                 (self.rang_org, self.freq_org), iono_org[:, :, mode],
#                 method='linear', bounds_error=False, fill_value=0
#             )
#             grid_coords = np.array([r.ravel(), f.ravel()]).T
#             iono_resampled[:, :, mode] = interpolator(grid_coords).reshape((output_size, output_size))

#         # Normalize and format
#         if np.max(iono_resampled) > 0:
#             iono_resampled = (iono_resampled / np.max(iono_resampled)) * 255
#         else:
#             iono_resampled = np.zeros_like(iono_resampled)
#         iono_resampled = iono_resampled.astype(np.uint8)
#         # Rotate for correct orientation
#         iono_resampled = np.rot90(iono_resampled, k=1)

#         return iono_resampled
    


#     def process_ionogram(self, data, times, plot=False, apply_amplitude_filter=False, result_path=None):
#         """
#         Processes ionograms through reconstruction and resampling.
#         Handles batch processing of multiple ionograms.
        
#         :param data: List (or array-like) of raw ionogram data arrays
#         :param times: Corresponding list of timestamp strings
#         :param plot: Whether to plot the reconstructed and resampled ionograms
#         :param result_path: Directory path to save results (optional)
#         :param apply_amplitude_filter: Pass through to `reconstruct_ionogram`
#         """
#         # Resampling parameters
#         output_size = 81
#         Frange = [1, 9]
#         Zrange = [80, 480]

#         for i in range(len(data)):
#             time = times[i]
#             data_i = data[i]

#             # Reconstruct original ionogram (with or without amplitude filter)
#             iono_org = self.reconstruct_ionogram(data_i, apply_amplitude_filter=apply_amplitude_filter)
#             str_org = f'{iono_org.shape[0]}x{iono_org.shape[1]}x{iono_org.shape[2]}'
            
#             # Resample to new grid
#             iono_resampled = self.resample_ionogram(iono_org, Frange, Zrange, output_size)
#             str_res = f'{iono_resampled.shape[0]}x{iono_resampled.shape[1]}x{iono_resampled.shape[2]}'
            
            
#             # Handle image saving
#             if result_path:
#                 iono_resampled_image = Image.fromarray(iono_resampled)
#                 time_str = datetime.strptime(time, "%Y.%m.%d_%H-%M-%S").strftime("%Y%m%d_%H%M")
#                 save_path = os.path.join(result_path, f"{time_str}.png")
#                 iono_resampled_image.save(save_path)

#             # Handle plotting
#             if plot:
#                 time_str = datetime.strptime(time, "%Y.%m.%d_%H-%M-%S").strftime("%Y-%m-%d %H:%M")
#                 fig, ax = plt.subplots(1, 2, width_ratios=[1, 0.6], figsize=(12, 5))
#                 fig.suptitle(time_str, fontsize=17)

#                 # Original ionogram (flipped vertically for displa y)
#                 iono_org_display = Image.fromarray(iono_org).transpose(Image.FLIP_TOP_BOTTOM)
#                 ax[0].imshow(
#                     iono_org_display,
#                     extent=[self.freq_org[0], self.freq_org[-1], self.rang_org[0], self.rang_org[-1]],
#                     aspect='auto'
#                 )
#                 ax[0].set_title("Original Ionogram", fontsize=17)

#                 # Resampled ionogram
#                 ax[1].imshow(
#                     iono_resampled,
#                     extent=[Frange[0], Frange[1], Zrange[0], Zrange[1]],
#                     aspect='auto'
#                 )
#                 ax[1].set_title("Resampled Ionogram", fontsize=17)

#                 # Add legends and labels
#                 for axis in ax:
#                     axis.set_xlabel("Frequency (MHz)", fontsize=13)
#                     axis.set_ylabel("Virtual Altitude (km)", fontsize=13)
#                     axis.legend(handles=[
#                         Patch(color='red', label='O-mode'),
#                         Patch(color='green', label='X-mode')
#                     ], loc='upper right')

#                 plt.tight_layout()
#                 plt.show()
                
                
                
                
                
                
                
                
                

#         print("Processing complete.")
        
    # def plot_single_ionogram(self, ionogram, extent=None, title="Ionogram", flip_vertical=False):
    #     """
    #     Plots a single ionogram, regardless of whether it's the original or resampled version.
        
    #     Parameters:
    #         ionogram (ndarray): The ionogram image array (uint8) to be plotted.
    #         extent (list, optional): Axis extents [xmin, xmax, ymin, ymax]. If not provided, 
    #                                  defaults to:
    #                                   - [self.freq_org[0], self.freq_org[-1], self.rang_org[0], self.rang_org[-1]]
    #                                     for an ionogram with dimensions matching the original ionogram,
    #                                   - [1, 9, 80, 480] for an ionogram with dimensions 81x81,
    #                                   - Otherwise, [0, width, 0, height].
    #         title (str): Title of the plot.
    #         flip_vertical (bool): If True, flips the ionogram vertically before plotting.
    #     """
    #     # Optionally flip the ionogram vertically for display
    #     if flip_vertical:
    #         ionogram = np.array(Image.fromarray(ionogram).transpose(Image.FLIP_TOP_BOTTOM))
        
    #     # Determine the extent if not provided
    #     if extent is None:
    #         if ionogram.shape[0] == len(self.rang_org) and ionogram.shape[1] == len(self.freq_org):
    #             extent = [self.freq_org[0], self.freq_org[-1], self.rang_org[0], self.rang_org[-1]]
    #         elif ionogram.shape[0] == 81 and ionogram.shape[1] == 81:
    #             extent = [1, 9, 80, 480]
    #         else:
    #             extent = [0, ionogram.shape[1], 0, ionogram.shape[0]]
        
    #     plt.figure(figsize=(6, 5))
    #     plt.imshow(ionogram, extent=extent, aspect='auto')
    #     plt.xlabel("Frequency (MHz)", fontsize=15)
    #     plt.ylabel("Virtual Altitude (km)", fontsize=15)
    #     plt.title(title)
        
    #     # Create custom legend patches for O-mode and X-mode
    #     legend_handles = [Patch(color='red', label='O-mode'),
    #                       Patch(color='green', label='X-mode')]
    #     plt.legend(handles=legend_handles, loc='upper right')
    #     plt.tight_layout()
    #     plt.show()














# import numpy as np
# from scipy.interpolate import RegularGridInterpolator
# from PIL import Image
# import matplotlib.pyplot as plt
# from matplotlib.patches import Patch
# import os
# from datetime import datetime

# class IonogramProcessing:
#     def __init__(self):
#         # Original ionogram axes and parameters
#         self.freq_org = np.arange(1, 16 + 0.05, 0.05)  # Frequency axis: 1-16 MHz, step 0.05
#         self.rang_org = np.arange(80, 640 + 5, 5)      # Range axis: 80-640 km, step 5
#         self.I_max = 160  # Maximum amplitude for scaling
#         self.I_min = 1  # Minimum amplitude for scaling
    
    
    
    
    
    
#     def reconstruct_ionogram(self, data_i):
#         """
#         Reconstructs the ionogram to its original dimensions from raw data.
#         Returns the reconstructed ionogram as a uint8 array.
#         """
        
        
#         # Extract data components
#         freq = np.around(data_i[:, 0], decimals=2)  # Frequency values
#         rang = np.around(data_i[:, 1], decimals=2)  # Range values
#         pol = np.round(data_i[:, 2])                # Polarization values
#         amp = data_i[:, 4]                          # Amplitude values
#         ang = np.round(data_i[:, 7])                # Angle values
        
        
#         # Initialize ionogram structure
#         iono_org = np.zeros((len(self.rang_org), len(self.freq_org), 3))
        
#         # Calculate indices for frequency and range
#         F_idx = np.clip(np.searchsorted(self.freq_org, freq), 0, len(self.freq_org)-1)
#         Z_idx = np.clip(np.searchsorted(self.rang_org, rang), 0, len(self.rang_org)-1)
#         I_idx = np.clip(amp, self.I_min, self.I_max)  # Clip amplitudes
        
        
#         # Initialize amplitude mask
#         mask_amp = np.zeros_like(freq, dtype=bool)
#         unique_freqs = np.unique(freq)
        
#         for f in unique_freqs:
#             indices = np.where(freq == f)[0]
#             if len(indices) == 0:
#                 continue
#             current_amps = amp[indices]
#             max_amp_f = np.max(current_amps)
#             threshold = 0.75 * max_amp_f
#             mask_amp[indices] = current_amps >= threshold
            
#         # Apply masks for polarization, angle, and amplitude
#         mask_O = (pol == 90) & (ang == 0) & mask_amp
#         mask_X = (pol == -90) & (ang == 0) & mask_amp
        
#         # # Create masks for O-mode and X-mode
#         # mask_O = (pol == 90) & (ang == 0)
#         # mask_X = (pol == -90) & (ang == 0)
        
#         # Populate O-mode and X-mode data
#         iono_org[Z_idx[mask_O], F_idx[mask_O], 0] = (I_idx[mask_O] - self.I_min) / (self.I_max - self.I_min)
#         iono_org[Z_idx[mask_X], F_idx[mask_X], 1] = (I_idx[mask_X] - self.I_min) / (self.I_max - self.I_min)
        
#         # Normalize and convert to uint8
#         if np.max(iono_org) > 0:
#             iono_org = (iono_org / np.max(iono_org)) * 255
#         else:
#             iono_org = np.zeros_like(iono_org)
#         iono_org = iono_org.astype(np.uint8)
        
#         return iono_org
    
    
    
    
#     def resample_ionogram(self, iono_org, Frange=[1, 9], Zrange=[80, 480], output_size=81):
#         """
#         Resamples the ionogram onto a new grid of specified size.
#         Returns the resampled ionogram as a uint8 array.
#         """
#         # Create new grid coordinates
#         frequency_axis = np.linspace(Frange[0], Frange[1], output_size)
#         range_axis = np.linspace(Zrange[0], Zrange[1], output_size)
#         r, f = np.meshgrid(range_axis, frequency_axis)

#         # Interpolate each mode
#         iono_resampled = np.zeros((output_size, output_size, 3))
#         for mode in range(2):  # Process O-mode and X-mode
#             interpolator = RegularGridInterpolator(
#                 (self.rang_org, self.freq_org), iono_org[:, :, mode],
#                 method='linear', bounds_error=False, fill_value=0
#             )
#             grid_coords = np.array([r.ravel(), f.ravel()]).T
#             iono_resampled[:, :, mode] = interpolator(grid_coords).reshape((output_size, output_size))

#         # Normalize and format
#         if np.max(iono_resampled) > 0:
#             iono_resampled = (iono_resampled / np.max(iono_resampled)) * 255
#         else:
#             iono_resampled = np.zeros_like(iono_resampled)
#         iono_resampled = iono_resampled.astype(np.uint8)
#         iono_resampled = np.rot90(iono_resampled, k=1)  # Rotate for correct orientation

#         return iono_resampled
    
    
    
    
    
#     def process_ionogram(self, data, times, plot=False, result_path=None):
#         """
#         Processes ionograms through reconstruction and resampling.
#         Handles batch processing of multiple ionograms.
#         """
#         # Resampling parameters
#         output_size = 81
#         Frange = [1, 9]
#         Zrange = [80, 480]

#         for i in range(len(data)):
#             time = times[i]
#             data_i = data[i]

#             # Reconstruct original ionogram
#             iono_org = self.reconstruct_ionogram(data_i)
            
#             # Resample to new grid
#             iono_resampled = self.resample_ionogram(iono_org, Frange, Zrange, output_size)

#             # Handle image saving
#             if result_path:
#                 iono_resampled_image = Image.fromarray(iono_resampled)
#                 time_str = datetime.strptime(time, "%Y.%m.%d_%H-%M-%S").strftime("%Y%m%d_%H%M")
#                 save_path = os.path.join(result_path, f"{time_str}.png")
#                 iono_resampled_image.save(save_path)

#             # Handle plotting
#             if plot:
#                 time_str = datetime.strptime(time, "%Y.%m.%d_%H-%M-%S").strftime("%Y-%m-%d_%H:%M")
#                 fig, ax = plt.subplots(1, 2, width_ratios=[1, 0.6], figsize=(12, 5))
#                 fig.suptitle(time_str)
                
                
#                 # Original ionogram (flipped vertically for display)
#                 iono_org_display = Image.fromarray(iono_org).transpose(Image.FLIP_TOP_BOTTOM)
#                 ax[0].imshow(iono_org_display, extent=[self.freq_org[0], self.freq_org[-1], 
#                             self.rang_org[0], self.rang_org[-1]], aspect='auto')
#                 ax[0].set_title("Original Ionogram")
                
#                 # Resampled ionogram
#                 ax[1].imshow(iono_resampled, extent=[Frange[0], Frange[1], Zrange[0], Zrange[1]], 
#                             aspect='auto')
#                 ax[1].set_title("Resampled Ionogram")

#                 # Add legends and labels
#                 for axis in ax:
#                     axis.set_xlabel("Frequency (MHz)")
#                     axis.set_ylabel("Virtual Altitude (km)")
#                     axis.legend(handles=[
#                         Patch(color='red', label='O-mode'),
#                         Patch(color='green', label='X-mode')
#                     ], loc='upper right')

#                 plt.tight_layout()
#                 plt.show()

#         print("Processing complete.")


#     def plot_single_ionogram(self, ionogram, extent=None, title="Ionogram", flip_vertical=False):
#         """
#         Plots a single ionogram, regardless of whether it's the original or resampled version.
        
#         Parameters:
#             ionogram (ndarray): The ionogram image array (uint8) to be plotted.
#             extent (list, optional): Axis extents [xmin, xmax, ymin, ymax]. If not provided, 
#                                      defaults to:
#                                       - [self.freq_org[0], self.freq_org[-1], self.rang_org[0], self.rang_org[-1]]
#                                         for an ionogram with dimensions matching the original ionogram,
#                                       - [1, 9, 80, 480] for an ionogram with dimensions 81x81,
#                                       - Otherwise, [0, width, 0, height].
#             title (str): Title of the plot.
#             flip_vertical (bool): If True, flips the ionogram vertically before plotting.
#         """
#         # Optionally flip the ionogram vertically for display
#         if flip_vertical:
#             ionogram = np.array(Image.fromarray(ionogram).transpose(Image.FLIP_TOP_BOTTOM))
        
#         # Determine the extent if not provided
#         if extent is None:
#             if ionogram.shape[0] == len(self.rang_org) and ionogram.shape[1] == len(self.freq_org):
#                 extent = [self.freq_org[0], self.freq_org[-1], self.rang_org[0], self.rang_org[-1]]
#             elif ionogram.shape[0] == 81 and ionogram.shape[1] == 81:
#                 extent = [1, 9, 80, 480]
#             else:
#                 extent = [0, ionogram.shape[1], 0, ionogram.shape[0]]
        
#         plt.figure(figsize=(6, 5))
#         plt.imshow(ionogram, extent=extent, aspect='auto')
#         plt.xlabel("Frequency (MHz)")
#         plt.ylabel("Virtual Altitude (km)")
#         plt.title(title)
        
#         # Create custom legend patches for O-mode and X-mode
#         legend_handles = [Patch(color='red', label='O-mode'),
#                           Patch(color='green', label='X-mode')]
#         plt.legend(handles=legend_handles, loc='upper right')
#         plt.tight_layout()
#         plt.show()







# import os
# import numpy as np
# from PIL import Image
# import matplotlib.pyplot as plt
# from matplotlib.patches import Patch
# from scipy.interpolate import RegularGridInterpolator
# from datetime import datetime





# class IonogramProcessing:
#     def __init__(self):
#         self.a = 1
    

#     def resample_ionogram(self, iono_org, freq_org, rang_org, f, r, output_size):
#         """
#         Resampling onto an 81x81 grid using RegularGridInterpolator
#         """

#         # Create interpolator for each mode separately
#         iono_resampled = np.zeros((output_size, output_size, 3))
#         for mode in range(2):  # Mode 0 is O-mode, Mode 1 is X-mode
#             interpolator = RegularGridInterpolator(
#                 (rang_org, freq_org), iono_org[:, :, mode], method='linear', bounds_error=False, fill_value=0
#             )
#             # Interpolate on the new grid
#             grid_coords = np.array([r.ravel(), f.ravel()]).T
#             iono_resampled[:, :, mode] = interpolator(grid_coords).reshape((output_size, output_size))

#         # Normalize and convert to uint8
#         iono_resampled = (iono_resampled / np.max(iono_resampled)) * 255
#         iono_resampled = iono_resampled.astype(np.uint8)
#         iono_resampled = np.rot90(iono_resampled, k=1)
        
#         return iono_resampled
    
    
    
#     def process_ionogram(self, data, times, plot=False, result_path=None):
#         """
#         Function for reconstructing ionograms to their original dimensions, then
#         resampling onto a 81x81 grid.
        
#         This function takes in data that has been processed by the "import_data"
#         function which contains ionograms over a whole day.
        
#         Input (type)        | DESCRIPTION
#         ------------------------------------------------
#         data  (np.ndarray)  | ndarray with original ionograms 
#         times (np.ndarray)  | Timestamps of the original ionograms
#         plot  (bool)        | Argument for plotting ionograms 
#         result_path (str)   | Path for saving processed ionograms
#         """
        
#         # Defining ionogram axes
#         freq_org = np.arange(1, 16 + 0.05, 0.05)  # original ionogram freq: [1, 16] MHz
#         rang_org = np.arange(80, 640 + 5, 5)      # original ionogram range: [80, 640] km 
        
#         # Max and min allowed amplitudes
#         I_max = 75
#         I_min = 20
        
#         # Prepare output grid for resampling
#         output_size = 81
#         Frange = [1, 9]  # Frequency range of interest
#         Zrange = [80, 480]  # Range of interest
#         frequency_axis = np.linspace(Frange[0], Frange[1], output_size)
#         range_axis = np.linspace(Zrange[0], Zrange[1], output_size)
#         r, f = np.meshgrid(range_axis, frequency_axis)
        
    
#         # Process each ionogram
#         for i in np.arange(0, len(data)):
#             time = times[i]
#             data_i = data[i]
            
#             # print(f'  - Processing ionogram at time {time}')
            
#             """ Reconstructing ionograms to original dimensions"""
#             freq = np.around(data_i[:, 0], decimals=2)     # frequencies [MHz]
#             rang = np.around(data_i[:, 1], decimals=2)     # ionosonde vertual range [km]
#             pol  = np.round(data_i[:, 2])                  # polarization (either 90 or -90 degrees)
#             amp  = data_i[:, 4]                            # backscatter amplitude
#             dop  = np.around(data_i[:, 5], decimals=2)     # doppler shift
#             ang  = np.round(data_i[:, 7])                  # received angle [deg]
    
#             """ Recreate ionogram """
            
#             iono_org = np.zeros((len(rang_org), len(freq_org), 3))
            
#             # Finding indices and ensuring they stay within valid bounds
#             F_idx = np.clip(np.searchsorted(freq_org, freq), 0, len(freq_org) - 1)
#             Z_idx = np.clip(np.searchsorted(rang_org, rang), 0, len(rang_org) - 1)
#             I_idx = np.clip(amp, I_min, I_max)  # only interested in amp: [21, 75]
            
#             mask_O = (pol == 90) & (ang == 0)   # mask for positive 90 deg pol values (O-mode) and 0 deg ang values
#             mask_X = (pol == -90) & (ang == 0)  # mask for positive -90 deg pol values (X-mode) and 0 deg ang values
    
#             # O and X-mode without doppler shift
#             iono_org[Z_idx[mask_O], F_idx[mask_O], 0] = (I_idx[mask_O] - I_min) / (I_max - I_min)  # Scale amplitude 0 to 1
#             iono_org[Z_idx[mask_X], F_idx[mask_X], 1] = (I_idx[mask_X] - I_min) / (I_max - I_min)  # Scale amplitude 0 to 1
            
            
#             # Normalize the ionogram data
#             iono_org = (iono_org / np.max(iono_org)) * 255  # multiplying by 255 for image purposes
#             iono_org = iono_org.astype(np.uint8)
            
            
#             iono_resampled = self.resample_ionogram(iono_org, freq_org, rang_org, f, r, output_size)
            
            
            
            
#             iono_org = Image.fromarray(iono_org).transpose(Image.FLIP_TOP_BOTTOM)
#             iono_image = Image.fromarray(iono_resampled)
            
            
#             # Save the resampled image if a result path is provided
#             if result_path:
#                 iono_resampled_image = Image.fromarray(iono_resampled)
                
#                 # Format the timestamp into yyyyMMdd_hhmm
#                 time_str = datetime.strptime(time, "%Y.%m.%d_%H-%M-%S").strftime("%Y%m%d_%H%M")
#                 save_filename = os.path.join(result_path, f"{time_str}.png")
                
#                 iono_resampled_image.save(save_filename)
#                 # print(f"  - Saved ionogram image to {save_filename}")
            
#             if plot:
#                 fig, ax = plt.subplots(1, 2, width_ratios=[1, 0.6], figsize=(12, 5))
                
#                 # Original ionogram
#                 ax[0].imshow(iono_org, extent=[freq_org[0], freq_org[-1], rang_org[0], rang_org[-1]],
#                                     aspect='auto')
#                 ax[0].set_title("Original Ionogram", fontsize=20)
#                 ax[0].set_xlabel("Frequency (MHz)", fontsize=15)
#                 ax[0].set_ylabel("Virtual Altitude (km)", fontsize=15)
                
                
#                 # Resampled ionogram
#                 ax[1].imshow(iono_resampled, extent=[Frange[0], Frange[1], Zrange[0], Zrange[1]],
#                                     aspect='auto')
#                 ax[1].set_title("Resampled Ionogram", fontsize=20)
#                 ax[1].set_xlabel("Frequency (MHz)", fontsize=15)
#                 ax[1].set_ylabel("Virtual Altitude (km)", fontsize=15)
                
                
#                 # Custom labels with filled squares
#                 green_patch = Patch(color='green', label='X-mode')
#                 red_patch = Patch(color='red', label='O-mode')
                
#                 # Add legends
#                 for axis in ax:
#                     axis.legend(handles=[red_patch, green_patch], loc='upper right', title="Modes", frameon=True)
                
                
#                 plt.tight_layout()
#                 plt.show()
            
    
#         print("Processing complete.")














