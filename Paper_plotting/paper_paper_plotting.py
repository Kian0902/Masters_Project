"""
Created on Mon Jan 27 16:23:19 2025

@author: Kian Sartipzadeh
"""


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, Normalize, LogNorm
from matplotlib.dates import DateFormatter
from matplotlib.patches import Patch
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
from matplotlib.widgets import Cursor
from datetime import datetime

from paper_utils import from_array_to_datetime, merge_nested_dict, merge_nested_pred_dict, inspect_dict

# import seaborn as sns
# For all plots: sns.set(style="dark", context=None, palette=None)
# For single plot: with sns.axes_style("dark"):








class PaperPlotter:
    def __init__(self, X_EIS, X_KIAN, X_ION, X_GEO):#, X_ART, X_ECH):
        self.X_EIS = X_EIS
        self.X_KIAN = X_KIAN
        self.X_ION = X_ION
        self.X_GEO = X_GEO
        # self.X_ART = X_ART
        # self.X_ECH = X_ECH
        # self.selected_indices = []
    
    
    def relative_error(self, X1, X2):
        err =  (X2 - X1)/X1
        return err
    
    
    def plot_compare_all(self):
        
        # ___________ Getting Data ___________
        
        # Merging all global keys
        x_eis = merge_nested_dict(self.X_EIS)['All']
        x_kian = merge_nested_pred_dict(self.X_KIAN)['All']
        x_ion = merge_nested_pred_dict(self.X_ION)['All']
        x_geo = merge_nested_pred_dict(self.X_GEO)['All']
        
        r_time = from_array_to_datetime(x_eis['r_time'])
        r_h = x_eis['r_h'].flatten()
        
        # Ne-profiles
        ne_eis  = x_eis["r_param"]
        ne_kian = 10**x_kian["r_param"]
        ne_ion  = 10**x_ion["r_param"]
        ne_geo  = 10**x_geo["r_param"]
        
        
        date_str = r_time[0].strftime('%b-%Y')
        
        
        
        # ___________ Defining axes ___________
        fig = plt.figure(figsize=(14, 8))
        fig.suptitle(f'Comparrison Between Prediction Models and Ground Truth\nDate: {date_str}', fontsize=17, y=1.01)
        
        gs = GridSpec(4, 2, width_ratios=[1, 0.015], wspace=0.1, hspace=0.3)
        ax0 = fig.add_subplot(gs[0, 0])
        ax1 = fig.add_subplot(gs[1, 0], sharex=ax0)
        ax2 = fig.add_subplot(gs[2, 0], sharex=ax0)
        ax3 = fig.add_subplot(gs[3, 0], sharex=ax0)
        cax = fig.add_subplot(gs[:, 1])
        
        
        
        #  ___________ Creating Plots  ___________
        MIN, MAX = 1e10, 1e12
        
        ne = ax0.pcolormesh(r_time, r_h, ne_eis, shading='gouraud', cmap='turbo', norm=LogNorm(vmin=MIN, vmax=MAX))
        ax1.pcolormesh(r_time, r_h, ne_kian, shading='gouraud', cmap='turbo', norm=LogNorm(vmin=MIN, vmax=MAX))
        ax2.pcolormesh(r_time, r_h, ne_ion, shading='gouraud', cmap='turbo', norm=LogNorm(vmin=MIN, vmax=MAX))
        ax3.pcolormesh(r_time, r_h, ne_geo, shading='gouraud', cmap='turbo', norm=LogNorm(vmin=MIN, vmax=MAX))
        
        
        # Set titles and labels
        ax0.set_title('(a) EISCAT UHF', fontsize=16)
        ax1.set_title('(b) KIAN-Net', fontsize=16)
        ax2.set_title('(c) Iono-CNN', fontsize=16)
        ax3.set_title('(d) Geo-DMLP', fontsize=16)
        
        # y-labels
        ax0.set_ylabel('Altitude [km]', fontsize=13)
        ax1.set_ylabel('Altitude [km]', fontsize=13)
        ax2.set_ylabel('Altitude [km]', fontsize=13)
        ax3.set_ylabel('Altitude [km]', fontsize=13)
        
        # x-label
        ax3.set_xlabel('UT [dd hh:mm]', fontsize=13)
        
        # Ticks
        ax0.tick_params(labelbottom=False)  # Hide x-ticks on EISCAT plot
        ax1.tick_params(labelbottom=False)
        ax2.tick_params(labelbottom=False)
        ax3.xaxis.set_major_formatter(DateFormatter('%d %H:%M'))
        
        plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='center')
        
        # Add colorbar
        cbar = fig.colorbar(ne, cax=cax, orientation='vertical')
        cbar.set_label('$log_{10}$ $N_e$  (m$^{-3}$)', fontsize=13)
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
#     def plot_vertical_compare_ne(self):
#         """
#         Function for comparing EISCAT UHF to KIAN-Net, Artist 4.5, and IRI in a 4x1 vertical grid.
#         """
#         r_time = from_array_to_datetime(self.X_EISCAT["r_time"])
#         r_h = self.X_EISCAT["r_h"].flatten()
        
#         # Linear scaling for electron density
#         ne_eis = self.X_EISCAT["r_param"]
#         ne_kian = 10**self.X_KIAN["r_param"]
#         ne_art = self.X_Artist["r_param"]
#         ne_iri = self.X_IRI["r_param"]
        
#         date_str = r_time[0].strftime('%Y-%m-%d')
        
#         # Figure
#         fig = plt.figure(figsize=(6, 10))
#         fig.suptitle(f'Date: {date_str}', fontsize=17, y=0.98)
        
#         gs = GridSpec(4, 2, width_ratios=[1, 0.05], height_ratios=[1, 1, 1, 1], hspace=0.3, wspace=0.05)
        
#         # Axes for plots
#         ax0 = fig.add_subplot(gs[0, 0])   # EISCAT UHF
#         ax1 = fig.add_subplot(gs[1, 0], sharex=ax0, sharey=ax0)  # KIAN-Net
#         ax2 = fig.add_subplot(gs[2, 0], sharex=ax0, sharey=ax0)  # Artist 4.5
#         ax3 = fig.add_subplot(gs[3, 0], sharex=ax0, sharey=ax0)  # IRI
#         cax = fig.add_subplot(gs[:, 1])   # Shared Colorbar
        
#         # Global settings
#         MIN, MAX = 1e10, 1e12
#         subtit_size, xlab_size, ylab_size, cbar_size = 17, 13, 13, 13
        
#         # Plot EISCAT UHF
#         ne_EISCAT = ax0.pcolormesh(r_time, r_h, ne_eis, shading='auto', cmap='turbo', norm=LogNorm(vmin=MIN, vmax=MAX))
#         ax0.set_title('EISCAT UHF', fontsize=subtit_size)
#         ax0.set_ylabel('Altitude [km]', fontsize=ylab_size)
#         ax0.xaxis.set_major_formatter(DateFormatter('%H:%M'))
#         ax0.tick_params(labelbottom=False)
        
#         # Plot KIAN-Net
#         ax1.pcolormesh(r_time, r_h, ne_kian, shading='auto', cmap='turbo', norm=LogNorm(vmin=MIN, vmax=MAX))
#         ax1.set_title('KIAN-Net', fontsize=subtit_size)
#         ax1.set_ylabel('Altitude [km]', fontsize=ylab_size)
#         ax1.xaxis.set_major_formatter(DateFormatter('%H:%M'))
#         ax1.tick_params(labelbottom=False)
        
#         # Plot Artist 4.5
#         ax2.pcolormesh(r_time, r_h, ne_art, shading='auto', cmap='turbo', norm=LogNorm(vmin=MIN, vmax=MAX))
#         ax2.set_title('Artist 4.5', fontsize=subtit_size)
#         ax2.set_ylabel('Altitude [km]', fontsize=ylab_size)
#         ax2.xaxis.set_major_formatter(DateFormatter('%H:%M'))
#         ax2.tick_params(labelbottom=False)
        
#         # Plot IRI
#         ax3.pcolormesh(r_time, r_h, ne_iri, shading='auto', cmap='turbo', norm=LogNorm(vmin=MIN, vmax=MAX))
#         ax3.set_title('IRI', fontsize=subtit_size)
#         ax3.set_xlabel('Time [UT]', fontsize=xlab_size)
#         ax3.set_ylabel('Altitude [km]', fontsize=ylab_size)
#         ax3.xaxis.set_major_formatter(DateFormatter('%H:%M'))
        
#         # Shared colorbar
#         cbar = fig.colorbar(ne_EISCAT, cax=cax, orientation='vertical')
#         cbar.set_label('$n_e$ [n/m$^3$]', fontsize=cbar_size)
        
#         # Rotate x-axis labels
#         for ax in [ax0, ax1, ax2, ax3]:
#             plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='center')
        
#         plt.show()

    
    
    
#     def plot_compare_error(self):
#         """
#         Function for comparing relative errors between EISCAT UHF, KIAN-Net, Artist 4.5 and IRI.
#         """
#         r_time = from_array_to_datetime(self.X_EISCAT["r_time"])
#         r_h = self.X_EISCAT["r_h"].flatten()
        
#         # Linear scaling for electron density
#         ne_eis = self.X_EISCAT["r_param"]
#         ne_kian = 10**self.X_KIAN["r_param"]
#         ne_art = self.X_Artist["r_param"]
#         ne_iri = self.X_IRI["r_param"]
        
        
#         err_kian = self.relative_error(ne_eis, ne_kian)
#         err_art = self.relative_error(ne_eis, ne_art)
#         err_iri = self.relative_error(ne_eis, ne_iri)
        
        
#         # date_str = r_time[0].strftime('%Y-%m-%d')

#         # Figure 
#         fig = plt.figure(figsize=(12, 5))
#         gs = GridSpec(1, 4, width_ratios=[1, 1, 1, 0.05], wspace=0.1)
        
        
#         # ___________ Defining axes ___________
#         # 1st row
#         ax00 = fig.add_subplot(gs[0, 0])                    # KIAN-Net
#         ax01 = fig.add_subplot(gs[0, 1], sharey=ax00)       # Artist 4.5
#         ax02 = fig.add_subplot(gs[0, 2], sharey=ax00)       # IRI
#         cax03 = fig.add_subplot(gs[0, 3])                   # Colorbar
        
        
#         #  ___________ Creating Plots  ___________
#         MIN, MAX = -1, 1
#         subtit_size, xlab_size, ylab_size, cbar_size = 17, 13, 13, 13
        
#         # Error KIAN-Net
#         error_kian = ax00.pcolormesh(r_time, r_h, err_kian, shading='auto', cmap='bwr', vmin=MIN, vmax=MAX)
#         ax00.set_title('KIAN-Net vs EISCAT', fontsize=subtit_size)
#         ax00.set_xlabel('Time [UT]', fontsize=xlab_size)
#         ax00.set_ylabel('Altitude [km]', fontsize=ylab_size)
#         ax00.xaxis.set_major_formatter(DateFormatter('%H:%M'))
        
#         # Error Artist 4.5
#         err_art_nan = np.zeros(err_art.shape)  # For grey color
#         ax01.pcolormesh(r_time, r_h, err_art_nan, shading='auto', cmap='gray', vmin=MIN, vmax=MAX, zorder=0)
#         ax01.pcolormesh(r_time, r_h, err_art, shading='auto', cmap='bwr', vmin=MIN, vmax=MAX, zorder=2)
#         ax01.set_title('Artist 4.5 vs EISCAT', fontsize=subtit_size)
#         ax01.set_xlabel('Time [UT]', fontsize=xlab_size)
#         ax01.xaxis.set_major_formatter(DateFormatter('%H:%M'))
#         ax01.tick_params(labelleft=False)
        
        
#         # Error IRI
#         ax02.pcolormesh(r_time, r_h, err_iri, shading='auto', cmap='bwr', vmin=MIN, vmax=MAX)
#         ax02.set_title('IRI vs EISCAT', fontsize=subtit_size)
#         ax02.set_xlabel('Time [UT]', fontsize=xlab_size)
#         ax02.xaxis.set_major_formatter(DateFormatter('%H:%M'))
#         ax02.tick_params(labelleft=False)
        
#         # Colorbar
#         cbar03 = fig.colorbar(error_kian, cax=cax03, orientation='vertical')
#         cbar03.set_label('Relative Error', fontsize=cbar_size)
        
    
#         # Rotate x-axis labels
#         for ax in [ax00, ax01, ax02]:
#             plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='center')
        

#         plt.show()


    
# class AdvancedPaperPlotter:
#     def __init__(self, X_EISCAT, X_KIAN, X_Artist, X_IRI, X_Ionogram, X_GEO):
#         self.X_EISCAT = X_EISCAT
#         self.X_KIAN = X_KIAN
#         self.X_Artist = X_Artist
#         self.X_IRI = X_IRI
#         self.X_Ionogram = X_Ionogram
#         self.X_GEO = X_GEO
#         # self.selected_days = []
    
#     def plot_vertical_compare_3days_ne(self, selected_days):
#         """
#         Function for comparing EISCAT UHF to KIAN-Net, Artist 4.5, and IRI over 3 selected days
#         in a 4x3 grid where each column represents a day, with a color bar per row on the far right.
#         """
        
#         fig = plt.figure(figsize=(12, 15))
#         fig.suptitle('Comparison Over 3 Days', fontsize=20, y=0.96)
        
#         gs = GridSpec(4, 4, width_ratios=[1, 1, 1, 0.05], height_ratios=[1, 1, 1, 1], hspace=0.3, wspace=0.2)
        
#         # Global settings
#         MIN, MAX = 1e10, 1e12
#         subtit_size, xlab_size, ylab_size, cbar_size = 17, 13, 13, 13
        
#         axes = []
#         model_pcolormeshes = [None] * 4  # To store pcolormesh objects for each row
        
#         for i, date in enumerate(selected_days):
#             r_time = from_array_to_datetime(self.X_EISCAT[date]["r_time"])
#             r_h = self.X_EISCAT[date]["r_h"].flatten()
            
#             ne_eis = self.X_EISCAT[date]["r_param"]
#             ne_kian = 10**self.X_KIAN[date]["r_param"]
#             ne_art = self.X_Artist[date]["r_param"]
#             ne_iri = self.X_IRI[date]["r_param"]
            
#             # Axes for plots
#             ax0 = fig.add_subplot(gs[0, i])   # EISCAT UHF
#             ax1 = fig.add_subplot(gs[1, i], sharex=ax0, sharey=ax0)  # KIAN-Net
#             ax2 = fig.add_subplot(gs[2, i], sharex=ax0, sharey=ax0)  # Artist 4.5
#             ax3 = fig.add_subplot(gs[3, i], sharex=ax0, sharey=ax0)  # IRI
            
#             axes.extend([ax0, ax1, ax2, ax3])
            
#             # Plot EISCAT UHF
#             ne_EISCAT = ax0.pcolormesh(r_time, r_h, ne_eis, shading='auto', cmap='turbo', norm=LogNorm(vmin=MIN, vmax=MAX))
#             ax0.set_title(f'{date}\nEISCAT UHF', fontsize=subtit_size, fontweight='bold')
#             ax0.tick_params(labelbottom=False)
#             if i == 0:
#                 model_pcolormeshes[0] = ne_EISCAT  # Save for colorbar
            
#             # Plot KIAN-Net
#             ne_kian_plot = ax1.pcolormesh(r_time, r_h, ne_kian, shading='auto', cmap='turbo', norm=LogNorm(vmin=MIN, vmax=MAX))
#             ax1.set_title('KIAN-Net', fontsize=subtit_size, fontweight='bold')
#             ax1.tick_params(labelbottom=False)
#             if i == 0:
#                 model_pcolormeshes[1] = ne_kian_plot
            
#             # Plot Artist 4.5
#             ne_art_plot = ax2.pcolormesh(r_time, r_h, ne_art, shading='auto', cmap='turbo', norm=LogNorm(vmin=MIN, vmax=MAX))
#             ax2.set_title('Artist 4.5', fontsize=subtit_size, fontweight='bold')
#             ax2.tick_params(labelbottom=False)
#             if i == 0:
#                 model_pcolormeshes[2] = ne_art_plot
            
#             # Plot IRI
#             ne_iri_plot = ax3.pcolormesh(r_time, r_h, ne_iri, shading='auto', cmap='turbo', norm=LogNorm(vmin=MIN, vmax=MAX))
#             ax3.set_title('IRI', fontsize=subtit_size, fontweight='bold')
#             ax3.set_xlabel('Time [UT]', fontsize=xlab_size)
#             ax3.xaxis.set_major_formatter(DateFormatter('%H:%M'))
#             if i == 0:
#                 model_pcolormeshes[3] = ne_iri_plot
            
#             # Set y-labels for the first column only
#             if i == 0:
#                 ax0.set_ylabel('Altitude [km]', fontsize=ylab_size)
#                 ax1.set_ylabel('Altitude [km]', fontsize=ylab_size)
#                 ax2.set_ylabel('Altitude [km]', fontsize=ylab_size)
#                 ax3.set_ylabel('Altitude [km]', fontsize=ylab_size)
#             else:
#                 ax0.tick_params(labelleft=False)
#                 ax1.tick_params(labelleft=False)
#                 ax2.tick_params(labelleft=False)
#                 ax3.tick_params(labelleft=False)
        
#         # Add colorbars for each row
#         for row in range(4):
#             cax = fig.add_subplot(gs[row, 3])
#             cbar = fig.colorbar(model_pcolormeshes[row], cax=cax, orientation='vertical')
#             cbar.set_label('$n_e$ [n/m$^3$]', fontsize=cbar_size)
        
#         # Rotate x-axis labels
#         for ax in axes:
#             plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='center')
        
#         plt.show()
    
