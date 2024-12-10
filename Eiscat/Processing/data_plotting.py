# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 17:16:54 2024

@author: Kian Sartipzadeh
"""


class EISCATPlotter:
    def __init__(self, X_EISCAT):
        self.X_EISCAT = X_EISCAT
        self.selected_indices = []
    
    
    def plot_day(self, day):
        
        X_eis_day = self.X_EISCAT[day]
        
        r_time = from_array_to_datetime(X_eis_day["r_time"])
        r_h = X_eis_day["r_h"].flatten()
        
        ne_eis = X_eis_day["r_param"]
        ne_err = X_eis_day["r_error"]
        
        
        
        
        
        # date_str = r_time[0].strftime('%Y-%m-%d')
        
        


# def plot(data):
#     """
#     Plot a comparison of original and averaged data using pcolormesh.

#     Input (type)                 | DESCRIPTION
#     ------------------------------------------------
#     original_data (dict)         | Dictionary containing the original data.
#     """
    
#     # Convert time arrays to datetime objects
#     r_time = np.array([datetime(year, month, day, hour, minute, second) 
#                             for year, month, day, hour, minute, second in data['r_time']])
#     r_h = data['r_h']
#     r_param = data['r_param']
#     r_error = data['r_error']
    
#     # Date
#     date_str = r_time[0].strftime('%Y-%m-%d')
    
#     # Creating the plots
#     fig, ax = plt.subplots(1, 2, figsize=(10, 8), sharey=True)
#     fig.suptitle(f'Date: {date_str}', fontsize=15)
#     fig.tight_layout()
    
    
#     # Plotting original data
#     pcm_ne = ax[0].pcolormesh(r_time, r_h.flatten(), np.log10(r_param), shading='auto', cmap='turbo', vmin=9, vmax=12)
#     ax[0].set_title('EISCAT UHF', fontsize=17)
#     ax[0].set_xlabel('Time [hours]')
#     ax[0].set_ylabel('Altitude [km]')
#     ax[0].xaxis.set_major_formatter(DateFormatter('%H:%M'))
#     fig.autofmt_xdate()
    
#     # Add colorbar for the original data
#     # cbar = fig.colorbar(pcm_ne, ax=ax[0], orientation='vertical', fraction=0.03, pad=0.04, aspect=20, shrink=1)
#     # cbar.set_label('log10(n_e) [g/cm^3]')
    
#     # fig.autofmt_xdate()
    
#     # Plotting original data
#     pcm_err = ax[1].pcolormesh(r_time, r_h.flatten(), np.log10(r_error), shading='auto', cmap='turbo', vmin=9, vmax=12)
#     ax[1].set_title('Measurement Error', fontsize=17)
#     ax[1].set_xlabel('Time [hours]')
#     # ax[1].set_ylabel('Altitude [km]')
#     ax[1].xaxis.set_major_formatter(DateFormatter('%H:%M'))
#     # fig.autofmt_xdate()
    
#     # Add colorbar for the original data
#     cbar = fig.colorbar(pcm_err, ax=ax[1], orientation='vertical', fraction=0.03, pad=0.04, aspect=44, shrink=3)
#     cbar.set_label(r'$log_{10}(n_e)$ [g/cm$^3$]', fontsize=17)
    
    
#     # Display the plots
#     plt.show()











