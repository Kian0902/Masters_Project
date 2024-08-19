# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 14:32:00 2023

@author: Kian Sartipzadeh
"""


"""
To do:
    - Understand the each step in "ionogram_processing".
    - Find ways of writing it more efficiently.
    - Access whether or not to make a class.
    - Maybe two seperate scripts?
"""







import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from PIL import Image




def ionogram_processing(data, times, plot=False, result_path=None):
    """
    Function for reconstructing ionograms to their
    original dimensions, then resampling onto a
    81x81 grid.
    
    
    Input (type) | DESCRIPTION
    ------------------------------------------------
    data  (np.ndarray)  | Array with original ionograms 
    times (np.ndarray)  | Timestamps of the original ionograms
    plot  (bool)        | Argument for plotting ionograms 
    result_path (str)   | Path for Saving processed ionograms

    """
    
    
    for i in np.arange(0, 1):

        time = times[i]
        test = data[i]
        
        """ Reconstructing ionograms to original dimensions"""
        # 1 Read Data
        freq = np.round(test[:, 0]*20)/20
        rang = np.round(test[:, 1]*5)/5
        pol  = test[:, 2]
        ion  = test[:, 4]
        ang  = test[:, 7]
        
        
        # 2 Recreate original ionogram
        freq_org = np.round(np.arange(1, 16 + 0.05, 0.05)*20)/20
        rang_org = np.round(np.arange(80, 640 + 5, 5)*5)/5
        
        length_freq = len(freq_org)
        length_rang = len(rang_org)
        
        I_min = 20
        I_max = 75
        
        
        iono_org = np.zeros((length_rang, length_freq))
        
        # Finding indices of ionogram data that is close to custom array
        for i in range(0, len(freq)):
            F_idx =  np.where(np.isclose(freq_org, freq[i]))[0]
            Z_idx =  np.where(np.isclose(rang_org, rang[i]))[0]
            P_idx =  np.round(pol[i])
            I_idx =  ion[i]
            A_idx =  np.round(ang[i])
            if I_idx > I_max:
                I_idx = 75
            elif I_idx < 21:
                I_idx = 21
            
            if A_idx == 0:
                if P_idx == 90:
                    iono_org[Z_idx, F_idx] = (I_idx - I_min)/(I_max-I_min)
            
        iono_org = (iono_org / np.max(iono_org)) * 255
        
        
        
        
        
        
        
        
        
        
        """ Resampling Ionograms on 81x81 grid"""
        
        # 3 Create an ionogram with reduced size
        OutputSize = 81
        Frange = [1, 9]
        Zrange = [80, 480]
        
        
        frequency  =  np.linspace(Frange[0], Frange[1], OutputSize)
        range_vals =  np.linspace(Zrange[0], Zrange[1], OutputSize)
        
        xg, yg = np.meshgrid(rang_org, freq_org)
        
        r, f = np.meshgrid(range_vals, frequency)
        

        # Resampling onto grid
        iono = np.uint8(griddata((xg.flatten(), yg.flatten()), iono_org.T.flatten(), (r.flatten(), f.flatten())).reshape(81, 81).T)
  
        
        ionogram = np.uint8((iono / np.max(iono)) * 255)
        
        

        # Convert the ionogram data to a Pillow image
        ionogram_image = Image.fromarray(ionogram)
        ionogram_image = ionogram_image.transpose(Image.FLIP_TOP_BOTTOM)
        
        if plot == True:

            plt.imshow(ionogram_image, cmap='gray')
            plt.axis("off")
            plt.show()
        
        
        
        if result_path is not None:
            iono_name = os.path.join(result_path, f"{time}.png")
            
            # Save the image to the specified file path
            ionogram_image.save(iono_name)
            
        
        
        


def import_data(datapath: str):
    """
    This function handles ionogram data in form of text files that has been
    pre-processed by the "SAO explorer" software.
    
    Each of these text files consist of 24-hour worth of ionosonde measurements
    with 15 minutes interval per data update. In other words, each 15 min
    interval has a time and date header followed by the ionosonde measurements.
    Each measurement (one row) has 8 ionosonde features represented as the
    columns. The features are: [Freq  Range  Pol  MPA  Amp  Doppler  Az  Zn].
    
    The number of measurements (rows) per "batch" changes depending on whether
    or not the Ionosonde was able to receive a backscatter signal. So each
    "batch" can contain different number of rows.
    
    
    Input (type)    | DESCRIPTION
    ------------------------------------------------
    datapath (str)  | Path to folder that contains original ionograms txt files
    
    Return (type)              | DESCRIPTION
    ------------------------------------------------
    ionogram_data (np.ndarray) | Procrssed ionogram data
    ionogram_time (np.ndarray) | Timestamps of ionogram data
    """
    
    ionogram_time = []
    ionogram_data = []
    with open(datapath, "r") as file:
        
        lines = file.readlines() # Reading all lines in txt file

        data_batch = []
        for line in lines:
            
            """ # When encountering new header containing date and time (Ex: "2018.09.21 (264) 00:00:00.000") """
            if len(line) == 30:                                               # length of header containing date and time which is 30
                iono_date = line[0:10]                                        # length of date (Ex: "2018.09.21" has length=10)
                iono_time = f"{line[-13:-11]}-{line[-10:-8]}-{line[-7:-5]}"   # defining new time format (Ex: 20-15-00)
                iono_datetime = f"{iono_date}_{iono_time}"                    # changing the format to be "yyyy.MM.dd_hh-mm-ss"
                ionogram_time.append(iono_datetime)
            
            
            """ When encountering ionogram data (Ex: 3.400  315.0  90  24  33  -1.172 270.0  30.0) """
            if len(line) == 46:                             # length of each line containing ionogram values which is 46
                line_split = line.split()                   # splitting strings in line by the whitespace between values e.g., ["3.14 0.4"] to ["3.14", "0.4"]
                line_final= [float(x) for x in line_split]  # Converting strings to floats
                data_batch.append(line_final)
            
            
            """ When encountering space between each batch of 15 min data """
            if len(line) == 1:                              # length of whitespace which is 1
                ionogram_data.append(np.array(data_batch))  # appending the "batch" to the total data list 
                data_batch = []                             # resetting the batch list 
            
            else:
                continue
    
    # Converting list into np.ndarrays
    ionogram_time = np.array(ionogram_time, dtype=object)
    ionogram_data = np.array(ionogram_data, dtype=object)
    return ionogram_time, ionogram_data







# resultpath = "justmadeiono"
datapath_folder = "ionograms_txt_data"

i=0
for file in os.listdir(datapath_folder):
    
    # print(i)
    
    file_path = os.path.join(datapath_folder, file)

    times, data = import_data(file_path)
    
    # print(data.shape, times.shape)
    
    
    ionogram_processing(data, times)
      
    i+=1

    break


