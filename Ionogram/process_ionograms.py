# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 14:32:00 2023

@author: Kian Sartipzadeh
"""


import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from PIL import Image




def IonogramProcessing(data, times, plot, result_path=None):
    """
    Function for reconstructing ionograms to their
    original dimensions, then resampling onto a
    81x81 grid.
    
    
    Parameters  |   TYPE   | DESCRIPTION
    ------------------------------------------------
    data        | np array | Array with original ionograms 
    times       | np array | Timestamps of the original ionograms
    data        |   Bool   | Argument for plotting ionograms 
    result_path |   str    | Path for Saving processed ionograms

    """
    
    for i in np.arange(0, len(data)):
        print(i, len(times))
        
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
            
        
        
        


def ImportData(datapath):
    """
    Function for importing ionogram data processed
    by SAO explorer in form of text files.
    
    Parameters  |   TYPE   | DESCRIPTION
    ------------------------------------------------
    datapath    |   str    | Path to folder that contains original ionograms 
    
    Return         |   TYPE   | DESCRIPTION
    ------------------------------------------------
    ionogram_data  | np array | Original ionogram data
    ionogramTime   | np array | Timstamps of ionogram
    
    """
    
    ionogramTime = []
    data = []
    with open(datapath, "r") as file:
        
        # Reading all lines in txt file
        lines = file.readlines()
        
        data_temp = []
        for line in lines:
            if len(line) == 30:
                
                iono_date = line[0:10]
                iono_time = f"{line[-13:-11]}-{line[-10:-8]}-{line[-7:-5]}"

                iono_datetime = f"{iono_date}_{iono_time}"
                
                ionogramTime.append(iono_datetime)
            
            # Condition for if encounter space
            if len(line) == 46:
                
                ll = line.split() # Splitting strings in line elementwise e.g., ["Hello There"] to ["Hiello", "There"]

                l = [float(x) for x in ll] # Converting strings to floats
                
                data_temp.append(l)
            
            
            if len(line) == 1:
                data.append(np.array(data_temp))
                data_temp = []
            
            else:
                continue
    
    ionogram_data = np.array(data, dtype=object)
    
    return ionogram_data, ionogramTime







resultpath = "C:\\Users\\kian0\\OneDrive\\Desktop\\FYS-3730 Project Paper\\training\\justmadeiono"
datapath_folder = "C:\\Users\\kian0\\OneDrive\\Desktop\\FYS-3730 Project Paper\\FromGRMtotxt\\Training Data"

i=0
for file in os.listdir(datapath_folder):
    
    print(i)
    
    file_path = os.path.join(datapath_folder, file)

    data, times = ImportData(file_path)

    IonogramProcessing(data, times, plot=True)
      
    i+=1




