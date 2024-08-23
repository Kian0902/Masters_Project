# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 12:12:44 2024

@author: kian0
"""






import os
import pickle
from scipy.io import loadmat

import numpy as np

# Main path to Folder containing EISCAT data
global_folder_path = "C:\\Users\\kian0\\OneDrive\\Desktop\\UiT Courses\\FYS_3931\\Scripts\\Masters_Project\\Eiscat\\Processing\\EISCAT_Ne"
                   

# Use a list comprehension to get subfolder names.
# Each subfolder contains data taken for one experiment at a specific day
global_folder_name = "EISCAT_Ne"
local_subfolder_names = [os.path.join(global_folder_name, subdir) for root, dirs, files in os.walk(global_folder_path) for subdir in dirs]






dataset = {}


iteration = 0
for subfolder in local_subfolder_names:
    
    # Looping and appending through EISCAT folder containing the matlab data files
    Files = [filename.path for filename in os.scandir(subfolder) if (filename.is_file()) and (filename.path[-3:] == 'mat')]
    
    day = os.path.basename(subfolder)  # Adjusted to properly extract the day part from subfolder name
    dataset[day] = {'r_time': [], 'r_h': [], 'r_param': [], 'r_error': []}
    
    print(f'Processing day: {day}')
    
    for file in Files:
        data = loadmat(file)
        
        # Extract the arrays and ensure they are in the correct shape
        r_time = data['r_time'].reshape(-1, 1)  # Ensuring it's (N, 1)
        r_h = data['r_h'].reshape(-1, 1)
        r_param = data['r_param'][:,0].reshape(-1, 1)
        r_error = data['r_error'][:,0].reshape(-1, 1)
        
        # Append each to the list
        dataset[day]['r_time'].append(r_time)
        dataset[day]['r_h'].append(r_h)
        dataset[day]['r_param'].append(r_param)
        dataset[day]['r_error'].append(r_error)
        
        print(r_h.shape)
        print(r_param.shape)
        print(r_error.shape)
        
        break
    # Convert lists to numpy arrays, concatenating along the time axis
    dataset[day]['r_time'] = np.concatenate(dataset[day]['r_time'], axis=1)
    dataset[day]['r_h'] = np.concatenate(dataset[day]['r_h'], axis=1)
    dataset[day]['r_param'] = np.concatenate(dataset[day]['r_param'], axis=1)
    dataset[day]['r_error'] = np.concatenate(dataset[day]['r_error'], axis=1)
    
    print(f'Finished processing day: {day}, r_time shape: {dataset[day]["r_time"].shape}')


    break





#         # Only getting interested keys
#         include = ["r_time", "r_h", "r_param", "r_error"]
#         data = {key: data[key] for key in data if key in include}
        
#         process = Filtering.DataProcessing(data)
#         process.filter_range("r_h", 90, 400)
#         process.handle_nan(replace_val=666)
        
#         X = process.return_data()
        
        

#         for key in include:
#             dataset[day][key].append(np.array(X[key]))
    
#     iteration += 1
    
# with open('Sorted_data.pkl', 'wb') as file:
#     pickle.dump(dataset, file)








