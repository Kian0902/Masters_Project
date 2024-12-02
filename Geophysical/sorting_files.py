# -*- coding: utf-8 -*-
"""
Created on Sat Sep 14 14:04:36 2024

@author: Kian Sartipzadeh
"""


# import os
# import glob
# import numpy as np
# import pandas as pd



# custom_header=['DoY/366', 'ToD/1440', 'Solar_Zenith/44', 'Kp', 'R', 'Dst',
#                'ap', 'F10_7', 'AE', 'AL', 'AU', 'PC_potential', 'Lyman_alpha',
#                'Bx', 'By', 'Bz', 'dBx', 'dBy', 'dBz']




# def extract_geophysical_data(folder):
    
#     df_s1 = []
#     for i, filename in enumerate(os.listdir(folder_path)):
#         if filename.endswith('.csv'):
#             if i % 1000 == 0:
#                 print(f'{i}/{len(os.listdir(folder_path))}')
            
#             file_path = os.path.join(folder_path, filename)
            
#             # Extract dat from first data souce
#             df_source1 = pd.read_csv(file_path, header=None, skiprows=1, nrows=19)
#             df_s1.append(df_source1.T) # Transpose
            
#     return df_s1




# if __name__ == "__main__":
    
#     folder_path = "ionograms_1D"
    
#     df_list = extract_geophysical_data(folder_path)
    
#     combined_df = pd.concat(df_list, axis=0)
#     numpy_array = combined_df.to_numpy()
    

#     np.save('Geophysical.npy', numpy_array)
#     print("Data saved as Geophysical.npy")



import os
import pandas as pd
from tqdm import tqdm

def extract_and_save_data(folder_path, output_folder1, output_folder2):
    """
    Extracts data from CSV files, splits them by source, and saves them to respective folders.
    """
    if not os.path.exists(output_folder1):
        os.makedirs(output_folder1)
    if not os.path.exists(output_folder2):
        os.makedirs(output_folder2)

    for filename in tqdm(os.listdir(folder_path)):
        if filename.endswith('.csv'):

            # Full path of the current file
            file_path = os.path.join(folder_path, filename)

            # Read the entire CSV file
            df = pd.read_csv(file_path, header=None, skiprows=1)

            # Split the data into two sources
            source1_data = df.iloc[:19, :].T  # Transpose first 19 rows
            source2_data = df.iloc[19:, :].T  # Transpose the rest

            # Save source1 data
            output_path1 = os.path.join(output_folder1, filename)
            source1_data.to_csv(output_path1, index=False, header=False)

            # Save source2 data
            output_path2 = os.path.join(output_folder2, filename)
            source2_data.to_csv(output_path2, index=False, header=False)
    print(f"Data has been divided and saved into '{output_folder1}' and '{output_folder2}'.")


if __name__ == "__main__":
    # Input folder containing the original CSV files
    folder_path = "ionograms_1D"
    
    # Output folders for source1 and source2 data
    output_folder1 = "geophys_data"
    output_folder2 = "ionogram1D_data"

    # Process the data
    extract_and_save_data(folder_path, output_folder1, output_folder2)





