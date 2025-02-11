# -*- coding: utf-8 -*-
"""
Created on Sun Sep 29 11:56:05 2024

@author: Kian Sartipzadeh
"""

import os
import pickle
import numpy as np

class IonogramSorting:
    def __init__(self):
        self.ionogram_dataset = {}  # Nested dictionary to store the data
    
    def import_data(self, datapath: str):
        """
        This function handles ionogram data in form of text files that has been
        pre-processed by the "SAO explorer" software.
        
        Each of these text files consist of 24-hour worth of ionosonde measurements
        with 15 minutes interval per data update. In other words, each 15 min
        interval ("batch") has a time and a date header followed by the ionosonde measurements.
        Each measurement (one row) has 8 ionosonde features represented as the
        columns as such: [Freq  Range  Pol  MPA  Amp  Doppler  Az  Zn].
        
        The number of measurements (rows) per "batch" changes depending on whether
        or not the Ionosonde was able to receive a backscatter signal. So each
        "batch" can contain different number of measurements.
        
        
        Input (type)    | DESCRIPTION
        ------------------------------------------------
        datapath (str)  | Path to folder that contains original ionograms txt files
        
        Return (type)              | DESCRIPTION
        ------------------------------------------------
        ionogram_data (np.ndarray) | Processed ionogram data
        ionogram_time (np.ndarray) | Timestamps of ionogram data
        """
        
        ionogram_time = []
        ionogram_data = []
        with open(datapath, "r") as file:
            
            lines = file.readlines()  # Reading all lines in txt file
    
            data_batch = []
            for line in lines:
                
                # When encountering new header containing date and time (Ex: "2018.09.21 (264) 00:00:00.000")
                if len(line) == 30:  # length of header containing date and time which is 30
                    iono_date = line[0:10]  # length of date (Ex: "2018.09.21" has length=10)
                    iono_time = f"{line[-13:-11]}-{line[-10:-8]}-{line[-7:-5]}"  # defining new time format (Ex: 20-15-00)
                    iono_datetime = f"{iono_date}_{iono_time}"  # changing the format to be "yyyy.MM.dd_hh-mm-ss"
                    ionogram_time.append(iono_datetime)
                
                # When encountering ionogram data (Ex: 3.400  315.0  90  24  33  -1.172 270.0  30.0)
                if len(line) == 46:  # length of each line containing ionogram values which is 46
                    line_split = line.split()  # splitting strings in line by the whitespace between values
                    line_final = [float(x) for x in line_split]  # Converting strings to floats
                    data_batch.append(line_final)
                
                # When encountering space between each batch of 15 min data
                if len(line) == 1:  # length of whitespace which is 1
                    if data_batch:  # Check if data_batch is not empty
                        ionogram_data.append(np.array(data_batch))  # appending the "batch" to the total data list 
                        data_batch = []  # resetting the batch list 
                
            # Handle the last batch if the file doesn't end with a whitespace
            if data_batch:
                ionogram_data.append(np.array(data_batch))
        
        # Process the collected data into the nested dictionary structure
        for i, time_str in enumerate(ionogram_time):
            # Split the time string into date and time parts
            date_part, time_part = time_str.split('_')
            
            # Parse date components (year, month, day)
            year, month, day = map(int, date_part.split('.'))
            date_key = f"{year}-{month}-{day}"
            
            # Parse time components (hour, minute, second)
            hh, MM, ss = map(int, time_part.split('-'))
            time_entry = [year, month, day, hh, MM, ss]
            data_array = ionogram_data[i]
            
            # Initialize the date entry if it doesn't exist
            if date_key not in self.ionogram_dataset:
                self.ionogram_dataset[date_key] = {'r_time': [], 'r_param': []}
            
            # Append the time and data to the respective lists
            self.ionogram_dataset[date_key]['r_time'].append(time_entry)
            self.ionogram_dataset[date_key]['r_param'].append(data_array)
        
        # Convert lists to numpy arrays for each date in the dataset
        for date_key in self.ionogram_dataset:
            self.ionogram_dataset[date_key]['r_time'] = np.array(self.ionogram_dataset[date_key]['r_time'])
            self.ionogram_dataset[date_key]['r_param'] = np.array(self.ionogram_dataset[date_key]['r_param'], dtype=object)
        
        
        # Converting list into np.ndarrays
        ionogram_time = np.array(ionogram_time, dtype=object)
        ionogram_data = np.array(ionogram_data, dtype=object)
        
        return ionogram_time, ionogram_data
    
    def return_dataset(self):
        return self.ionogram_dataset
    
    def save_as_dict(self, folder_path: str):
        """
        Saves each entry of the ionogram_dataset dictionary to a separate pickle file
        in the specified folder. Each file is named after its corresponding date key,
        e.g. '2023-2-11.pkl', so that the filename reflects the date of the month.
        """
        os.makedirs(folder_path, exist_ok=True)
        for date_key, data in self.ionogram_dataset.items():
            file_path = os.path.join(folder_path, f"{date_key}.pkl")
            with open(file_path, 'wb') as f:
                pickle.dump(data, f)
                


# import os
# import pickle
# import numpy as np



# class IonogramSorting:
#     def __init__(self):
#         self.ionogram_dataset = {}
    
    
    
    
#     def import_data(self, datapath: str):
#         """
#         This function handles ionogram data in form of text files that has been
#         pre-processed by the "SAO explorer" software.
        
#         Each of these text files consist of 24-hour worth of ionosonde measurements
#         with 15 minutes interval per data update. In other words, each 15 min
#         interval ("batch") has a time and a date header followed by the ionosonde measurements.
#         Each measurement (one row) has 8 ionosonde features represented as the
#         columns as such: [Freq  Range  Pol  MPA  Amp  Doppler  Az  Zn].
        
#         The number of measurements (rows) per "batch" changes depending on whether
#         or not the Ionosonde was able to receive a backscatter signal. So each
#         "batch" can contain different number of measurements.
        
        
#         Input (type)    | DESCRIPTION
#         ------------------------------------------------
#         datapath (str)  | Path to folder that contains original ionograms txt files
        
#         Return (type)              | DESCRIPTION
#         ------------------------------------------------
#         ionogram_data (np.ndarray) | Procrssed ionogram data
#         ionogram_time (np.ndarray) | Timestamps of ionogram data
#         """
        
#         ionogram_time = []
#         ionogram_data = []
#         with open(datapath, "r") as file:
            
#             lines = file.readlines() # Reading all lines in txt file
    
#             data_batch = []
#             for line in lines:
                
#                 """ # When encountering new header containing date and time (Ex: "2018.09.21 (264) 00:00:00.000") """
#                 if len(line) == 30:                                               # length of header containing date and time which is 30
#                     iono_date = line[0:10]                                        # length of date (Ex: "2018.09.21" has length=10)
#                     iono_time = f"{line[-13:-11]}-{line[-10:-8]}-{line[-7:-5]}"   # defining new time format (Ex: 20-15-00)
#                     iono_datetime = f"{iono_date}_{iono_time}"                    # changing the format to be "yyyy.MM.dd_hh-mm-ss"
#                     ionogram_time.append(iono_datetime)
                
                
#                 """ When encountering ionogram data (Ex: 3.400  315.0  90  24  33  -1.172 270.0  30.0) """
#                 if len(line) == 46:                             # length of each line containing ionogram values which is 46
#                     line_split = line.split()                   # splitting strings in line by the whitespace between values e.g., ["3.14 0.4"] to ["3.14", "0.4"]
#                     line_final= [float(x) for x in line_split]  # Converting strings to floats
#                     data_batch.append(line_final)
                
                
#                 """ When encountering space between each batch of 15 min data """
#                 if len(line) == 1:                              # length of whitespace which is 1
#                     ionogram_data.append(np.array(data_batch))  # appending the "batch" to the total data list 
#                     data_batch = []                             # resetting the batch list 
                
#                 else:
#                     continue
        
#             # Converting list into np.ndarrays
#             ionogram_time = np.array(ionogram_time, dtype=object)
#             ionogram_data = np.array(ionogram_data, dtype=object)
        
#         # Store ionogram_time and ionogram_data as key-value pairs in the dictionary
#         for i, time in enumerate(ionogram_time):
#             self.ionogram_dataset[time] = ionogram_data[i]
        
#         return ionogram_time, ionogram_data
    
    
    
#     def return_dataset(self):
#         return self.ionogram_dataset













# import pickle

# class IonogramSorting:
#     def __init__(self):
#         # This will eventually hold the nested dictionary:
#         # { "yyyy-m-d": { "r_time": np.array(...), "r_param": np.array(...) }, ... }
#         self.ionogram_dataset = {}

#     def import_data(self, datapath: str):
#         """
#         Processes one monthly ionosonde text file into a nested dictionary.
        
#         The file is assumed to contain multiple 15-minute batches. Each batch
#         starts with a header line of length 30 containing a date and time (e.g.,
#         "2018.09.21 (264) 00:00:00.000"), followed by one or more data lines
#         (each of length 46) with 8 ionogram features. A blank line (or a line 
#         that becomes empty after stripping) indicates the end of a batch.
        
#         After processing, the nested dictionary is built so that the global keys
#         are day strings ("yyyy-m-d"). For each day the two keys are:
#             "r_time": an array of timestamps (each row: [year, month, day, hour, minute, second])
#             "r_param": an object array where each element is a NumPy array of shape (N,8)
        
#         Parameters:
#             datapath (str): Path to the monthly ionosonde text file.
#         """
#         # This dictionary will be built incrementally.
#         sorted_dict = {}

#         with open(datapath, "r") as file:
#             lines = file.readlines()

#         # Temporary holders for a given 15-min batch:
#         data_batch = []          # List of data rows (each a list of 8 floats)
#         current_time = None      # Will hold the current timestamp as a list of ints
#         current_day_key = None   # Global key string "yyyy-m-d" for the batch
        
#         for line in lines:
#             # --- Check for a header line ---
#             # (Your original code assumes header lines have length 30.)
#             if len(line) == 30:
#                 # Example header: "2018.09.21 (264) 00:00:00.000"
#                 # Extract the date part and convert to integers.
#                 iono_date = line[0:10]  # e.g., "2018.09.21"
#                 try:
#                     year, month, day = [int(x) for x in iono_date.split('.')]
#                 except Exception as e:
#                     print("Error parsing date in header:", line)
#                     continue
                
#                 # Extract the time from the header.
#                 try:
#                     hour   = int(line[-13:-11])  # e.g., "00"
#                     minute = int(line[-10:-8])   # e.g., "00"
#                     second = int(line[-7:-5])    # e.g., "00"
#                 except Exception as e:
#                     print("Error parsing time in header:", line)
#                     continue
                
#                 # Save the current timestamp (as a list of ints).
#                 current_time = [year, month, day, hour, minute, second]
#                 # Create the global key for this day using the desired "yyyy-m-d" format.
#                 current_day_key = f"{year}-{month}-{day}"
                
#             # --- Check for a data line (ionogram measurements) ---
#             # (Your original code assumes data lines have length 46.)
#             elif len(line) == 46:
#                 parts = line.split()
#                 try:
#                     # Convert each string to a float.
#                     values = [float(x) for x in parts]
#                 except Exception as e:
#                     print("Error converting data line to floats:", line)
#                     continue
#                 data_batch.append(values)
                
#             # --- Check for a blank line (end of batch) ---
#             # (Your original code uses a line length of 1 for whitespace; here we use strip().)
#             elif len(line.strip()) == 0:
#                 if current_time is not None:
#                     # End of the current batch – convert collected data to a NumPy array.
#                     # If no data were collected (which might occur), we create an empty (0,8) array.
#                     batch_array = np.array(data_batch) if data_batch else np.empty((0, 8))
                    
#                     # If the current day is not yet a key, create its entry.
#                     if current_day_key not in sorted_dict:
#                         sorted_dict[current_day_key] = {"r_time": [], "r_param": []}
                    
#                     # Append the timestamp and the parameter array for this batch.
#                     sorted_dict[current_day_key]["r_time"].append(current_time)
#                     sorted_dict[current_day_key]["r_param"].append(batch_array)
                    
#                     # Reset temporary holders for the next batch.
#                     data_batch = []
#                     current_time = None
#                     current_day_key = None
#             else:
#                 # If the line doesn't match any of the above lengths, skip it.
#                 continue

#         # In case the file does not end with a blank line, check if a batch remains:
#         if data_batch and current_time is not None and current_day_key is not None:
#             batch_array = np.array(data_batch) if data_batch else np.empty((0, 8))
#             if current_day_key not in sorted_dict:
#                 sorted_dict[current_day_key] = {"r_time": [], "r_param": []}
#             sorted_dict[current_day_key]["r_time"].append(current_time)
#             sorted_dict[current_day_key]["r_param"].append(batch_array)
        
#         # Finally, for each day, convert the lists into NumPy arrays.
#         for day in sorted_dict:
#             # "r_time" becomes an integer array of shape (M, 6)
#             sorted_dict[day]["r_time"] = np.array(sorted_dict[day]["r_time"], dtype=int)
#             # "r_param" becomes an object array; each element is a NumPy array of shape (N, 8)
#             sorted_dict[day]["r_param"] = np.array(sorted_dict[day]["r_param"], dtype=object)
        
#         # Store the nested dictionary in the instance attribute.
#         self.ionogram_dataset = sorted_dict

#     def save_dataset(self, out_folder: str, filename: str):
#         """
#         Saves the processed nested dictionary to a file using pickle.
        
#         Parameters:
#             out_folder (str): The folder in which to save the file.
#             filename (str): The filename (for example, "processed_data.pkl").
#         """
#         if not os.path.exists(out_folder):
#             os.makedirs(out_folder)
        
#         save_path = os.path.join(out_folder, filename)
#         with open(save_path, "wb") as f:
#             pickle.dump(self.ionogram_dataset, f)
#         print(f"Dataset saved to {save_path}")

#     def return_dataset(self):
#         """Returns the processed nested dictionary."""
#         return self.ionogram_dataset




# # Example usage:
# if __name__ == '__main__':
#     # Suppose you loop through your monthly text files:
#     input_folder = "Ionogram_TXT"
#     output_folder = "sorted_ionogram_dicts"
    
#     # Process each file:
#     for txt_filename in tqdm(os.listdir(input_folder)):
#         # if not txt_filename.endswith(".txt"):
#             # continue
#         filepath = os.path.join(input_folder, txt_filename)
        
#         sorter = IonogramSorting()
#         sorter.import_data(filepath)
        
#         # You can now work with the nested dict:
#         nested_dict = sorter.return_dataset()
#         # For example, print the keys (days) and shapes of the arrays:
#         for day, data in nested_dict.items():
#             print(f"Day: {day}, r_time shape: {data['r_time'].shape}, "
#                   f"r_param length: {data['r_param'].shape}")
        
#         # Save the processed nested dict. You might choose a filename that reflects the source.
#         save_filename = os.path.splitext(txt_filename)[0] + "_processed.pkl"
#         sorter.save_dataset(output_folder, save_filename)































