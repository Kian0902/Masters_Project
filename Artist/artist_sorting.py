# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 13:34:18 2024

@author: Kian Sartipzadeh
"""


import pandas as pd
import numpy as np
from tqdm import tqdm


class ArtistSorter:
    def __init__(self, file_path):
        self.file_path = file_path
        self.raw_data = None
        self.processed_data = {}

    def load_data(self):
        """
        Loads the radar data from the CSV file.
        """
        self.raw_data = pd.read_csv(self.file_path)
        self.raw_data['Timestamp'] = pd.to_datetime(self.raw_data['Timestamp'], format='%d-%b-%Y %H:%M:%S')

    def process_data(self):
        """
        Processes the raw data into the nested dictionary format.
        """
        if self.raw_data is None:
            raise ValueError("Data not loaded. Please run load_data() first.")

        # Get the unique dates in the dataset
        unique_dates = self.raw_data['Timestamp'].dt.date.unique()

        # Define the altitudes (columns excluding Timestamp)
        altitude_columns = self.raw_data.columns[1:]


        # Loop over each unique date to create the nested structure
        for date in tqdm(unique_dates):
            formatted_date = f"{date.year}-{date.month}-{date.day}"
            
            # Filter the dataframe for the current date
            daily_data = self.raw_data[self.raw_data['Timestamp'].dt.date == date]



            # Extract the 'r_time' array
            r_time = np.array([[timestamp.year, timestamp.month, timestamp.day, timestamp.hour, timestamp.minute, timestamp.second]
                               for timestamp in daily_data['Timestamp']])

            # Extract the 'r_h' array (altitudes)
            r_h = np.array([list(map(lambda x: int(x.split('_')[1]), altitude_columns))])

            # Extract the 'r_param' array (electron densities)
            r_param = ((daily_data[altitude_columns].to_numpy()*1e6)**2)/9**2

            # Assign the arrays to the nested dictionary
            self.processed_data[formatted_date] = {
                "r_time": r_time,
                "r_h": r_h.T,
                "r_param": r_param.T
            }

    def get_processed_data_for_date(self, date):
        """
        Returns the processed data for a specific date.
        """
        if not self.processed_data:
            raise ValueError("Data not processed. Please run process_data() first.")
        return self.processed_data.get(date, "Date not found.")






if __name__ == "__main__":
    print("...")




