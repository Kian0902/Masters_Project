
import os
import pandas as pd
import numpy as np  # Added from testing_mlp

# SP19_samples
# EISCAT_samples

def list_csv_files(folder_path):
    return [f for f in os.listdir(folder_path) if f.endswith('.csv')]

def match_files(A_files, B_files):
    radar_dates = {f.split('.')[0]: f for f in A_files}
    satellite_dates = {f.split('.')[0]: f for f in B_files}
    matched_pairs = []
    for date in satellite_dates.keys():
        if date in radar_dates:
            matched_pairs.append(date)
    print(len(matched_pairs))

eiscat_files = list_csv_files("EISCAT_samples")
sp19_files = list_csv_files("SP19_samples")

match_files(eiscat_files, sp19_files)
