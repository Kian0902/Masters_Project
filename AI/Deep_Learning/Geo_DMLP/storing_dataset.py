# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 16:03:29 2025

@author: kian0
"""



import os
import shutil
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image

from torch.utils.data import Dataset

import torch
from torchvision import transforms




class StoreGeoDataset(Dataset):
    def __init__(self, data):
        self.data = torch.tensor(np.array(data), dtype=torch.float32)

        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]







class MatchingGeoPair:
    def __init__(self, geophys_folder):
        self.geophys_folder = geophys_folder
    
    
    def list_geo_csv_files(self):
        return [f for f in os.listdir(self.geophys_folder) if f.endswith('.csv')]


    def get_filename_without_extension(self, filename):
        return os.path.splitext(filename)[0]
    
    
    def get_matching_filenames(self):

        geophys_names = self.list_geo_csv_files()
        
        geophys_filenames = set(self.get_filename_without_extension(f) for f in geophys_names)
        
        return sorted(list(geophys_filenames))
        
        
    def find_pairs(self, return_date:bool = False):
        i=0
        
        GEO = []
        
        matching_filenames = self.get_matching_filenames()
        for filename in matching_filenames:
            geophys_path = os.path.join(self.geophys_folder, f"{filename}.csv")
            geophys_data = np.genfromtxt(geophys_path, dtype=np.float64, delimiter=",")
            
            GEO.append(geophys_data)
            
            i+=1
            
        if return_date is True:
            return GEO, matching_filenames
        
        else:
            return GEO
















class StoreDataset(Dataset):
    def __init__(self, data, targets):
        self.data = torch.tensor(np.array(data), dtype=torch.float32)
        self.targets = torch.tensor(np.array(targets), dtype=torch.float32)
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]
        







class MatchingPairs:
    def __init__(self, geophys_folder, radar_folder):
        self.geophys_folder = geophys_folder
        self.radar_folder = radar_folder
    
    
    def list_csv_files(self):
        return [f for f in os.listdir(self.radar_folder) if f.endswith('.csv')]
    
    def list_geo_csv_files(self):
        return [f for f in os.listdir(self.geophys_folder) if f.endswith('.csv')]


    def get_filename_without_extension(self, filename):
        return os.path.splitext(filename)[0]
    
    
    def get_matching_filenames(self):
        radar_names = self.list_csv_files()
        geophys_names = self.list_geo_csv_files()
        
        radar_filenames = set(self.get_filename_without_extension(f) for f in radar_names)
        geophys_filenames = set(self.get_filename_without_extension(f) for f in geophys_names)
        
        return sorted(list(geophys_filenames.intersection(radar_filenames)))
        
        
    def find_pairs(self, return_date:bool = False):
        i=0
        
        RAD = []
        GEO = []
        
        matching_filenames = self.get_matching_filenames()
        for filename in matching_filenames:
            geophys_path = os.path.join(self.geophys_folder, f"{filename}.csv")
            geophys_data = np.genfromtxt(geophys_path, dtype=np.float64, delimiter=",")
            
            radar_path = os.path.join(self.radar_folder, f"{filename}.csv")
            radar_data = np.genfromtxt(radar_path, dtype=np.float64, delimiter=",")
            
            
            RAD.append(radar_data)
            GEO.append(geophys_data)
            
            i+=1
            
        if return_date is True:
            return RAD, GEO, matching_filenames
        
        else:
            return RAD, GEO
        
    def save_matching_files(self, new_geophys_folder, new_radar_folder):
        # Create new folders if they don't exist
        os.makedirs(new_geophys_folder, exist_ok=True)
        os.makedirs(new_radar_folder, exist_ok=True)

        matching_filenames = self.get_matching_filenames()

        for filename in matching_filenames:
            # Copy geophys files
            geophys_src = os.path.join(self.geophys_folder, f"{filename}.csv")
            geophys_dst = os.path.join(new_geophys_folder, f"{filename}.csv")
            shutil.copy2(geophys_src, geophys_dst)

            # Copy radar files
            radar_src = os.path.join(self.radar_folder, f"{filename}.csv")
            radar_dst = os.path.join(new_radar_folder, f"{filename}.csv")
            shutil.copy2(radar_src, radar_dst)
            




if __name__ == "__main__":
    geophys_folder = "testing_data/geo_test_flow"
    radar_folder = "testing_data/eis_test_flow"

    new_geophys_folder = "testing_data/geo_test_flow_new"
    new_radar_folder = "testing_data/eis_test_flow_new"

    matcher = MatchingPairs(geophys_folder, radar_folder)
    matcher.save_matching_files(new_geophys_folder, new_radar_folder)
    # print("...")





