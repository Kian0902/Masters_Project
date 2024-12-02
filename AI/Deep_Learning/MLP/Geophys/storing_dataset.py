# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 14:55:24 2024

@author: Kian Sartipzadeh
"""



import os
import shutil
import numpy as np
import matplotlib.pyplot as plt


import torch
from torch.utils.data import Dataset

from tqdm import tqdm 




class StoreDataset(Dataset):
    def __init__(self, data, targets):
        self.data = torch.tensor(np.array(data), dtype=torch.float32)
        self.targets = torch.tensor(np.array(targets), dtype=torch.float32)
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]




class MatchingPairs:
    def __init__(self, target_folder, data_folder):
        self.target_folder = target_folder
        self.data_folder = data_folder
    
    def list_csv_files(self):
        return [f for f in os.listdir(self.target_folder) if f.endswith('.csv')]
    
    def list_csv_files_data(self):
        return [f for f in os.listdir(self.data_folder) if f.endswith('.csv')]
    
    def get_filename_without_extension(self, filename):
        return os.path.splitext(filename)[0]
    
    
    def get_matching_filenames(self):
        target_names = self.list_csv_files()
        data_names = self.list_csv_files_data()
        
        target_filenames = set(self.get_filename_without_extension(f) for f in target_names)
        data_filenames = set(self.get_filename_without_extension(f) for f in data_names)
        return sorted(list(data_filenames.intersection(target_filenames)))
        
    
    def find_pairs(self):
        TAR = []
        GEO= []
        
        matching_filenames = self.get_matching_filenames()
        for filename in tqdm(matching_filenames, desc="Finding sample pairs"):
            
            target_path = os.path.join(self.target_folder, f"{filename}.csv")
            target_data = np.genfromtxt(target_path, dtype=np.float64, delimiter=",")
            
            data_path = os.path.join(self.data_folder, f"{filename}.csv")
            data_data = np.genfromtxt(data_path, dtype=np.float64, delimiter=",")
            
            TAR.append(target_data)
            GEO.append(data_data)
            
        print("\nDone!")
        return TAR, GEO


    def save_matching_files(self, new_target_folder, new_data_folder):
        # Create new folders if they don't exist
        os.makedirs(new_target_folder, exist_ok=True)
        os.makedirs(new_data_folder, exist_ok=True)

        matching_filenames = self.get_matching_filenames()

        for filename in tqdm(matching_filenames, desc="Saving pairs to folders"):

            # Copy target files
            target_src = os.path.join(self.target_folder, f"{filename}.csv")
            target_dst = os.path.join(new_target_folder, f"{filename}.csv")
            shutil.copy2(target_src, target_dst)

            # Copy data files
            data_src = os.path.join(self.data_folder, f"{filename}.csv")
            data_dst = os.path.join(new_data_folder, f"{filename}.csv")
            shutil.copy2(data_src, data_dst)
            
        print("\nDone!")




# if __name__ == "__main__":
#     eiscat_folder = "EISCAT_samples"
#     geophys_folder = "geophys_samples"


#     Pairs = MatchingPairs(eiscat_folder, geophys_folder)
    
#     Pairs.save_matching_files("train_eiscat_folder", "train_geophys_folder")
#     # eis, ge = Pairs.find_pairs()














