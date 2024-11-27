# -*- coding: utf-8 -*-
"""
Created on Sun Sep 29 15:17:01 2024

@author: Kian Sartipzadeh
"""


import os
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from torch.utils.data import Dataset
import torch
from torchvision import transforms




class StoreDataset(Dataset):
    def __init__(self, data, targets, transform=transforms.Compose([transforms.ToTensor()])):
        self.data = np.array(data)
        self.targets = torch.tensor(np.array(targets), dtype=torch.float32)
        self.transform = transform
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        
        return self.transform(self.data[idx]), self.targets[idx]
        
    
    def plot_sample_pair(self, idx):
        """
        Plots a single pair of ionogram image and radar data.
        
        Args:
            idx (int): Index of the sample pair to plot.
        """
        
        r_h = np.array([[ 91.5687711 ],[ 94.57444598],[ 97.57964223],[100.57010953],
               [103.57141624],[106.57728701],[110.08393175],[114.60422289],
               [120.1185208 ],[126.61221111],[134.1346149 ],[142.53945817],
               [152.05174717],[162.57986185],[174.09833378],[186.65837945],
               [200.15192581],[214.62769852],[230.12198695],[246.64398082],
               [264.11728204],[282.62750673],[302.15668686],[322.70723831],
               [344.19596481],[366.64409299],[390.113117  ]])
        
        # Retrieve the specified sample
        ionogram_image, radar_values = self.__getitem__(idx)

        # Convert image tensor back to a PIL Image for plotting
        if isinstance(ionogram_image, torch.Tensor):
            ionogram_image = transforms.ToPILImage()(ionogram_image)

        # Create a figure with 2 subplots (1x2 layout)
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        
        ax[0].imshow(ionogram_image)
        ax[0].axis("off")
        
        ax[1].plot(radar_values.squeeze().numpy(), r_h, color='skyblue')
        ax[1].set_xlabel("Measurement Index")
        ax[1].set_ylabel("Value")
        plt.show()






class MatchingPairs:
    def __init__(self, ionogram_folder, radar_folder):
        self.ionogram_folder = ionogram_folder
        self.radar_folder = radar_folder
    
    
    def list_csv_files(self):
        return [f for f in os.listdir(self.radar_folder) if f.endswith('.csv')]


    def list_png_files(self):
        return [f for f in os.listdir(self.ionogram_folder) if f.endswith('.png')]


    def get_filename_without_extension(self, filename):
        return os.path.splitext(filename)[0]
    
    
    def get_matching_filenames(self):
        radar_names = self.list_csv_files()
        ionogram_names = self.list_png_files()
        
        radar_filenames = set(self.get_filename_without_extension(f) for f in radar_names)
        ionogram_filenames = set(self.get_filename_without_extension(f) for f in ionogram_names)
        
        return sorted(list(ionogram_filenames.intersection(radar_filenames)))
        
    
    def find_pairs(self):
        i=0
        
        RAD = []
        ION = []
        
        matching_filenames = self.get_matching_filenames()
        for filename in matching_filenames:
            ionogram_path = os.path.join(self.ionogram_folder, f"{filename}.png")
            ionogram_image = Image.open(ionogram_path)
            ionogram_array = np.array(ionogram_image)
            
            radar_path = os.path.join(self.radar_folder, f"{filename}.csv")
            radar_data = np.genfromtxt(radar_path, dtype=np.float64, delimiter=",")
            
            
            RAD.append(radar_data)
            ION.append(ionogram_array)
            
            i+=1
        
        return RAD, ION





# Example usage
if __name__ == "__main__":
    # ionogram_folder = "Good_Ionograms_Images"
    # radar_folder = "EISCAT_test_samples"
    # sp19_folder = "SP19_samples"

    # new_ionogram_folder = "test_ionogram_folder"
    # new_radar_folder = "test_radar_folder"
    # new_sp19_folder = "test_sp19_folder"

    # matcher = Matching3Pairs(ionogram_folder, radar_folder, sp19_folder)
    # matcher.save_matching_files(new_ionogram_folder, new_radar_folder, new_sp19_folder)
    print("...")


