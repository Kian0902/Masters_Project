# -*- coding: utf-8 -*-
"""
Created on Sun Sep 29 15:17:01 2024

@author: Kian Sartipzadeh
"""



import os
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image

from torch.utils.data import Dataset, DataLoader

import torch
from torchvision import transforms





class IonoEisDataset(Dataset):
    def __init__(self, ionogram_folder, radar_folder, matching_filenames, transform=None):
        """
        Custom PyTorch Dataset for pairing ionograms and radar measurements.
        
        Args:
            ionogram_folder (str): Path to the folder containing ionogram images.
            radar_folder (str): Path to the folder containing radar CSV files.
            matching_filenames (list): List of matching filenames without extensions.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.ionogram_folder = ionogram_folder
        self.radar_folder = radar_folder
        self.matching_filenames = matching_filenames
        self.transform = transform

    def __len__(self):
        return len(self.matching_filenames)

    def __getitem__(self, idx):
        filename = self.matching_filenames[idx]
        
        
        ionogram_path = os.path.join(self.ionogram_folder, f"{filename}.png")
        ionogram_image = Image.open(ionogram_path)
        
        
        
        if self.transform:
            ionogram_image = self.transform(ionogram_image)
        
        radar_path = os.path.join(self.radar_folder, f"{filename}.csv")
        radar_data = np.genfromtxt(radar_path, dtype=np.float64, delimiter=",")
        
        
        radar_values = torch.tensor(radar_data, dtype=torch.float32)

        return ionogram_image, radar_values
    
    
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



# def list_csv_files(folder_path):
#     return [f for f in os.listdir(folder_path) if f.endswith('.csv')]


# def list_png_files(folder_path):
#     return [f for f in os.listdir(folder_path) if f.endswith('.png')]


# def get_filename_without_extension(filename):
#     return os.path.splitext(filename)[0]


# radar_folder = "EISCAT_samples"
# ionogram_folder = "Ionogram_sampled_images"







# radar_names = list_csv_files(radar_folder)
# ionogram_names = list_png_files(ionogram_folder)


# # Extract base filenames without ".csv"
# radar_filenames = set(get_filename_without_extension(f) for f in radar_names)
# ionogram_filenames = set(get_filename_without_extension(f) for f in ionogram_names)


# # Find the matching filenames
# matching_filenames = sorted(list(ionogram_filenames.intersection(radar_filenames)))



# transform = transforms.Compose([transforms.ToTensor()])


# A = IonoEisDataset(ionogram_folder, radar_folder, matching_filenames, transform=transform)


# # for i in range(1500):
# #     A.plot_sample_pair(i)




# train_loader = DataLoader(A, batch_size=32, shuffle=True)


# for batch_idx, (images, radar_values) in enumerate(train_loader):
#     print(f"Batch {batch_idx + 1}")
#     print(f"Image batch shape: {images.shape}")
#     print(f"Radar batch shape: {radar_values.shape}")
#     break














