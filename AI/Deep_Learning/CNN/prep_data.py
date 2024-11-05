# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 13:26:48 2024

@author: Kian Sartipzadeh
"""



import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt  # Import matplotlib for plotting

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




    def find_empty_and_good_ionograms(self, variance_threshold):
        empty_ionograms = []
        non_empty_ionograms = []
        matching_filenames = self.get_matching_filenames()

        for filename in matching_filenames:
            ionogram_path = os.path.join(self.ionogram_folder, f"{filename}.png")
            ionogram_image = Image.open(ionogram_path)
            ionogram_array = np.array(ionogram_image)

            # Compute the variance of the pixel values
            variance = np.var(ionogram_array)

            # If variance is below the threshold, consider it empty or almost empty
            if variance < variance_threshold:
                empty_ionograms.append(filename)
            else:
                non_empty_ionograms.append(filename)
        return empty_ionograms, non_empty_ionograms
    
    
