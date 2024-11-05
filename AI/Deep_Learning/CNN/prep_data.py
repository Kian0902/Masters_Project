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
    
    
    
    def plot_variance_distribution(self):
        variances = []
        matching_filenames = self.get_matching_filenames()
    
        for filename in matching_filenames:
            ionogram_path = os.path.join(self.ionogram_folder, f"{filename}.png")
            ionogram_image = Image.open(ionogram_path)
            ionogram_array = np.array(ionogram_image)
            variance = np.var(ionogram_array)
            variances.append(variance)
    
        # Plot the histogram of variances
        plt.hist(variances, bins=50)
        plt.xlim(-100, 1000)
        plt.xlabel('Variance')
        plt.ylabel('Number of Images')
        plt.title('Variance Distribution of Ionogram Images')
        plt.show()

    
    def plot_empty_ionograms(self, empty_ionograms):
        """
        Plots each empty or almost empty ionogram in its own plot and loops over all of them.

        Parameters:
        - empty_ionograms: List of filenames of empty ionograms.
        """
        for filename in empty_ionograms:
            ionogram_path = os.path.join(self.ionogram_folder, f"{filename}.png")
            ionogram_image = Image.open(ionogram_path)
            ionogram_array = np.array(ionogram_image)

            plt.figure()
            plt.imshow(ionogram_array)
            plt.title(f"Empty Ionogram: {filename}")
            plt.axis('off')
            plt.show()
    
    
    def plot_good_ionograms(self, non_empty_ionograms):
        """
        Plots each non empty ionogram in its own plot and loops over all of them.

        Parameters:
        - empty_ionograms: List of filenames of empty ionograms.
        """
        for filename in non_empty_ionograms:
            ionogram_path = os.path.join(self.ionogram_folder, f"{filename}.png")
            ionogram_image = Image.open(ionogram_path)
            ionogram_array = np.array(ionogram_image)

            plt.figure()
            plt.imshow(ionogram_array)
            plt.title(f"Good Ionogram: {filename}")
            plt.axis('off')
            plt.show()


    # Optional: Save non-empty ionograms to files
    def save_good_ionograms(self, good_ionograms, save_folder='Good_Ionograms_Images'):
        """
        Saves non-empty ionograms to a specified folder.

        Parameters:
        - non_empty_ionograms: List of filenames of non-empty ionograms.
        - save_folder: Folder to save the images.
        """
        os.makedirs(save_folder, exist_ok=True)
        for filename in good_ionograms:
            ionogram_path = os.path.join(self.ionogram_folder, f"{filename}.png")
            save_path = os.path.join(save_folder, f"{filename}.png")
            ionogram_image = Image.open(ionogram_path)
            ionogram_image.save(save_path)




# Set your folders
radar_folder = "EISCAT_samples"
ionogram_folder = "Ionogram_images"

# Initialize the class
Pairs = MatchingPairs(ionogram_folder, radar_folder)

# Define a variance threshold (adjust this based on your data)
variance_threshold = 11  # Example threshold

# Find empty or almost empty ionograms
empty_ionograms, good_ionograms = Pairs.find_empty_and_good_ionograms(variance_threshold)


# Plot the empty ionograms
# Pairs.plot_empty_ionograms(empty_ionograms)
# Pairs.plot_good_ionograms(good_ionograms)


Pairs.plot_variance_distribution()

Pairs.save_good_ionograms(good_ionograms)


print(f"Empty ionograms: {len(empty_ionograms)}")
print(f"Good ionograms: {len(good_ionograms)}")

