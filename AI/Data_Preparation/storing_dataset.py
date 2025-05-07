# -*- coding: utf-8 -*-
"""
Created on Sun Sep 29 15:17:01 2024

@author: Kian Sartipzadeh
"""

import os
import shutil
import numpy as np

from PIL import Image
from tqdm import tqdm




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
        
    
    def find_pairs(self, return_date:bool = False):
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
            
        if return_date is True:
            return RAD, ION, matching_filenames
        
        else:
            return RAD, ION
    
    
    def save_matching_files(self, new_ionogram_folder, new_radar_folder):
        # Create new folders if they don't exist
        os.makedirs(new_ionogram_folder, exist_ok=True)
        os.makedirs(new_radar_folder, exist_ok=True)


        matching_filenames = self.get_matching_filenames()

        for filename in tqdm(matching_filenames):
            # Copy ionogram files
            ionogram_src = os.path.join(self.ionogram_folder, f"{filename}.png")
            ionogram_dst = os.path.join(new_ionogram_folder, f"{filename}.png")
            shutil.copy2(ionogram_src, ionogram_dst)

            # Copy radar files
            radar_src = os.path.join(self.radar_folder, f"{filename}.csv")
            radar_dst = os.path.join(new_radar_folder, f"{filename}.csv")
            shutil.copy2(radar_src, radar_dst)



class Matching3Pairs:
    def __init__(self,  eiscat_folder, geophys_folder, ionogram_folder):
        self.eiscat_folder = eiscat_folder
        self.geophys_folder = geophys_folder
        self.ionogram_folder = ionogram_folder
        
        
    def list_csv_files(self, dataset):
        return [f for f in os.listdir(dataset) if f.endswith('.csv')]
    

    def list_png_files(self, images):
        return [f for f in os.listdir(images) if f.endswith('.png')]


    def get_filename_without_extension(self, filename):
        return os.path.splitext(filename)[0]
    
    
    def get_matching_filenames(self):
        eiscat_names = self.list_csv_files(self.eiscat_folder)
        geophys_names = self.list_csv_files(self.geophys_folder)
        ionogram_names = self.list_png_files(self.ionogram_folder)
        
        eiscat_filenames = set(self.get_filename_without_extension(f) for f in eiscat_names)
        geophys_filenames = set(self.get_filename_without_extension(f) for f in geophys_names)
        ionogram_filenames = set(self.get_filename_without_extension(f) for f in ionogram_names)
        
        
        common = sorted(list(eiscat_filenames.intersection(geophys_filenames, ionogram_filenames)))
        return common
        
    
    def find_pairs(self, return_date:bool = False):
        i=0
        
        RAD = []
        GEO= []
        ION = []
        
        matching_filenames = self.get_matching_filenames()
        for filename in matching_filenames:
            
            eiscat_path = os.path.join(self.eiscat_folder, f"{filename}.csv")
            eiscat_data = np.genfromtxt(eiscat_path, dtype=np.float64, delimiter=",")
            
            geophys_path = os.path.join(self.geophys_folder, f"{filename}.csv")
            geophys_data = np.genfromtxt(geophys_path, dtype=np.float64, delimiter=",")
            
            ionogram_path = os.path.join(self.ionogram_folder, f"{filename}.png")
            ionogram_image = Image.open(ionogram_path)
            ionogram_array = np.array(ionogram_image)
            
            
            
            RAD.append(eiscat_data)
            GEO.append(geophys_data)
            ION.append(ionogram_array)
            
            i+=1
        
        if return_date is True:
            return RAD, GEO, ION, matching_filenames
        
        else:
            return RAD, GEO, ION
        


    def save_matching_files(self):
        new_eiscat_folder = self.eiscat_folder+"_new"
        new_geophys_folder = self.geophys_folder+"_new"
        new_ionogram_folder = self.ionogram_folder+"_new"
        
        
        # Create new folders if they don't exist
        os.makedirs(new_eiscat_folder, exist_ok=True)
        os.makedirs(new_geophys_folder, exist_ok=True)
        os.makedirs(new_ionogram_folder, exist_ok=True)
        
        matching_filenames = self.get_matching_filenames()

        for filename in tqdm(matching_filenames):
            # Copy eiscat files
            eiscat_src = os.path.join(self.eiscat_folder, f"{filename}.csv")
            eiscat_dst = os.path.join(new_eiscat_folder, f"{filename}.csv")
            shutil.copy2(eiscat_src, eiscat_dst)

            # Copy geophys files
            geophys_src = os.path.join(self.geophys_folder, f"{filename}.csv")
            geophys_dst = os.path.join(new_geophys_folder, f"{filename}.csv")
            shutil.copy2(geophys_src, geophys_dst)

            # Copy ionogram files
            ionogram_src = os.path.join(self.ionogram_folder, f"{filename}.png")
            ionogram_dst = os.path.join(new_ionogram_folder, f"{filename}.png")
            shutil.copy2(ionogram_src, ionogram_dst)


# Example usage
if __name__ == "__main__":
    geophys_folder = "geophys_10days"
    eiscat_folder = "eiscat_10days"
    ionogram_folder = "ionogram_10days"
    
    matcher = Matching3Pairs(eiscat_folder, geophys_folder, ionogram_folder)
    matcher.save_matching_files()
    print("...")


