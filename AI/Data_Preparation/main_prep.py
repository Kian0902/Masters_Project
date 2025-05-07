# -*- coding: utf-8 -*-
"""
Created on Wed May  7 14:47:55 2025

@author: Kian Sartipzadeh
"""

from prep_matching_datasets import Matching2Pairs, Matching3Pairs, MatchingXPairs


if __name__ == "__main__":
    eiscat_folder = "eiscat_10days"
    geophys_folder = "geophys_10days"
    ionogram_folder = "ionogram_10days"
    
    
    # matcher = MatchingPairs(geophys_folder, ionogram_folder)
    # matcher.get_matching_filenames()
    matcher = MatchingXPairs(eiscat_folder, geophys_folder, ionogram_folder)
    matcher.save_matching_files()
    print("...")














