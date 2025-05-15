# -*- coding: utf-8 -*-
"""
Created on Wed May  7 14:47:55 2025

@author: Kian Sartipzadeh
"""


from prep_check_leakage import LeakageCleaner
from prep_matching_datasets import Matching2Pairs, Matching3Pairs, MatchingXPairs


if __name__ == "__main__":
    eiscat_folder = "eiscat_test_days"
    geophys_folder = "geophys_test_days"
    ionogram_folder = "ionogram_test_days"
    
    
    # matcher = MatchingPairs(geophys_folder, ionogram_folder)
    # matcher.get_matching_filenames()
    matcher = MatchingXPairs(ionogram_folder, geophys_folder)
    matcher.save_matching_files()
    print("...")








    # Example usage:
    # cleaner = DatasetOverlapCleaner('train/input', 'train/gt', 'test/input', 'test/gt')
    # print(cleaner.find_overlap())
    # print(cleaner.clean_training(dry_run=True))
    
    



