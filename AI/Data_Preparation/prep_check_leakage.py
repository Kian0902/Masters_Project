# -*- coding: utf-8 -*-
"""
Created on Wed May  7 16:35:12 2025

@author: Kian Sartipzadeh
"""


from pathlib import Path
import shutil

class LeakageCleaner:
    """
    Cleans overlapping files between training and testing datasets.
    """
    def __init__(
        self,
        train_folder_a,
        train_folder_b,
        test_folder_a,
        test_folder_b,
        extensions=None,
    ):
        """
        Parameters:
        - train_folder_a: path to training inputs
        - train_folder_b: path to training ground truths
        - test_folder_a: path to testing inputs
        - test_folder_b: path to testing ground truths
        - extensions: list of file extensions to consider (including the dot)
        """
        self.trainA = Path(train_folder_a)
        self.trainB = Path(train_folder_b)
        self.testA = Path(test_folder_a)
        self.testB = Path(test_folder_b)
        self.extensions = extensions or ['.png', '.csv']

    def _map_files(self, folder):
        """
        Map each file stem to its Path for known extensions.
        """
        files = {}
        for p in folder.iterdir():
            if p.is_file() and p.suffix in self.extensions:
                files[p.stem] = p
        return files

    def _matching_pairs(self, folder1, folder2):
        """
        Return a dict mapping stem to (path_in_folder1, path_in_folder2)
        for all common stems.
        """
        map1 = self._map_files(folder1)
        map2 = self._map_files(folder2)
        common = set(map1) & set(map2)
        pairs = {}
        for stem in common:
            pairs[stem] = (map1[stem], map2[stem])
        return pairs

    def find_overlap(self):
        """
        Returns a set of stems appearing in both train and test pairs.
        """
        train_pairs = self._matching_pairs(self.trainA, self.trainB)
        test_pairs = self._matching_pairs(self.testA, self.testB)
        return set(train_pairs.keys()) & set(test_pairs.keys())

    def clean_training(self, dry_run=False):
        """
        Remove overlapping files from training folders.
        If dry_run=True, only returns the list of files to remove.
        Returns a list of Path objects.
        """
        overlaps = self.find_overlap()
        train_pairs = self._matching_pairs(self.trainA, self.trainB)
        removed = []
        for stem in overlaps:
            a_path, b_path = train_pairs[stem]
            removed.extend([a_path, b_path])
            if not dry_run:
                a_path.unlink()
                b_path.unlink()
        return removed


