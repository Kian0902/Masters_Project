# -*- coding: utf-8 -*-
"""
Created on Sun Sep 29 15:17:01 2024

@author: Kian Sartipzadeh
"""


import shutil
import numpy as np

from PIL import Image
from tqdm import tqdm
from pathlib import Path


class Matching2Pairs:
    def __init__(self, folder_a, folder_b):
        self.A = Path(folder_a)
        self.B = Path(folder_b)

        # Map each extension to a loader function
        self._loaders = {
            '.png': lambda p: np.array(Image.open(p)),
            '.csv': lambda p: np.genfromtxt(p, dtype=np.float64, delimiter=','),
            # add more ext→loader as needed
        }

    def _map_files(self, folder: Path) -> dict[str, Path]:
        """
        Return a dict mapping each file’s stem (basename) → its full Path.
        Only includes files whose extension we know how to load.
        """
        files = {}
        for p in folder.iterdir():
            if not p.is_file():
                continue
            if p.suffix in self._loaders:
                files[p.stem] = p
        return files

    def _matching_pairs(self) -> dict[str, tuple[Path, Path]]:
        """
        Returns a dict mapping stem → (path_in_A, path_in_B)
        for every stem that exists in both A and B.
        """
        map_a = self._map_files(self.A)
        map_b = self._map_files(self.B)
        common = set(map_a) & set(map_b)
        return {stem: (map_a[stem], map_b[stem]) for stem in common}

    def find_pairs(self, return_names: bool = False):
        A_data, B_data, names = [], [], []
        for stem, (pa, pb) in self._matching_pairs().items():
            loader_a = self._loaders[pa.suffix]
            loader_b = self._loaders[pb.suffix]
            A_data.append(loader_a(pa))
            B_data.append(loader_b(pb))
            names.append(stem)

        if return_names:
            return A_data, B_data, names
        return A_data, B_data

    def save_matching_files(self, dest_a: str = None, dest_b: str = None):
        dest_a = Path(dest_a or f"{self.A}_new")
        dest_b = Path(dest_b or f"{self.B}_new")
        dest_a.mkdir(exist_ok=True)
        dest_b.mkdir(exist_ok=True)

        for stem, (pa, pb) in tqdm(self._matching_pairs().items()):
            shutil.copy2(pa, dest_a / pa.name)
            shutil.copy2(pb, dest_b / pb.name)




class Matching3Pairs:
    def __init__(self, folder_a, folder_b, folder_c):
        self.folder_a = Path(folder_a)
        self.folder_b = Path(folder_b)
        self.folder_c = Path(folder_c)

        # Map file extensions to loader functions
        self._loaders = {
            '.csv': lambda p: np.genfromtxt(p, dtype=np.float64, delimiter=','),
            '.png': lambda p: np.array(Image.open(p)),
            # add more extension→loader entries here as needed
        }

    def _map_files(self, folder: Path) -> dict[str, Path]:
        """
        Return a mapping stem -> Path for all files in `folder` whose suffix
        is in our loader map.
        """
        files = {}
        for p in folder.iterdir():
            if p.is_file() and (p.suffix in self._loaders):
                files[p.stem] = p
        return files

    def _matching_triples(self) -> dict[str, tuple[Path, Path, Path]]:
        """
        Find all stems common to folder_a, folder_b, and folder_c, returning
        stem -> (path_a, path_b, path_c).
        """
        ma = self._map_files(self.folder_a)
        mb = self._map_files(self.folder_b)
        mc = self._map_files(self.folder_c)
        common = set(ma) & set(mb) & set(mc)
        return {stem: (ma[stem], mb[stem], mc[stem]) for stem in sorted(common)}

    def find_pairs(self, return_names: bool = False):
        """
        Load and return three lists of data arrays;
        optionally also return the list of matching stems.
        """
        data_a, data_b, data_c, names = [], [], [], []
        for stem, (pa, pb, pc) in self._matching_triples().items():
            loader_a = self._loaders[pa.suffix]
            loader_b = self._loaders[pb.suffix]
            loader_c = self._loaders[pc.suffix]
            
            data_a.append(loader_a(pa))
            data_b.append(loader_b(pb))
            data_c.append(loader_c(pc))
            names.append(stem)
            
        if return_names:
            return data_a, data_b, data_c, names
        return data_a, data_b, data_c

    def save_matching_files(self,
                            dest_a: str = None,
                            dest_b: str = None,
                            dest_c: str = None):
        """
        Copy each matching file into a parallel `_new` folder (or custom
        destinations if provided).
        """
        da = Path(dest_a or f"{self.folder_a}_new"); da.mkdir(exist_ok=True)
        db = Path(dest_b or f"{self.folder_b}_new"); db.mkdir(exist_ok=True)
        dc = Path(dest_c or f"{self.folder_c}_new"); dc.mkdir(exist_ok=True)

        for stem, (pa, pb, pc) in tqdm(self._matching_triples().items()):
            shutil.copy2(pa, da/pa.name)
            shutil.copy2(pb, db/pb.name)
            shutil.copy2(pc, dc/pc.name)




class MatchingXPairs:
    """
    Generalized matching for X folders.  
    Given any number of folders containing files with supported extensions,
    this class can find common filenames (by stem), load their data,
    and copy matching files to new destinations.
    """
    def __init__(self, *folders: str):
        # Convert to Path objects and store
        self.folders = [Path(f) for f in folders]

        # Map each file extension to a loader function
        self._loaders = {
            '.csv': lambda p: np.genfromtxt(p, dtype=np.float64, delimiter=','),
            '.png': lambda p: np.array(Image.open(p)),
            # add more extension→loader entries here as needed
        }

    def _map_files(self, folder: Path) -> dict[str, Path]:
        """
        Return a dict mapping each file's stem -> its full Path,
        but only for files whose suffix is in our loader map.
        """
        return {
            p.stem: p
            for p in folder.iterdir()
            if p.is_file() and p.suffix in self._loaders
        }

    def _matching_groups(self) -> dict[str, list[Path]]:
        """
        Find all stems common across all folders.
        Returns a dict mapping stem -> list of Paths (one per folder).
        """
        maps = [self._map_files(f) for f in self.folders]
        # Intersection of all stems
        common_stems = set(maps[0])
        for m in maps[1:]:
            common_stems &= set(m)
        # Build list of paths for each common stem
        return {
            stem: [m[stem] for m in maps]
            for stem in sorted(common_stems)
        }

    def find_pairs(self, return_names: bool = False):
        """
        Load data for each matching filename across folders.

        Returns a list of lists `data_lists` where each inner list
        contains the loaded data for that folder.  If `return_names`
        is True, also returns the list of common stems.

        Example:
            data_lists, names = matcher.find_pairs(return_names=True)
            folder1_data = data_lists[0]
            folder2_data = data_lists[1]
            ...
        """
        groups = self._matching_groups()
        data_lists = [[] for _ in self.folders]
        names = []

        for stem, paths in groups.items():
            for idx, p in enumerate(paths):
                loader = self._loaders[p.suffix]
                data_lists[idx].append(loader(p))
            names.append(stem)

        if return_names:
            return data_lists, names
        return data_lists

    def save_matching_files(self,
                            dest_folders: list[str] = None):
        """
        Copy each set of matching files into parallel destination folders.

        - If `dest_folders` is None, each source folder `F` will map to
          a new folder named `F_new`.
        - Otherwise, `dest_folders` must be a list of the same length as
          the input folders, specifying each destination path.
        """
        # Determine destinations
        if dest_folders is None:
            dests = [Path(f"{f}_new") for f in self.folders]
        else:
            if len(dest_folders) != len(self.folders):
                raise ValueError("dest_folders must match number of input folders")
            dests = [Path(d) for d in dest_folders]

        # Create destination directories
        for d in dests:
            d.mkdir(exist_ok=True)

        # Copy files
        for stem, paths in self._matching_groups().items():
            for src, dst_folder in zip(paths, dests):
                shutil.copy2(src, dst_folder / src.name)
