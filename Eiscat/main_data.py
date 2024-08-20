# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 14:13:45 2024

@author: Kian Sartipzadeh
"""


from sorting_data import EISCATDataSorter
from filter_data import BulkFiltering



# Main path to Folder containing EISCAT data
path = "C:\\Users\\kian0\\OneDrive\\Desktop\\UiT Courses\\FYS_3931\\Scripts\\Masters_Project\\Eiscat\\EISCAT_Ne"


A = EISCATDataSorter(path, "EISCAT_Ne")

A.sort_data()


data = A.return_data()

Filter = BulkFiltering(data)

filter_data = Filter.process_bulk()


