# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 14:13:45 2024

@author: Kian Sartipzadeh
"""


from sorting_data import EISCATDataSorter



# Main path to Folder containing EISCAT data
path = "C:\\Users\\kian0\\OneDrive\\Desktop\\UiT Courses\\FYS_3931\\Scripts\\Masters_Project\\Eiscat\\Processing\\EISCAT_Ne"


A = EISCATDataSorter(path, "EISCAT_Ne")

A.sort_data(save_data=True)


data = A.return_data()



