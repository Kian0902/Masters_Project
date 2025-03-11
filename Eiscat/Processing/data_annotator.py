# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 20:24:40 2025

@author: kian0
"""


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import shutil



r_h = np.array([[ 91.46317376],
       [ 94.45832965],
       [ 97.58579014],
       [100.70052912],
       [103.70339597],
       [106.70131251],
       [109.94666419],
       [113.57783654],
       [118.09858718],
       [123.86002718],
       [130.60948667],
       [138.3173866 ],
       [147.00535981],
       [156.73585448],
       [167.59278336],
       [179.57721244],
       [192.43624664],
       [206.30300179],
       [221.17469246],
       [237.20685488],
       [254.39143579],
       [272.34757393],
       [291.47736465],
       [311.68243288],
       [332.82898428],
       [354.92232018],
       [377.86223818]]).flatten()

    
    
    



# Define the source and destination folders
source_folder = 'EISCAT_Samples'  # Folder with all CSV files
healthy_folder = 'EISCAT_Healthy'  # Folder for healthy CSV files

# Create the healthy folder if it doesn’t exist
os.makedirs(healthy_folder, exist_ok=True)

# Get a list of CSV files that aren’t already in the healthy folder
csv_files = [
    f for f in os.listdir(source_folder)
    if f.endswith('.csv') and not os.path.exists(os.path.join(healthy_folder, f))
]

# Global variable to store your decision
decision = None

# Callback function for the "Include" button
def include_callback(event):
    global decision
    decision = 'include'
    plt.close()

# Callback function for the "Skip" button
def skip_callback(event):
    global decision
    decision = 'skip'
    plt.close()

# Process each CSV file one at a time
for csv_file in csv_files:
    # Full path to the current file
    file_path = os.path.join(source_folder, csv_file)
    
    # Read the CSV file (assuming no header)
    df = pd.read_csv(file_path, header=None)
    
    # Extract 27 electron density values
    if df.shape[0] == 27 and df.shape[1] == 1:
        # Case: 27 rows, 1 column
        data = df[0].values
    elif df.shape[0] == 1 and df.shape[1] == 27:
        # Case: 1 row, 27 columns
        data = df.iloc[0].values
    else:
        # Skip files with unexpected shapes and notify you
        print(f"Skipping {csv_file}: Unexpected shape {df.shape}")
        continue
    
    # Create a plot for the current file
    fig, ax = plt.subplots()
    ax.plot(data, r_h)
    ax.set_xscale('log')
    ax.set_title(csv_file)  # Show the filename as the plot title
    
    # Add "Include" and "Skip" buttons
    ax_include = plt.axes([0.7, 0.05, 0.1, 0.075])  # Position: [left, bottom, width, height]
    ax_skip = plt.axes([0.81, 0.05, 0.1, 0.075])
    b_include = Button(ax_include, 'Include')
    b_skip = Button(ax_skip, 'Skip')
    
    # Link buttons to their callback functions
    b_include.on_clicked(include_callback)
    b_skip.on_clicked(skip_callback)
    
    # Display the plot
    plt.show()
    
    # Wait for your decision before moving on
    while decision is None:
        plt.pause(0.1)  # Brief pause to keep the plot responsive
    
    # Handle your decision
    if decision == 'include':
        shutil.copy(file_path, os.path.join(healthy_folder, csv_file))
    
    # Reset decision for the next file
    decision = None
