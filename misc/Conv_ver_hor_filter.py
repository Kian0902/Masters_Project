# -*- coding: utf-8 -*-
"""
Created on Sun Nov  3 16:29:46 2024

@author: Kian Sartipzadeh
"""


import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d



def main():
    # Load
    image = cv2.imread('Iono_ex_color.png', cv2.IMREAD_GRAYSCALE)
    print(image.shape)
    
    if image is None:
        print("Error: Image not found.")
        return

    # Define Sobel kernels for edge detection with filter size 5
    sobel_vertical = np.array([[-1, -2, 0, 2, 1],
                               [-4, -8, 0, 8, 4],
                               [-6, -12, 0, 12, 6],
                               [-4, -8, 0, 8, 4],
                               [-1, -2, 0, 2, 1]])
    
    sobel_horizontal = np.array([[1, 4, 6, 4, 1],
                                 [2, 8, 12, 8, 2],
                                 [0, 0, 0, 0, 0],
                                 [-2, -8, -12, -8, -2],
                                 [-1, -4, -6, -4, -1]])
    # Convolve the image with the vertical and horizontal kernels using scipy's convolve2d
    vertical_edges = convolve2d(image, sobel_vertical, mode='same', boundary='fill', fillvalue=0)
    horizontal_edges = convolve2d(image, sobel_horizontal, mode='same', boundary='fill', fillvalue=0)
    
    
    # Create an RGB version of the original image with only Red and Green channels
    rgb_image = cv2.imread('Iono_ex_color.png')
    if rgb_image is None:
        print("Error: Image not found.")
        return
    # Switch the Red and Blue channels
    rgb_image[:, :, [0, 2]] = rgb_image[:, :, [2, 0]]
    rgb_image[:, :, 2] = 0  # Set the Blue channel to 0 (keep only Red and Green)
    
    
    # Plotting the results
    plt.figure(figsize=(10, 5))
    plt.title("Original Image", fontsize=20)
    plt.imshow(rgb_image)
    plt.axis('off')
    plt.show()

    # Vertical edges
    plt.figure(figsize=(10, 5))
    plt.title("Vertical Edges", fontsize=20)
    plt.imshow(vertical_edges, cmap='gray')
    plt.axis('off')
    plt.show()

    # Horizontal edges
    plt.figure(figsize=(10, 5))
    plt.title("Horizontal Edges", fontsize=20)
    plt.imshow(horizontal_edges, cmap='gray')
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    main()