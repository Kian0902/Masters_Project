# -*- coding: utf-8 -*-
"""
Created on Sun Nov  3 16:29:46 2024

@author: Kian Sartipzadeh
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image in grayscale
image = cv2.imread('Me_Goofy.png', cv2.IMREAD_GRAYSCALE)

# Apply Sobel filter for vertical edges
sobel_vertical = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=9)
# sobel_vertical = cv2.convertScaleAbs(sobel_vertical)

# Apply Sobel filter for horizontal edges
sobel_horizontal = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=9)
# sobel_horizontal = cv2.convertScaleAbs(sobel_horizontal)

# Plotting the results
plt.figure(figsize=(10, 5))

# Original image
# plt.subplot(1, 3, 1)
plt.figure(figsize=(10, 5))
plt.title("Original Image")
plt.imshow(image, cmap='gray')
plt.axis('off')
plt.show()

# Vertical edges
plt.figure(figsize=(10, 5))
plt.title("Vertical Edges")
plt.imshow(sobel_vertical, cmap='gray')
plt.axis('off')
plt.show()

# Horizontal edges
plt.figure(figsize=(10, 5))
plt.title("Horizontal Edges")
plt.imshow(sobel_horizontal, cmap='gray')
plt.axis('off')
plt.show()
