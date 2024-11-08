# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 11:15:04 2024

@author: Kian Sartipzadeh
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import cv2
import numpy as np

# Load and preprocess image
def load_image(image_path):
    # Load image using OpenCV and convert it to RGB
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Resize image to a square for simplicity
    image = cv2.resize(image, (128, 128))
    # Normalize image to [0, 1]
    image = image / 255.0
    # Convert image to PyTorch tensor and add a batch dimension
    image_tensor = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
    return image_tensor

# Plotting function
def plot_image(image_tensor, title):
    # Remove batch dimension and permute to H x W x C for plotting
    image = image_tensor.squeeze().permute(1, 2, 0).detach().numpy()
    plt.imshow(image)
    plt.title(title)
    plt.axis('off')
    plt.show()

# Define maxpooling operation layers
maxpool_layers = [
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.MaxPool2d(kernel_size=2, stride=2)
]

# Main function
def main():
    # Load your image (provide the correct path to your image)
    image_path = 'your_image_path_here.jpg'
    image_tensor = load_image(image_path)

    # Plot the original image
    plot_image(image_tensor, 'Original Image')

    # Apply maxpooling layers and plot after each layer
    current_tensor = image_tensor
    for i, maxpool in enumerate(maxpool_layers):
        current_tensor = maxpool(current_tensor)
        plot_image(current_tensor, f'After MaxPool Layer {i+1}')

if __name__ == "__main__":
    main()





















