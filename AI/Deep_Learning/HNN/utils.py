# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 12:06:41 2024

@author: Kian Sartipzadeh
"""


import matplotlib.pyplot as plt
from torchvision import transforms





class CNNShapeCalculator:
    def __init__(self, input_shape):
        
        self.input_shape = input_shape
        self.current_shape = input_shape

    def conv2d(self, kernel_size, stride=1, padding=0, dilation=1):
        
        if isinstance(stride, int):
            stride = (stride, stride)
        if isinstance(padding, int):
            padding = (padding, padding)
        if isinstance(dilation, int):
            dilation = (dilation, dilation)

        h_in, w_in, c_in = self.current_shape

        h_out = (h_in + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) // stride[0] + 1
        w_out = (w_in + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) // stride[1] + 1

        self.current_shape = (h_out, w_out, c_in)

        return self.current_shape

    def max_pool2d(self, pool_size, stride=None, padding=0):
        
        if stride is None:
            stride = pool_size
        if isinstance(stride, int):
            stride = (stride, stride)
        if isinstance(padding, int):
            padding = (padding, padding)

        h_in, w_in, c_in = self.current_shape

        h_out = (h_in + 2 * padding[0] - pool_size[0]) // stride[0] + 1
        w_out = (w_in + 2 * padding[1] - pool_size[1]) // stride[1] + 1

        self.current_shape = (h_out, w_out, c_in)

        return self.current_shape

    def get_current_shape(self):

        
        return self.current_shape


def plot_ionogram(ionogram_image):
    ionogram_image = transforms.ToPILImage()(ionogram_image)
    
    fig, ax = plt.subplots()
    
    ax.imshow(ionogram_image)
    ax.axis("on")
    plt.show()













































