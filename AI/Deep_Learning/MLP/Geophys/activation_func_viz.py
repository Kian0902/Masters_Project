# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 15:12:06 2024

@author: kian0
"""



import numpy as np
import matplotlib.pyplot as plt


def softplus(x, beta, threshold):
    # Use np.where to handle the threshold for numerical stability
    return np.where(
        x * beta > threshold,
        x,  # Linear approximation for large input
        (1 / beta) * np.log(1 + np.exp(beta * x))
    )

# Example usage:
x = np.linspace(-10, 1000, 1000)


y = softplus(x, beta=0.15, threshold=1)


plt.plot(x, y)
plt.show()

print(np.min(y), np.max(y))



























