# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 16:25:12 2024

@author: Kian Sartipzadeh
"""


import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs


# Generating data
X, y, centers = make_blobs(n_samples=100, centers=2, n_features=2, random_state=42, cluster_std=2, return_centers=True)




# Example Figure Supervised
fig, ax = plt.subplots(figsize=(6, 5))
ax.scatter(X[y == 0][:, 0], X[y == 0][:, 1], color="C0", edgecolor='k', marker='o', s=100, label="Class 0")
ax.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color="C1", edgecolor='k', marker='o', s=100, label="Class 1")
ax.spines[['right', 'top']].set_visible(False)
ax.spines[['left' , 'bottom']].set_linewidth(2)
ax.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)
plt.show()


# Example Figure Unsupervised
fig, ax = plt.subplots(figsize=(6, 5))
ax.scatter(X[:, 0], X[:, 1], color="grey", edgecolor='k', marker='o', s=100)
circle1 = plt.Circle((centers[0][0],centers[0][1]), radius=4, linewidth=2, linestyle="--", fill=False)
circle2 = plt.Circle((centers[1][0],centers[1][1]+1), radius=4, linewidth=2, linestyle="--", fill=False)
ax.add_artist(circle1)
ax.add_artist(circle2)
ax.spines[['right', 'top']].set_visible(False)
ax.spines[['left' , 'bottom']].set_linewidth(2)
ax.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)
plt.show()






