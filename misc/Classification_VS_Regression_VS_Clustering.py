# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 16:25:12 2024

@author: Kian Sartipzadeh
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs




# Generating data
X, y, centers = make_blobs(n_samples=300, centers=[[2, 2], [6, 6], [10, 9]], n_features=2, random_state=42, cluster_std=1.3, return_centers=True)
y_reg = np.sqrt(X[:, 0]**2 + X[:, 1]**2) + np.random.normal(0, 0.5, 300)



# Example Figure Classification
fig, ax = plt.subplots(figsize=(6, 5))
fig.suptitle("Classification", fontsize=20)
ax.scatter(X[:, 0], X[:, 1], c=y, edgecolor='k', marker='o', s=100)
ax.spines[['right', 'top', 'left' , 'bottom']].set_visible(False)
ax.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)
plt.savefig("Example_Classification.png", dpi=150, bbox_inches='tight')
plt.show()


# Example Figure Regression
fig, ax = plt.subplots(figsize=(6, 5))
fig.suptitle("Regression", fontsize=20)
ax.scatter(X[:, 0], X[:, 1], c=y_reg, edgecolor='k', marker='o', s=100)
ax.axline((0, 0.9), slope=0.8, color="black", linewidth=5)
ax.spines[['right', 'top', 'left' , 'bottom']].set_visible(False)
ax.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)
plt.savefig("Example_Regression.png", dpi=150, bbox_inches='tight')
plt.show()



# Example Figure Clustering
fig, ax = plt.subplots(figsize=(6, 5))
fig.suptitle("Clustering", fontsize=20)
ax.scatter(X[:, 0], X[:, 1], color="grey", edgecolor='k', marker='o', s=100)
circle1 = plt.Circle((centers[0][0], centers[0][1]), radius=3, linewidth=2, linestyle="--", fill=False)
circle2 = plt.Circle((centers[1][0], centers[1][1]), radius=3, linewidth=2, linestyle="--", fill=False)
circle3 = plt.Circle((centers[2][0], centers[2][1]), radius=3, linewidth=2, linestyle="--", fill=False)
ax.add_artist(circle1)
ax.add_artist(circle2)
ax.add_artist(circle3)
ax.spines[['right', 'top', 'left' , 'bottom']].set_visible(False)
ax.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)
plt.savefig("Example__Clustering.png", dpi=150, bbox_inches='tight')
plt.show()






