# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 18:15:43 2024

@author: Kian Sartipzadeh
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_blobs
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from matplotlib.lines import Line2D
sns.set(style="dark", context=None, palette=None)


# Create a 2D blobs dataset with 3 clusters
n_samples = 500
n_clusters = 3
X, y_true = make_blobs(n_samples=n_samples, centers=[[0, 0], [0, 0], [0, 0]], n_features=2, random_state=42)

# Apply a transformation to make the blobs have different shapes (e.g., ovals)
scaler = StandardScaler()
scales = [np.array([[4, 1], [1, 4]]), np.array([[-1, 3], [-1, 1]]), np.array([[4, 1], [-1, 1]])]

for i in range(n_clusters):
    X[y_true == i] = X[y_true == i].dot(scales[i])

# Plot the original dataset
fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(X[:, 0], X[:, 1], color='C0', s=30, alpha=0.8)
ax.set_title("Original Dataset", fontsize=15)
ax.set_xlabel("Feature 1", fontsize=15)
ax.set_ylabel("Feature 2", fontsize=15)
ax.grid(True)
plt.show()

# Apply Gaussian Mixture Model
gmm = GaussianMixture(n_components=n_clusters, random_state=42,)
labels = gmm.fit_predict(X)


# Plot the original dataset
fig, ax = plt.subplots(figsize=(6, 6))
sns.scatterplot(x=X[y_true == 0, 0], y=X[y_true == 0, 1], ax=ax, label="true label 0", color="C0", s=30, edgecolor="black", linewidth=0.5, zorder=1)
sns.scatterplot(x=X[y_true == 1, 0], y=X[y_true == 1, 1], ax=ax, label="true label 1", color="C1", s=30, edgecolor="black", linewidth=0.5, zorder=3)
sns.scatterplot(x=X[y_true == 2, 0], y=X[y_true == 2, 1], ax=ax, label="true label 2", color="red", s=30, edgecolor="black", linewidth=0.5, zorder=2)
ax.set_title("Original Dataset", fontsize=15)
ax.set_xlabel("Feature 1", fontsize=15)
ax.set_ylabel("Feature 2", fontsize=15)
ax.set_xlim(xmin=-15.2, xmax=15.2)
ax.set_ylim(ymin=-15.2, ymax=15.2)
ax.grid(True)
plt.show()



fig, ax = plt.subplots(figsize=(6, 6))
sns.kdeplot(x=X[labels == 0, 0], y=X[labels == 0, 1], fill=False, ax=ax, levels=10, color="C0", zorder=1)
sns.kdeplot(x=X[labels == 1, 0], y=X[labels == 1, 1], fill=False, ax=ax, levels=10, color="C1", zorder=2)
sns.kdeplot(x=X[labels == 2, 0], y=X[labels == 2, 1], fill=False, ax=ax, levels=10, color="red")
ax.set_title("KDE plot", fontsize=15)
ax.set_xlabel("Feature 1", fontsize=15)
ax.set_ylabel("Feature 2", fontsize=15)
ax.set_xlim(xmin=-15.2, xmax=15.2)
ax.set_ylim(ymin=-15.2, ymax=15.2)
ax.grid()
kde_legend = [
    Line2D([0], [0], color="C0", lw=2, label="Cluster 0"),
    Line2D([0], [0], color="C1", lw=2, label="Cluster 1"),
    Line2D([0], [0], color="red", lw=2, label="Cluster 2"),
]
ax.legend(handles=kde_legend, fontsize=9.5, frameon=True)
plt.show()





