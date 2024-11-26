# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 13:09:04 2024

@author: Kian Sartipzadeh
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from scipy.stats import norm
sns.set(style="dark", context=None, palette=None)

# Step 1: Create a simplified 2D dataset with 2 blobs
X, y = make_blobs(n_samples=300, centers=[[5, 5], [10, 10]], cluster_std=1.5, random_state=42)

# Step 2: Standardize the dataset
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Step 3: Fit PCA to the standardized dataset
pca = PCA(n_components=2)
pca.fit(X)
X_pca = pca.transform(X)

# Step 4: Plot the original dataset with principal components
fig, ax = plt.subplots(figsize=(6, 6))

ax.scatter(X[y==0, 0], X[y==0, 1], color="C0", edgecolors="black", label="Class 1", zorder=2, alpha=0.8)
ax.scatter(X[y==1, 0], X[y==1, 1], color="C1", edgecolors="black", label="Class 2", zorder=2, alpha=0.8)

# Plot the principal components as arrows
mean = pca.mean_
components = pca.components_
explained_variance = pca.explained_variance_

for length, vector in zip(explained_variance, components):
    v = vector * 3 * np.sqrt(length)
    ax.arrow(mean[0], mean[1], v[0], v[1], color='r', width=0.04, head_width=0.2)
    ax.arrow(mean[0], mean[1], -v[0], -v[1], color='r', width=0.04, head_width=0.2)

ax.arrow(0, 0, 0, 0, color='r', width=0, head_width=0, label="Principal Components")
ax.set_title('Original Data with Principal Components', fontsize=15)
ax.set_xlabel('Feature 1', fontsize=15)
ax.set_ylabel('Feature 2', fontsize=15)
ax.axhline(0, color='black', lw=0.5)
ax.axvline(0, color='black', lw=0.5)
ax.grid(True)
ax.legend()
plt.gca().set_aspect(1)
plt.show()

# Step 5: Plot the data transformed onto the first principal component
fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(X_pca[y==0, 0], np.zeros_like(X_pca[y==0, 0]), alpha=0.8, edgecolors="black", label="Class 1", zorder=2)
ax.scatter(X_pca[y==1, 0], np.zeros_like(X_pca[y==1, 0]), alpha=0.8, edgecolors="black", label="Class 2", zorder=2)

# Plot the probability density functions for each blob
x_range = np.linspace(X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1, 1000)
mean_class_1 = X_pca[y==0, 0].mean()
std_class_1 = X_pca[y==0, 0].std()
mean_class_2 = X_pca[y==1, 0].mean()
std_class_2 = X_pca[y==1, 0].std()

ax.plot(x_range, norm.pdf(x_range, mean_class_1, std_class_1), color='C0', linestyle='--', zorder=1)
ax.plot(x_range, norm.pdf(x_range, mean_class_2, std_class_2), color='C1', linestyle='--', zorder=1)

ax.set_title('Data Transformed onto First Principal Component', fontsize=15)
ax.set_xlabel('Principal Component 1', fontsize=15)
ax.set_ylabel('Probability Density', fontsize=15)
ax.set_ylim(ymin=-0.05, ymax=1)
ax.axhline(0, color='black', lw=0.5, zorder=1)
# ax.axvline(0, color='black', lw=0.5)
ax.grid(True)
ax.legend()


plt.show()
