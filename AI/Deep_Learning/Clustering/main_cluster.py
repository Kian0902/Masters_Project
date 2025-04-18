# -*- coding: utf-8 -*-
"""
Created on Thu Mar 20 13:30:46 2025

@author: kian0
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
import matplotlib.gridspec as gridspec
from scipy.stats import norm


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


# Define preprocessing and analysis functions
def normalize(data):
    """Normalize data by replacing NaNs and small values with 1e6, then taking log10."""
    data[np.isnan(data)] = 1e6
    data[data < 1e3] = 1e6
    return np.log10(data)

def apply_pca(data, n=2):
    """Apply PCA to reduce data to n components."""
    pca = PCA(n_components=n)
    pca.fit(data)
    return pca.transform(data)

def apply_gmm(data, n_components=4):
    """Apply GMM clustering with specified number of components."""
    gmm = GaussianMixture(n_components=n_components, random_state=43, init_params='kmeans')
    gmm.fit(data)
    y = gmm.predict(data)
    return gmm, y



def plot_pca(data, label):
    plt.scatter(data[:, 0], data[:, 1], s=3, c=label)
    plt.xlabel("pc0")
    plt.ylabel("pc1")
    plt.show()
    
    plt.scatter(data[:, 0], data[:, 2], s=3, c=label)
    plt.xlabel("pc0")
    plt.ylabel("pc2")
    plt.show()
    
    plt.scatter(data[:, 1], data[:, 2], s=3, c=label)
    plt.xlabel("pc1")
    plt.ylabel("pc2")
    plt.show()

def plot_pca_with_core(X, y, core_indices):
    """
    Plot the PCA scatter plots and overlay the 'core' samples (retrieved via core_indices)
    in red on top of the cluster-coloured points.

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)
        PCA-transformed data.
    y : ndarray, shape (n_samples,)
        Cluster labels for each sample.
    core_indices : dict
        Dictionary mapping cluster labels to indices of core samples.
    """
    # Plot for PC0 vs PC1
    plt.figure()
    plt.scatter(X[:, 0], X[:, 1], s=3, c=y, cmap='viridis', label='All samples')
    for k, indices in core_indices.items():
        plt.scatter(X[indices, 0], X[indices, 1], s=50, c='red', marker='o', 
                    label=f'Core Cluster {k}')
    plt.xlabel("PC0")
    plt.ylabel("PC1")
    plt.title("PCA (PC0 vs PC1) with Core Samples Highlighted")
    plt.legend()
    plt.show()

    # Plot for PC0 vs PC2
    plt.figure()
    plt.scatter(X[:, 0], X[:, 2], s=3, c=y, cmap='viridis', label='All samples')
    for k, indices in core_indices.items():
        plt.scatter(X[indices, 0], X[indices, 2], s=50, c='red', marker='o', 
                    label=f'Core Cluster {k}')
    plt.xlabel("PC0")
    plt.ylabel("PC2")
    plt.title("PCA (PC0 vs PC2) with Core Samples Highlighted")
    plt.legend()
    plt.show()

    # Plot for PC1 vs PC2
    plt.figure()
    plt.scatter(X[:, 1], X[:, 2], s=3, c=y, cmap='viridis', label='All samples')
    for k, indices in core_indices.items():
        plt.scatter(X[indices, 1], X[indices, 2], s=50, c='red', marker='o', 
                    label=f'Core Cluster {k}')
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("PCA (PC1 vs PC2) with Core Samples Highlighted")
    plt.legend()
    plt.show()



def plot_cluster_histograms(X, y, gmm, n_components=4):
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k'][:n_components]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for i in range(3):
        ax = axes[i]
        
        # Compute consistent bin edges
        bins = np.histogram_bin_edges(X[:, i], bins='auto')
        
        for k in range(n_components):
            
            cluster_data = X[y == k, i]
            
            ax.hist(cluster_data, bins=bins, density=True, alpha=0.5, color=colors[k])
            
            mu = gmm.means_[k, i]
            sigma = np.sqrt(gmm.covariances_[k, i, i])
            
            # PDF
            x_range = np.linspace(min(X[:, i]), max(X[:, i]), 100)
            pdf = norm.pdf(x_range, loc=mu, scale=sigma)
            
            
            ax.plot(x_range, pdf, color=colors[k], label=f'Cluster {k}')
            
        # Set title and add legend to the first subplot only
        ax.set_title(f'PC {i}')
        ax.set_xlabel('Samples')
        ax.set_ylabel('Prob Density')
        if i == 0:
            ax.legend()
    
    # Adjust layout and display
    plt.tight_layout()
    plt.show()

def get_cluster_indices(data, label, gmm, threshold_ratio=0.5):
    """
    Retrieve the indices of samples in each cluster whose likelihood is at least
    threshold_ratio (default 0.5, i.e. 50%) of the peak value of the cluster's
    Gaussian density. This is done using the Mahalanobis distance threshold.
    
    For a Gaussian, the condition:
        pdf(x) >= threshold_ratio * pdf(μ)
    leads to:
        (x - μ)^T Σ⁻¹ (x - μ) <= -2 * ln(threshold_ratio)
    
    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)
        Data (e.g., PCA-transformed) on which GMM was applied.
    y : ndarray, shape (n_samples,)
        Cluster labels for each sample.
    gmm : GaussianMixture object
        A fitted Gaussian Mixture Model.
    threshold_ratio : float, optional (default=0.5)
        Ratio of the maximum probability density (at the mean) to use as the threshold.
    
    Returns
    -------
    core_indices : dict
        Dictionary where keys are cluster labels and values are arrays of indices
        corresponding to the samples that are within the threshold region.
    """
    core_indices = {}
    # Compute the Mahalanobis distance squared threshold.
    # For threshold_ratio=0.5, this equals 2*ln2.
    threshold = -2 * np.log(threshold_ratio)
    
    # Process each cluster separately
    for k in np.unique(label):
        # Get indices and data points for cluster k
        indices = np.where(label == k)[0]
        X_cluster = data[indices]
        
        # Compute the difference from the cluster mean
        diff = X_cluster - gmm.means_[k]
        # Invert the covariance matrix for the current cluster.
        inv_cov = np.linalg.inv(gmm.covariances_[k])
        
        # Compute squared Mahalanobis distances:
        # For each sample: d² = (x - μ)^T Σ⁻¹ (x - μ)
        d2 = np.sum(diff @ inv_cov * diff, axis=1)
        
        # Select indices where the squared distance is within the threshold.
        core_indices[k] = indices[d2 <= threshold]
    
    return core_indices



def merge_core_indices(indices, label1, label2):

    # Check that both labels exist in the dictionary
    if label1 not in indices or label2 not in indices:
        raise ValueError("Both labels must be present in the indices dictionary.")
    
    # Define the merged label (using the smaller label)
    merged_label = min(label1, label2)
    
    # Combine the indices from both clusters
    merged_values = np.concatenate([indices[label1], indices[label2]])
    
    # Create a new dictionary without the two original keys
    merged_ind = {k: v for k, v in indices.items() if k not in (label1, label2)}
    
    # Add the merged cluster using the merged_label
    merged_ind[np.int64(merged_label)] = merged_values
    
    return merged_ind

def augment_clusters(data, labels, indices, n_new, random_state=None):
    """
    Augment each cluster by randomly sampling additional data points from the given cluster indices.
    
    Parameters
    ----------
    data : ndarray, shape (n_samples, n_features)
        Original dataset (e.g., PCA-transformed data).
    labels : ndarray, shape (n_samples,)
        Original cluster labels for each data point.
    indices : dict
        Dictionary where keys are cluster labels and values are arrays of indices corresponding
        to the core samples (or any samples) in that cluster.
    n_new : int
        Number of new data points to sample (with replacement) from each cluster.
    random_state : int or None, optional
        Seed for reproducibility.
        
    Returns
    -------
    augmented_data : ndarray, shape (n_samples + n_clusters*n_new, n_features)
        New dataset with additional samples appended.
    augmented_labels : ndarray, shape (n_samples + n_clusters*n_new,)
        New labels array corresponding to augmented_data.
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    extra_data_list = []
    extra_labels_list = []
    
    for k, idx_array in indices.items():
        # Randomly sample n_new indices from the given cluster indices (with replacement)
        sampled_idx = np.random.choice(idx_array, size=n_new, replace=True)
        extra_data_list.append(data[sampled_idx])
        extra_labels_list.append(labels[sampled_idx])
    
    # Concatenate additional samples for all clusters
    if extra_data_list:
        extra_data = np.concatenate(extra_data_list, axis=0)
        extra_labels = np.concatenate(extra_labels_list, axis=0)
        augmented_data = np.concatenate([data, extra_data], axis=0)
        augmented_labels = np.concatenate([labels, extra_labels], axis=0)
    else:
        augmented_data = data
        augmented_labels = labels
    
    return augmented_data, augmented_labels


def augment_pca_cores(pca_data, core_indices, n_new, random_state=None):
    """
    Augment the PCA cores by randomly sampling additional data points from the core
    points of each cluster. This function assumes that core_indices is a dictionary
    mapping each cluster label to the indices (into pca_data) of its core samples.

    Parameters
    ----------
    pca_data : ndarray, shape (n_samples, n_features)
        The PCA-transformed data (from which the cores have been selected).
    core_indices : dict
        Dictionary with keys as cluster labels and values as numpy arrays of indices
        corresponding to the core samples in that cluster.
    n_new : int
        Number of additional samples to randomly generate (with replacement) for each cluster.
    random_state : int or None, optional
        Seed for reproducibility.
        
    Returns
    -------
    augmented_core_data : ndarray, shape (n_total, n_features)
        The new core dataset containing the original core samples plus the additional samples.
    augmented_core_labels : ndarray, shape (n_total,)
        The cluster labels corresponding to augmented_core_data.
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    augmented_data_list = []
    augmented_labels_list = []
    
    # Process each cluster in the cores
    for cluster, indices in core_indices.items():
        # Get the original core data for this cluster
        cluster_core = pca_data[indices]
        n_core = cluster_core.shape[0]
        
        # Append the original core data and labels
        augmented_data_list.append(cluster_core)
        augmented_labels_list.append(np.full(n_core, cluster))
        
        # Sample additional points from the core indices
        sampled_idx = np.random.choice(indices, size=n_new, replace=True)
        extra_data = pca_data[sampled_idx]
        
        augmented_data_list.append(extra_data)
        augmented_labels_list.append(np.full(n_new, cluster))
    
    # Concatenate all clusters together
    augmented_core_data = np.concatenate(augmented_data_list, axis=0)
    augmented_core_labels = np.concatenate(augmented_labels_list, axis=0)
    
    return augmented_core_data, augmented_core_labels




def plot_interactive_pca(data, label, norm_x):
    
    
    fig = plt.figure(figsize=(10, 15))
    gs = gridspec.GridSpec(3, 2)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[2, 0])
    ax4 = fig.add_subplot(gs[:, 1])

    
    ax1.scatter(data[:, 0], data[:, 1], s=3, c=label)
    ax1.set_xlabel("PC0")
    ax1.set_ylabel("PC1")
    
    
    ax2.scatter(data[:, 0], data[:, 2], s=3, c=label)
    ax2.set_xlabel("PC0")
    ax2.set_ylabel("PC2")

    ax3.scatter(data[:, 1], data[:, 2], s=3, c=label)
    ax3.set_xlabel("PC1")
    ax3.set_ylabel("PC2")

    
    def on_click(event):
        """Handle click events on PCA plots to update the original data plot."""
        if event.inaxes == ax1:
            col1, col2 = 0, 1
        elif event.inaxes == ax2:
            col1, col2 = 0, 2
        elif event.inaxes == ax3:
            col1, col2 = 1, 2
        else:
            return


        dist = np.sqrt((data[:, col1] - event.xdata)**2 + (data[:, col2] - event.ydata)**2)
        idx = np.argmin(dist)

        cluster_label = label[idx]
        
        ax4.clear()
        ax4.plot(norm_x[idx, :], r_h)
        ax4.set_xlabel('Altitude Index')
        ax4.set_ylabel('log10(Density)')
        ax4.set_title(f'Cluster: {cluster_label}', fontsize=12)
        fig.canvas.draw()

    fig.canvas.mpl_connect('button_press_event', on_click)
    plt.tight_layout()
    plt.show()




x = np.load('train_data.npy')
norm_x = normalize(x)
X = apply_pca(norm_x, n=5)
gmm, y_label = apply_gmm(X, n_components=4)

ind = get_cluster_indices(X, y_label, gmm, threshold_ratio=0.5)


idx = merge_core_indices(ind, 1, 0)


index = np.concatenate(list(idx.values()))


y_label[y_label==0] = 1  # Merging labels


# plot_interactive_pca(X[index], y_label[index], norm_x[index])



# aug_data, aug_labels = augment_clusters(X[index], y_label[index], idx, n_new=500, random_state=42)



# plot_interactive_pca(aug_data[index], aug_labels[index], norm_x[index])
aug_data, aug_labels = augment_pca_cores(X, idx, 1000)


plot_pca(aug_data, aug_labels)




