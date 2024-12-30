# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 13:24:43 2024

@author: Kian Sartipzadeh
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn.cluster import DBSCAN
from sklearn.metrics import pairwise_distances
from matplotlib.lines import Line2D
from matplotlib.widgets import Cursor
import seaborn as sns
from matplotlib.gridspec import GridSpec
sns.set(style="dark", context=None, palette=None)

class GMMClusterPlotter:
    def __init__(self, X_pca, n_components, random_state=42):
        self.X_pca = X_pca
        self.n_components = n_components
        self.random_state = random_state
        self.gmm = GaussianMixture(n_components=n_components, random_state=random_state)
        self.labels = None
        self.cluster_counts = {}

    def fit_predict(self):
        self.gmm.fit(self.X_pca)
        self.labels = self.gmm.predict(self.X_pca)
        self._compute_cluster_counts()

    def _compute_cluster_counts(self):
        unique_labels = np.unique(self.labels)
        for label in unique_labels:
            self.cluster_counts[label] = np.sum(self.labels == label)

    def get_cluster_samples(self, n_samples):
        """
        Get the indices of `n_samples` closest points to the cluster mean for each cluster.

        Parameters:
            n_samples (int): Number of closest samples to retrieve per cluster.

        Returns:
            closest_indices (list): List of indices for closest samples to each cluster mean.
            closest_labels (list): List of corresponding cluster labels for these indices.
        """
        closest_indices = []
        closest_labels = []
        unique_labels = np.unique(self.labels)  # Unique cluster labels

        for cluster_label in unique_labels:
            # Get the indices and samples of the current cluster
            cluster_indices = np.where(self.labels == cluster_label)[0]
            cluster_samples = self.X_pca[cluster_indices]

            # Compute the cluster mean
            cluster_mean = np.mean(cluster_samples, axis=0)

            # Compute distances to the cluster mean
            distances = np.linalg.norm(cluster_samples - cluster_mean, axis=1)

            # Get indices of the n_samples closest samples
            sorted_indices = cluster_indices[np.argsort(distances)[:n_samples]]

            # Append to results
            closest_indices.append(sorted_indices)
            closest_labels.extend([cluster_label] * len(sorted_indices))

        # Flatten the list of indices for compatibility
        closest_indices = np.concatenate(closest_indices)
        return closest_indices, np.array(closest_labels)

    def plot_clusters(self):
        pairs = [(0, 1), (0, 2), (1, 2)]
        colors = [f"C{i}" for i in range(self.n_components)]

        for pair in pairs:
            fig, ax = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

            # Scatter plot
            for label, count in self.cluster_counts.items():
                sns.scatterplot(
                    x=self.X_pca[self.labels == label, pair[0]],
                    y=self.X_pca[self.labels == label, pair[1]],
                    ax=ax[0],
                    label=f"Cluster {label}: {count} samples",
                    color=colors[label],
                    s=30,
                    edgecolor="black",
                    linewidth=0.5
                )
            ax[0].set_title(f"Scatter plot (PC {pair[0] + 1} vs PC {pair[1] + 1})", fontsize=15)
            ax[0].set_xlabel(f"PC {pair[0] + 1}", fontsize=12)
            ax[0].set_ylabel(f"PC {pair[1] + 1}", fontsize=12)
            ax[0].grid()
            ax[0].legend()

            # KDE plot
            for label in self.cluster_counts.keys():
                sns.kdeplot(
                    x=self.X_pca[self.labels == label, pair[0]],
                    y=self.X_pca[self.labels == label, pair[1]],
                    fill=False,
                    ax=ax[1],
                    levels=10,
                    color=colors[label]
                )
            ax[1].set_title(f"KDE plot (PC {pair[0] + 1} vs PC {pair[1] + 1})", fontsize=15)
            ax[1].set_xlabel(f"PC {pair[0] + 1}", fontsize=12)
            ax[1].grid()

            kde_legend = [
                Line2D([0], [0], color=colors[label], lw=2, label=f"Cluster {label}")
                for label in self.cluster_counts.keys()
            ]
            ax[1].legend(handles=kde_legend, fontsize=9.5, frameon=True)

            plt.tight_layout()
            plt.show()

    def plot_additional_kde_and_samples(self, closest_indices, closest_labels, X_original, r):
        pairs = [(0, 1), (0, 2)]
        colors = [f"C{i}" for i in range(self.n_components)]

        for pair in pairs:
            fig, ax = plt.subplots(figsize=(6, 6))

            for label in self.cluster_counts.keys():
                sns.kdeplot(
                    x=self.X_pca[self.labels == label, pair[0]],
                    y=self.X_pca[self.labels == label, pair[1]],
                    fill=False,
                    ax=ax,
                    levels=10,
                    color=colors[label]
                )
            sns.scatterplot(
                x=self.X_pca[closest_indices, pair[0]],
                y=self.X_pca[closest_indices, pair[1]],
                ax=ax,
                color="red",
                s=50,
                marker="o",
                edgecolor="black",
                zorder=4
            )

            ax.set_title(f"KDE with Closest Samples (PC {pair[0] + 1} vs PC {pair[1] + 1})", fontsize=15)
            ax.set_xlabel(f"PC {pair[0] + 1}", fontsize=15)
            ax.set_ylabel(f"PC {pair[1] + 1}", fontsize=15)
            ax.grid()

            kde_legend = [
                Line2D([0], [0], color=colors[label], lw=2, label=f"Cluster {label}")
                for label in self.cluster_counts.keys()
            ]
            kde_legend.append(
                Line2D([0], [0], marker="o", color="red", markersize=6, label="Closest Samples", linestyle="None", markeredgecolor="black")
            )
            ax.legend(handles=kde_legend, loc="lower left", fontsize=9.5, frameon=True)

            plt.show()

        # Electron Density Plot
        fig, ax = plt.subplots(figsize=(6, 6))
        for i in range(len(closest_indices)):
            ax.plot(
                np.log10(X_original[int(closest_indices[i]), :]),
                r,
                c=colors[int(closest_labels[i])]
            )

        ax.set_title("Electron Density Plot", fontsize=15)
        ax.set_xlabel(r"log$_{10}$(n_e) [g/cm$^2$]", fontsize=13)
        ax.set_ylabel("Altitude [km]", fontsize=13)
        ax.grid()
        plot_legend = [
            Line2D([0], [0], color=colors[label], lw=2, label=f"Cluster {label}")
            for label in self.cluster_counts.keys()
        ]
        ax.legend(handles=plot_legend, loc="upper left", fontsize=9.5, frameon=True)

        plt.show()

    def plot_interactive_contour(self, X_original, r):
        """
        Create an interactive plot with a contour and corresponding X_original visualization.
        """

        fig = plt.figure(figsize=(14, 5))
        gs = GridSpec(1, 4)

        ax0 = fig.add_subplot(gs[0, 0])
        ax1 = fig.add_subplot(gs[0, 1])
        ax2 = fig.add_subplot(gs[0, 2])
        colors = [f"C{i}" for i in range(self.n_components)]

        for label in self.cluster_counts.keys():
            sns.kdeplot(
                x=self.X_pca[self.labels == label, 0],
                y=self.X_pca[self.labels == label, 1],
                fill=False,
                ax=ax0,
                levels=10,
                color=colors[label]
            )

            sns.kdeplot(
                x=self.X_pca[self.labels == label, 0],
                y=self.X_pca[self.labels == label, 2],
                fill=False,
                ax=ax1,
                levels=10,
                color=colors[label]
            )

            sns.kdeplot(
                x=self.X_pca[self.labels == label, 1],
                y=self.X_pca[self.labels == label, 2],
                fill=False,
                ax=ax2,
                levels=10,
                color=colors[label]
            )

        # Detail plot axes
        with sns.axes_style("dark"):
            ax_eis = fig.add_subplot(gs[0, 3])

        # Red dot to show the clicked point
        red_dot0, = ax0.plot([], [], "ro")
        red_dot1, = ax1.plot([], [], "ro")
        red_dot2, = ax2.plot([], [], "ro")

        # Placeholder for electron density plot
        line_eis, = ax_eis.plot([], [])
        ax_eis.set_title("Electron Density", fontsize=15)
        ax_eis.set_xlabel(r"log$_{10}$(n_e) [g/cm$^2$]", fontsize=12)
        ax_eis.set_ylabel("Altitude [km]", fontsize=12)
        ax_eis.grid()

        def onclick(event):
            nearest_idx = None

            if event.inaxes == ax0 and event.xdata is not None and event.ydata is not None:
                # Get the nearest point in the dataset for PC1 vs PC2
                distances = np.linalg.norm(self.X_pca[:, :2] - [event.xdata, event.ydata], axis=1)
                nearest_idx = np.argmin(distances)

            elif event.inaxes == ax1 and event.xdata is not None and event.ydata is not None:
                # Get the nearest point in the dataset for PC1 vs PC3
                distances = np.linalg.norm(self.X_pca[:, [0, 2]] - [event.xdata, event.ydata], axis=1)
                nearest_idx = np.argmin(distances)

            elif event.inaxes == ax2 and event.xdata is not None and event.ydata is not None:
                # Get the nearest point in the dataset for PC2 vs PC3
                distances = np.linalg.norm(self.X_pca[:, 1:] - [event.xdata, event.ydata], axis=1)
                nearest_idx = np.argmin(distances)

            if nearest_idx is not None:
                # Update the red dot positions across all plots
                red_dot0.set_data([self.X_pca[nearest_idx, 0]], [self.X_pca[nearest_idx, 1]])
                red_dot1.set_data([self.X_pca[nearest_idx, 0]], [self.X_pca[nearest_idx, 2]])
                red_dot2.set_data([self.X_pca[nearest_idx, 1]], [self.X_pca[nearest_idx, 2]])

                # Update the electron density plot with the corresponding cluster color
                cluster_label = self.labels[nearest_idx]
                line_eis.set_data(np.log10(X_original[nearest_idx, :]), r)
                line_eis.set_color(colors[cluster_label])

                # Rescale the density plot
                ax_eis.relim()
                ax_eis.autoscale_view()

                # Redraw the updated plot
                fig.canvas.draw_idle()

        # Connect the event handler
        fig.canvas.mpl_connect("button_press_event", onclick)
        Cursor(ax0, useblit=True, color="red", linewidth=1)
        Cursor(ax1, useblit=True, color="red", linewidth=1)
        Cursor(ax2, useblit=True, color="red", linewidth=1)
        plt.tight_layout()
        plt.show()

            
def import_file(file_name):
    with open(file_name, 'rb') as f:
        dataset = pickle.load(f)
    
    return dataset



def detect_nan_in_arrays(data_dict):
    for key, array in data_dict.items():
        if np.isnan(array).any():
            print(f"NaN detected {key}: {array.shape}")



def create_ml_dataset(data):
    samples = []
    
    for date, day_data in data.items():
        r_time = day_data["r_time"]  # Shape (M, 6)
        r_param = day_data["r_param"]  # Shape (N, M)
        r_h = day_data["r_h"]         # Shape (N, 1)
        
        # Transpose r_param to make it (M, N)
        r_param_transposed = r_param.T
        
        
        # Iterate over each measurement (M)
        for i in range(r_time.shape[0]):
            # Combine date-time information with features
            date_time_info = r_time[i]  # Shape (6,)
            features = r_param_transposed[i]  # Shape (N,)
            sample = np.concatenate([date_time_info, features])  # Concatenate date-time and features
            samples.append(sample)
    
    # Convert the list of samples to a numpy array
    dataset_array = np.array(samples)
    X = dataset_array[:,6:]
    r = r_h.flatten()
    times = dataset_array[:,:6]
    return X, r, times







def pca(data: np.ndarray, reduce_to_dim: int=2):
    """
    Reduces the data features into 2 dimensions.
    
    
    Input (type)    | DESCRIPTION
    ------------------------------------------------
    data (np.array) | Data to be reduced.
    
    
    Return (type)          | DESCRIPTION
    ------------------------------------------------
    pca_data (np.ndarray)  | Reduced data.
    """
    
    PCA_model = PCA(n_components = reduce_to_dim)
    pca_data = PCA_model.fit_transform(data.T)
    return pca_data.T


def select_random_indices(data, n):
    """
    Select `n` random indices from the dataset.

    Parameters:
        data (np.ndarray): The dataset to sample from.
        n (int): Number of random indices to select.

    Returns:
        np.ndarray: Array of randomly selected indices.
    """
    return np.random.choice(data.shape[0], size=n, replace=False)



data = import_file("X_avg_train_data")


X, r, t = create_ml_dataset(data)
X[X < 1e6] = 1e6



# # =============================================================================
# #                             PCA analysis plot
# # =============================================================================


# pca = PCA()
# pca.fit(np.log10(X))

# # Explained variance ratio
# explained_variance_ratio = pca.explained_variance_ratio_
# cumulative_variance_ratio = np.cumsum(explained_variance_ratio)

# retain_var=0.90
# n_components = np.argmax(cumulative_variance_ratio >= retain_var) + 1


# plt.figure(figsize=(8, 6))
# plt.plot(range(1, len(cumulative_variance_ratio) + 1), cumulative_variance_ratio, marker='o', label='Cumulative Variance')
# plt.axhline(y=retain_var, color='r', linestyle='--', label=f'{retain_var*100}% Threshold')
# plt.xlabel('Number of Principal Components', fontsize=15)
# plt.ylabel('Cumulative Explained Variance', fontsize=15)
# plt.title('PCA Cumulative Explained Variance', fontsize=15)
# plt.legend()
# plt.grid()
# plt.show()




PCA_model = PCA(n_components = 3)
X_pca = PCA_model.fit_transform(np.log10(X))


n=select_random_indices(X_pca, X.shape[0])

# # =============================================================================
# #                        Data plots after PCA
# # =============================================================================


# #____________ PC 1 and 2 ____________
# fig, ax = plt.subplots(figsize=(6, 6))
# fig.tight_layout()

# # Plot for Principal Components 1 and 2
# sns.scatterplot(x=X_pca[:n, 0], y=X_pca[:n, 1], ax=ax, label="Reduced Electron Densities", color="C0", s=30, edgecolor="black", linewidth=0.5)
# ax.set_title("Dataset after PCA feature reduction", fontsize=15)
# ax.set_xlabel("PC 1", fontsize=15)
# ax.set_ylabel("PC 2", fontsize=15)
# ax.grid()
# ax.legend()
# plt.show()



# #____________ PC 1 and 3 ____________
# fig, ax = plt.subplots(figsize=(6, 6))
# fig.tight_layout()

# # Plot for Principal Components 1 and 2
# sns.scatterplot(x=X_pca[:n, 0], y=X_pca[:n, 2], ax=ax, label="Reduced Electron Densities", color="C0", s=30, edgecolor="black", linewidth=0.5)
# ax.set_title("Dataset after PCA feature reduction", fontsize=15)
# ax.set_xlabel("PC 1", fontsize=15)
# ax.set_ylabel("PC 3", fontsize=15)
# ax.grid()
# ax.legend()
# plt.show()



# #____________ PC 2 and 3 ____________
# fig, ax = plt.subplots(figsize=(6, 6))
# fig.tight_layout()

# # Plot for Principal Components 1 and 2
# sns.scatterplot(x=X_pca[:n, 1], y=X_pca[:n, 2], ax=ax, label="Reduced Electron Densities", color="C0", s=30, edgecolor="black", linewidth=0.5)
# ax.set_title("Dataset after PCA feature reduction", fontsize=15)
# ax.set_xlabel("PC 2", fontsize=15)
# ax.set_ylabel("PC 3", fontsize=15)
# ax.grid()
# ax.legend()
# plt.show()







n_components = 3  # Change this to the desired number of clusters
n_samples = 6  # Number of closest samples to retrieve per cluster

cluster_plotter = GMMClusterPlotter(X_pca[n], n_components)
cluster_plotter.fit_predict()
# closest_indices, closest_labels = cluster_plotter.get_cluster_samples(n_samples)

# cluster_plotter.plot_clusters()
# cluster_plotter.plot_additional_kde_and_samples(closest_indices, closest_labels, X[n], r)


cluster_plotter.plot_interactive_contour(X[n], r)


# # Usage example
# # Assuming X_pca is your PCA-transformed dataset
# n_components = 3  # Change this to the desired number of clusters
# # X_pca = np.random.rand(100, 3)  # Example data; replace with your actual PCA data

# cluster_plotter = GMMClusterPlotter(X_pca[:n], n_components)
# cluster_plotter.fit_predict()
# cluster_plotter.plot_clusters()


# gmm = GaussianMixture(n_components=3, random_state=42)
# gmm.fit(X_pca)
# labels = gmm.predict(X_pca)




# n_closest = 6
# closest_indices, closest_labels = get_cluster_samples(X_pca[:n], labels[:n], n_closest)


# X_pca = X_pca[:n,:]
# labels = labels[:n]

# # Number of each class
# n_cluster0 = len(X_pca[labels==0])
# n_cluster1 = len(X_pca[labels==1])
# n_cluster2 = len(X_pca[labels==2])



# # =============================================================================
# #                   6 plots:  Scatter and KDE
# # =============================================================================

# #____________ PC 1 and 2 ____________
# fig, ax = plt.subplots(1, 2, figsize=(10, 5), sharey=True)
# fig.tight_layout()

# # Plot for Principal Components 1 and 2
# sns.scatterplot(x=X_pca[labels == 0, 0], y=X_pca[labels == 0, 1], ax=ax[0], label=f"label 0: {n_cluster0} samples", color="C0", s=30, edgecolor="black", linewidth=0.5, zorder=2)
# sns.scatterplot(x=X_pca[labels == 1, 0], y=X_pca[labels == 1, 1], ax=ax[0], label=f"label 1: {n_cluster1} samples", color="C1", s=30, edgecolor="black", linewidth=0.5)
# sns.scatterplot(x=X_pca[labels == 2, 0], y=X_pca[labels == 2, 1], ax=ax[0], label=f"label 2: {n_cluster2} samples", color="green", s=30, edgecolor="black", linewidth=0.5)
# # sns.scatterplot(x=X_pca[closest_indices, 0], y=X_pca[closest_indices, 1],ax=ax[0], label="", color="red", s=50, marker="o", edgecolor="black", zorder=3)
# ax[0].set_title("Scatter plot", fontsize=15)
# ax[0].set_xlabel("PC 1", fontsize=15)
# ax[0].set_ylabel("PC 2", fontsize=15)
# ax[0].grid()
# ax[0].legend()

# # KDE for Principal Components 1 and 2
# sns.kdeplot(x=X_pca[labels == 0, 0], y=X_pca[labels == 0, 1], fill=False, ax=ax[1], levels=10, color="C0", zorder=3)
# sns.kdeplot(x=X_pca[labels == 1, 0], y=X_pca[labels == 1, 1], fill=False, ax=ax[1], levels=10, color="C1", zorder=2)
# sns.kdeplot(x=X_pca[labels == 2, 0], y=X_pca[labels == 2, 1], fill=False, ax=ax[1], levels=10, color="green")
# ax[1].set_title("KDE plot", fontsize=15)
# ax[1].set_xlabel("PC 1", fontsize=15)
# ax[1].grid()
# kde_legend = [
#     Line2D([0], [0], color="C0", lw=2, label="Cluster 0"),
#     Line2D([0], [0], color="C1", lw=2, label="Cluster 1"),
#     Line2D([0], [0], color="green", lw=2, label="Cluster 2"),
# ]
# ax[1].legend(handles=kde_legend, fontsize=9.5, frameon=True)
# plt.show()



# #____________ PC 1 and 3 ____________
# fig, ax = plt.subplots(1, 2, figsize=(10, 5), sharey=True)
# fig.tight_layout()

# # Plot for Principal Components 1 and 3
# sns.scatterplot(x=X_pca[labels == 0, 0], y=X_pca[labels == 0, 2], ax=ax[0], label=f"label 0: {n_cluster0} samples", color="C0", s=30, edgecolor="black", linewidth=0.5)
# sns.scatterplot(x=X_pca[labels == 1, 0], y=X_pca[labels == 1, 2], ax=ax[0], label=f"label 1: {n_cluster1} samples", color="C1", s=30, edgecolor="black", linewidth=0.5)
# sns.scatterplot(x=X_pca[labels == 2, 0], y=X_pca[labels == 2, 2], ax=ax[0], label=f"label 2: {n_cluster2} samples", color="green", s=30, edgecolor="black", linewidth=0.5)
# # sns.scatterplot(x=X_pca[closest_indices, 0], y=X_pca[closest_indices, 1],ax=ax[0], label="", color="red", s=50, marker="o", edgecolor="black", zorder=3)
# ax[0].set_title("Scatter plot", fontsize=15)
# ax[0].set_xlabel("PC 1", fontsize=15)
# ax[0].set_ylabel("PC 3", fontsize=15)
# ax[0].grid()
# ax[0].legend()

# # KDE for Principal Components 1 and 3
# sns.kdeplot(x=X_pca[labels == 0, 0], y=X_pca[labels == 0, 2], fill=False, ax=ax[1], levels=10, color="C0", zorder=3)
# sns.kdeplot(x=X_pca[labels == 1, 0], y=X_pca[labels == 1, 2], fill=False, ax=ax[1], levels=10, color="C1", zorder=2)
# sns.kdeplot(x=X_pca[labels == 2, 0], y=X_pca[labels == 2, 2], fill=False, ax=ax[1], levels=10, color="green")
# ax[1].set_title("KDE plot", fontsize=15)
# ax[1].set_xlabel("PC 1", fontsize=15)
# ax[1].grid()
# kde_legend = [
#     Line2D([0], [0], color="C0", lw=2, label="Cluster 0"),
#     Line2D([0], [0], color="C1", lw=2, label="Cluster 1"),
#     Line2D([0], [0], color="green", lw=2, label="Cluster 2"),
# ]
# ax[1].legend(handles=kde_legend, loc="lower right", fontsize=9.5, frameon=True)
# plt.show()



# #____________ PC 2 and 3 ____________
# fig, ax = plt.subplots(1, 2, figsize=(10, 5), sharey=True)
# fig.tight_layout()

# # Plot for Principal Components 1 and 3
# sns.scatterplot(x=X_pca[labels == 0, 1], y=X_pca[labels == 0, 2], ax=ax[0], label=f"label 0: {n_cluster0} samples", color="C0", s=30, edgecolor="black", linewidth=0.5)
# sns.scatterplot(x=X_pca[labels == 1, 1], y=X_pca[labels == 1, 2], ax=ax[0], label=f"label 1: {n_cluster1} samples", color="C1", s=30, edgecolor="black", linewidth=0.5)
# sns.scatterplot(x=X_pca[labels == 2, 1], y=X_pca[labels == 2, 2], ax=ax[0], label=f"label 2: {n_cluster2} samples", color="green", s=30, edgecolor="black", linewidth=0.5)
# # sns.scatterplot(x=X_pca[closest_indices, 0], y=X_pca[closest_indices, 1],ax=ax[0], label="", color="red", s=50, marker="o", edgecolor="black", zorder=3)
# ax[0].set_title("Scatter plot", fontsize=15)
# ax[0].set_xlabel("PC 2", fontsize=15)
# ax[0].set_ylabel("PC 3", fontsize=15)
# ax[0].grid()
# ax[0].legend()

# # KDE for Principal Components 1 and 3
# sns.kdeplot(x=X_pca[labels == 0, 1], y=X_pca[labels == 0, 2], fill=False, ax=ax[1], levels=10, color="C0", zorder=3)
# sns.kdeplot(x=X_pca[labels == 1, 1], y=X_pca[labels == 1, 2], fill=False, ax=ax[1], levels=10, color="C1", zorder=2)
# sns.kdeplot(x=X_pca[labels == 2, 1], y=X_pca[labels == 2, 2], fill=False, ax=ax[1], levels=10, color="green")
# ax[1].set_title("KDE plot", fontsize=15)
# ax[1].set_xlabel("PC 2", fontsize=15)
# ax[1].grid()
# kde_legend = [
#     Line2D([0], [0], color="C0", lw=2, label="Cluster 0"),
#     Line2D([0], [0], color="C1", lw=2, label="Cluster 1"),
#     Line2D([0], [0], color="green", lw=2, label="Cluster 2"),
# ]
# ax[1].legend(handles=kde_legend, loc="lower left", fontsize=9.5, frameon=True)
# plt.show()







# # =============================================================================
# #                   3 plots:   KDE and Example samples
# # =============================================================================



# #____________ PC 1 and 2    PC 1 and 3 ____________



# # KDE for Principal Components 1 and 2
# fig, ax = plt.subplots(figsize=(6, 6))
# sns.kdeplot(x=X_pca[labels == 0, 0], y=X_pca[labels == 0, 1], fill=False, ax=ax, levels=10, color="C0", zorder=3)
# sns.kdeplot(x=X_pca[labels == 1, 0], y=X_pca[labels == 1, 1], fill=False, ax=ax, levels=10, color="C1", zorder=2)
# sns.kdeplot(x=X_pca[labels == 2, 0], y=X_pca[labels == 2, 1], fill=False, ax=ax, levels=10, color="green")
# sns.scatterplot(x=X_pca[closest_indices, 0], y=X_pca[closest_indices, 1], ax=ax, color="red", s=50, marker="o", edgecolor="black", zorder=4)
# ax.set_title("KDE", fontsize=15)
# ax.set_xlabel("PC 1", fontsize=15)
# ax.set_ylabel("PC 2", fontsize=15)
# ax.grid()
# kde_legend = [
#     Line2D([0], [0], color="C0", lw=2, label="Cluster 0"),
#     Line2D([0], [0], color="C1", lw=2, label="Cluster 1"),
#     Line2D([0], [0], color="green", lw=2, label="Cluster 2"),
#     Line2D([0], [0], marker="o", color="red", markersize=6, label="Cluster\nSamples", linestyle="None", markeredgecolor="black"),
# ]
# ax.legend(handles=kde_legend, loc="lower left", fontsize=9.5, frameon=True)
# plt.show()



# # KDE for Principal Components 1 and 3
# fig, ax = plt.subplots(figsize=(6, 6))
# sns.kdeplot(x=X_pca[labels == 0, 0], y=X_pca[labels == 0, 2], fill=False, ax=ax, levels=10, color="C0", zorder=3)
# sns.kdeplot(x=X_pca[labels == 1, 0], y=X_pca[labels == 1, 2], fill=False, ax=ax, levels=10, color="C1", zorder=2)
# sns.kdeplot(x=X_pca[labels == 2, 0], y=X_pca[labels == 2, 2], fill=False, ax=ax, levels=10, color="green")
# sns.scatterplot(x=X_pca[closest_indices, 0], y=X_pca[closest_indices, 2], ax=ax, color="red", s=50, marker="o", edgecolor="black", zorder=4)
# ax.set_title("KDE", fontsize=15)
# ax.set_xlabel("PC 1", fontsize=15)
# ax.set_ylabel("PC 3", fontsize=15)
# ax.grid()
# kde_legend = [
#     Line2D([0], [0], color="C0", lw=2, label="Cluster 0"),
#     Line2D([0], [0], color="C1", lw=2, label="Cluster 1"),
#     Line2D([0], [0], color="green", lw=2, label="Cluster 2"),
#     Line2D([0], [0], marker="o", color="red", markersize=6, label="Cluster\nSamples", linestyle="None", markeredgecolor="black"),
# ]
# ax.legend(handles=kde_legend, loc="lower left", fontsize=9.5, frameon=True)
# plt.show()






# fig, ax = plt.subplots(figsize=(6, 6))

# colors = {0: "C0", 1: "C1", 2: "green"}
# for i in range(0, len(closest_indices)):
#       ax.plot(np.log10(X[int(closest_indices[i]),:]), r, c=colors[int(closest_labels[i])])


# ax.set_title("Electron Density plot", fontsize=15)
# ax.set_xlabel(r"log$_{10}$($n_e$) [g/cm$^2$]", fontsize=13)
# ax.set_ylabel("Altitude [km]", fontsize=13)
# ax.grid()
# plot_legend = [
#     Line2D([0], [0], color="C0", lw=2, label="E+F region"),
#     Line2D([0], [0], color="C1", lw=2, label="F region"),
#     Line2D([0], [0], color="green", lw=2, label="E region"),
# ]
# ax.legend(handles=plot_legend, loc="upper left", fontsize=9.5, frameon=True)
# plt.show()


































# fig, ax = plt.subplots(1, 3, figsize=(14, 6))

# ax[0].set_title("Principal Components 1 and 2")
# ax[0].scatter(X_pca[labels==0, 0], X_pca[labels==0, 1], c="C0", s=30, edgecolors="black", linewidths=0.3, label="0", zorder=2)
# ax[0].scatter(X_pca[labels==1, 0], X_pca[labels==1, 1], c="C1", s=30, edgecolors="black", linewidths=0.3, label="1", zorder=1)
# ax[0].scatter(X_pca[labels==2, 0], X_pca[labels==2, 1], c="green", s=30, edgecolors="black", linewidths=0.3, label="2")
# ax[0].scatter(X_pca[closest_indices, 0], X_pca[closest_indices, 1], color="red", s=50, marker="o", edgecolors="black", label="Cluster\nSamples", zorder=2)
# ax[0].set_xlabel("PC 1")
# ax[0].set_ylabel("PC 2")
# ax[0].grid()
# ax[0].legend()



# ax[1].set_title("Principal Components 1 and 3")
# ax[1].scatter(X_pca[labels==0, 0], X_pca[labels==0, 2], c="C0", s=30, edgecolors="black", linewidths=0.3, label="0", zorder=2)
# ax[1].scatter(X_pca[labels==1, 0], X_pca[labels==1, 2], c="C1", s=30, edgecolors="black", linewidths=0.3, label="1", zorder=1)
# ax[1].scatter(X_pca[labels==2, 0], X_pca[labels==2, 2], c="green", s=30, edgecolors="black", linewidths=0.3, label="2", zorder=2)
# ax[1].scatter(X_pca[closest_indices, 0], X_pca[closest_indices, 2], color="red", s=50, marker="o", edgecolors="black", label="Cluster\nSamples", zorder=2)
# ax[1].set_xlabel("PC 1")
# ax[1].set_ylabel("PC 3")
# ax[1].grid()
# ax[1].legend()



# ax[2].set_title("Principal Components 2 and 3")
# ax[2].scatter(X_pca[labels==0, 1], X_pca[labels==0, 2], c="C0", s=30, edgecolors="black", linewidths=0.3, label="0", zorder=2)
# ax[2].scatter(X_pca[labels==1, 1], X_pca[labels==1, 2], c="C1", s=30, edgecolors="black", linewidths=0.3, label="1", zorder=1)
# ax[2].scatter(X_pca[labels==2, 1], X_pca[labels==2, 2], c="green", s=30, edgecolors="black", linewidths=0.3, label="2")
# ax[2].scatter(X_pca[closest_indices, 1], X_pca[closest_indices, 2], color="red", s=50, marker="o", edgecolors="black", label="Cluster\nSamples", zorder=2)
# ax[2].set_xlabel("PC 2")
# ax[2].set_ylabel("PC 3")
# ax[2].grid()
# ax[2].legend()
# plt.show()




# colors = {0: "C0", 1: "C1", 2: "green"}

# fig, ax = plt.subplots()

# for i in range(0, len(closest_indices)):
#     ax.plot(np.log10(X[int(closest_indices[i]),:]), r, c=colors[int(closest_labels[i])])

# ax.set_xlim(9, 12)
# ax.set_xlabel("Electron density")
# ax.set_ylabel("Altitude")
# plt.show()







# # Example: Print the number of samples selected for each cluster
# for cluster, indices in cluster_sample_indices.items():
#     print(f"Cluster {cluster}: Selected {len(indices)} samples")






































