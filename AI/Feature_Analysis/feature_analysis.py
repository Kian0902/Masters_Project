# -*- coding: utf-8 -*-
"""
Created on Sat Sep 14 16:07:10 2024

@author: Kian Sartipzadeh
"""


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA


class FeatureAnalysis:
    
    
    def __init__(self, dataset, feature_names):
        
        self.dataset = dataset
        self.feature_names = feature_names
    
    
    
    def exclude_feature(self, drop_features):
        
        # Ensure features_to_drop is a list
        if isinstance(drop_features, str):
            drop_features = [drop_features]
        
        drop_indices = [self.feature_names.index(f) for f in drop_features if f in self.feature_names]
        
        self.dataset = np.delete(self.dataset, drop_indices, axis=1)
        self.feature_names = [f for f in self.feature_names if f not in drop_features]
    
    
    
    
    def merge_feature(self, features_to_reduce, n_components=1, pca_component_name=["PCA_component"]):
        
        reduce_indices = [self.feature_names.index(f) for f in features_to_reduce if f in self.feature_names]
        features_data = self.dataset[:, reduce_indices]
        
        
        # Perform PCA
        pca = PCA(n_components=n_components)
        reduced_data = pca.fit_transform(features_data)
        
        self.dataset = np.delete(self.dataset, reduce_indices, axis=1)
        self.feature_names = [f for i, f in enumerate(self.feature_names) if i not in reduce_indices]
        
        # Add the reduced features to the dataset
        for i in range(n_components):
            component_name = pca_component_name[i]
            self.dataset = np.column_stack((self.dataset, reduced_data[:, i]))
            self.feature_names.append(component_name)
    
    
    
    
    
    
    def correlation_matrix(self, plot_correlation_matrix=False, plot_correlogram=False):
        corr_matrix = np.corrcoef(self.dataset, rowvar=False)
        
        if plot_correlation_matrix:
            self.plot_correlation_matrix()
        
        if plot_correlogram:
            self.plot_correlogram()
        
        return corr_matrix
    
    

    def plot_correlation_matrix(self):
        
        # Calculating correlation matrix
        corr_matrix = self.correlation_matrix()
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True, linewidths=.5, 
                    xticklabels=self.feature_names, yticklabels=self.feature_names, vmin=-1, vmax=1)
        plt.title('Correlation Matrix')
        plt.show()
    
    
    
    def plot_correlogram(self):

        # Convert the dataset to a DataFrame for use with Seaborn
        df = pd.DataFrame(self.dataset, columns=self.feature_names)
        
        sns.pairplot(df, kind="scatter", diag_kind="kde")
        plt.suptitle('Correlogram')
        plt.show()








    # def analyze_pca(self):
    #     pca = PCA()
    #     pca.fit(self.dataset)
    
    #     # Explained variance (eigenvalues) for each component
    #     eigenvalues = pca.explained_variance_
    #     explained_variance_ratio = pca.explained_variance_ratio_
        
    #     # Plotting the eigenvalues
    #     plt.figure(figsize=(8, 5))
    #     plt.plot(range(1, len(eigenvalues) + 1), eigenvalues, marker='o', linestyle='--')
    #     plt.axhline(y=1, color='r', linestyle='-')  # Kaiser Criterion threshold
    #     plt.title('Eigenvalues of Principal Components')
    #     plt.xlabel('Principal Component')
    #     plt.ylabel('Eigenvalue')
    #     plt.grid(True)
    #     plt.show()
    
    #     # Plotting the cumulative explained variance
    #     cumulative_explained_variance = pca.explained_variance_ratio_.cumsum()
    #     plt.plot(range(1, len(cumulative_explained_variance) + 1), cumulative_explained_variance, marker='o', linestyle='--')
    #     plt.axhline(y=0.9, color='r', linestyle='-')
    #     plt.title('Cumulative Explained Variance')
    #     plt.xlabel('Number of Components')
    #     plt.ylabel('Cumulative Explained Variance')
    #     plt.grid(True)
    #     plt.show()
        
    #     # Plot the feature loadings for the first principal component
    #     self.plot_feature_loadings(pca)
    
    # def plot_feature_loadings(self, pca):
    #     # Loadings (components)
    #     loadings = pd.DataFrame(pca.components_.T, columns=[f'PC{i+1}' for i in range(len(pca.components_))], index=self.feature_names)
        
    #     # Plotting the loadings for the first principal component
    #     plt.figure(figsize=(10, 7))
    #     plt.bar(loadings.index, loadings['PC1'], color='c')
    #     plt.title('Feature Loadings for the First Principal Component')
    #     plt.xlabel('Features')
    #     plt.ylabel('Loading')
    #     plt.xticks(rotation=90)
    #     plt.grid(True)
    #     plt.show()

        # If you want to plot loadings for more components, you can modify or extend this method
    

# class FeatureEngineering:















