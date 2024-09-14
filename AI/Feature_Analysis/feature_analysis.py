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



















