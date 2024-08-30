# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 11:04:56 2024

@author: Kian Sartipzadeh
"""

import numpy as np
from scipy.stats import zscore
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

class EISCATOutlierDetection:
    """
    Class for detecting outliers present in EISCAT data.
    """
    def __init__(self, dataset: dict):
        """
        Initialize with dataset containing EISCAT dat from one day.
        
        Attributes (type) | DESCRIPTION
        ------------------------------------------------
        dataset (dict)    | Dictionary containing the filtered EISCAT data
        """
        self.dataset = dataset
        self.detection_methods = {'z-score': self.z_score_method,
                                  'IQR': self.iqr_method}
    
    
    
    
    
    # Z-score
    def z_score_method(self, data: np.array, threshold: int=3):
        """
        Detect outliers using the Z-score method.
        
        Input (type)        | DESCRIPTION
        ------------------------------------------------
        data  (np.ndarray)  | Data from one key to be analyzed.
        threshold (int)     | Z-score threshold for identifying outliers (default is 3).
        
        
        Return (type)                    | DESCRIPTION
        ------------------------------------------------
        detected_outliers  (np.ndarray)  | Boolean array where True indicates an outlier.
        """
        z_score = zscore(data, axis=0)  # get z-scores
        detected_outliers = np.abs(z_score) > threshold
        return detected_outliers
    
    
    
    # Inter-Quantile Range
    def iqr_method(self, data: np.array, lower_percent: int=5, upper_percent: int=95):
        """
        Detect outliers using the Interquartile Range (IQR) method.
    
        Input (type)             | DESCRIPTION
        ------------------------------------------------
        data (np.ndarray)        | Data from one key to be analyzed.
        lower_percent (int)      | Percentile to determine the lower bound (default is 10).
        upper_percent (int)      | Percentile to determine the upper bound (default is 90).
    
        Return (type)                    | DESCRIPTION
        ------------------------------------------------
        detected_outliers (np.ndarray)   | Boolean array where True indicates an outlier.
        """
        Q1 = np.percentile(data, lower_percent, axis=0)  # first quantile
        Q3 = np.percentile(data, upper_percent, axis=0)  # second quantile
        
        IQR = Q3 - Q1
        
        lower_fence = Q1 - 1.5*IQR
        upper_fence = Q3 + 1.5*IQR
        
        detect_outliers = (data < lower_fence) | (data > upper_fence)
        return detect_outliers
    
    

    
    def detect_outliers(self, method_name: str):
        """
        Detects outliers using the specified method.
        """
        
        if method_name not in self.detection_methods:
            raise ValueError(f"Method {method_name} not recognized.")
        
        r_param = self.dataset['r_param']
        outliers = self.detection_methods[method_name](r_param[:, :])
        
        # Find indices of minutes (rows) where any outlier is detected
        minutes_with_outliers = np.any(outliers, axis=0)
        outlier_indices = np.where(minutes_with_outliers)[0]
        
        print(outlier_indices)
        return outlier_indices
    
    
    
    
    
    def t_sne(self, ind_bad):
        
        r_h     = self.dataset['r_h'].flatten()
        r_param = self.dataset['r_param']
        
        
        model = TSNE(n_components=2, random_state=0, max_iter=1000, perplexity=15)
        tsne_data = model.fit_transform(r_param.T)
        
        print(tsne_data.shape)
        

        
        
        # Scatter plot
        plt.figure(figsize=(10, 8))
        plt.scatter(tsne_data[:, 0], tsne_data[:, 1], s=50, zorder=0)
        plt.scatter(tsne_data[ind_bad, 0], tsne_data[ind_bad, 1], s=50, color="red", zorder=1)
        # Labels and title
        plt.title('t-SNE Scatter Plot')
        plt.xlabel('t-SNE 1')
        plt.ylabel('t-SNE 2')
        
        # Show plot
        plt.show()
    

    def pca(self, ind_bad):
        
        r_h     = self.dataset['r_h'].flatten()
        r_param = self.dataset['r_param']
        
        
        plt.plot(r_param[:, ind_bad], r_h)
        plt.show()
        
        # print(r_param.shape)
        
        model = PCA(n_components=2)
        pca_data = model.fit_transform(r_param.T)
        
        print(pca_data.shape)
        

        
        
        # Scatter plot
        plt.figure(figsize=(10, 8))
        plt.scatter(pca_data[:, 0], pca_data[:, 1], s=50, zorder=0)
        plt.scatter(pca_data[ind_bad, 0], pca_data[ind_bad, 1], s=50, color="red", zorder=1)
        # Labels and title
        plt.title('PCA Scatter Plot')
        plt.xlabel('PCA 1')
        plt.ylabel('PCA 2')
        
        # Show plot
        plt.show()



