# -*- coding: utf-8 -*-
"""
Created on Sat Sep 14 16:07:10 2024

@author: Kian Sartipzadeh
"""


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt




class FeatureAnalysis:
    
    
    def __init__(self, dataset, feature_names):
        
        self.dataset = dataset
        self.feature_names = feature_names
    
    
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
                    xticklabels=self.feature_names, yticklabels=self.feature_names)
        plt.title('Correlation Matrix')
        plt.show()



    def plot_correlogram(self):
        print("h")
        
        # Convert the dataset to a DataFrame for use with Seaborn
        df = pd.DataFrame(self.dataset, columns=self.feature_names)
        
        
        print("d")
        
        sns.pairplot(df, kind='scatter', diag_kind='kde')
        plt.suptitle('Correlogram')
        plt.show()




















