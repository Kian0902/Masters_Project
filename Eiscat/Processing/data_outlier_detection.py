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

        self.dataset_outliers = {}
    
    
    def batch_detection(self, method_name: str, save_plot=False):
        
        for key in list(self.dataset.keys()):
            
            self.dataset_outliers[key] = self.detect_outliers(self.dataset[key], method_name=method_name, save_plot=save_plot)
            
    
    
    # Z-score
    def z_score_method(self, data: np.ndarray, threshold: int=6):
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
        z_score = zscore(data, axis=1)  # get z-scores
        detected_outliers = np.abs(z_score) > threshold
        return detected_outliers
    
    
    
    # Inter-Quantile Range
    def iqr_method(self, data: np.ndarray, lower_percent: int=0.5, upper_percent: int=99.5):
        """
        Detect outliers using the Interquartile Range (IQR) method.
    
        Input (type)             | DESCRIPTION
        ------------------------------------------------
        data (np.ndarray)        | Data from one key to be analyzed.
        lower_percent (int)      | Percentile to determine the lower bound (default is 1).
        upper_percent (int)      | Percentile to determine the upper bound (default is 99).
    
        Return (type)                    | DESCRIPTION
        ------------------------------------------------
        detected_outliers (np.ndarray)   | Boolean array where True indicates an outlier.
        """
        Q1 = np.percentile(data, lower_percent, axis=1)  # first quantile
        Q3 = np.percentile(data, upper_percent, axis=1)  # second quantile
        
        
        IQR = Q3 - Q1
        
        lower_fence = Q1 - 1.5*IQR
        upper_fence = Q3 + 1.5*IQR
        
        detect_outliers = (data < lower_fence.reshape(-1, 1)) | (data > upper_fence.reshape(-1, 1))
        return detect_outliers
       

    
    
    def pca(self, data: np.ndarray, reduce_to_dim: int=2):
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
    
    
    
    
    
    
    def detect_outliers(self, data: dict, method_name: str, save_plot: bool=False):
        """
        Detects outliers in the dataset using the specified method.
        
        Input (type)    | DESCRIPTION
        ------------------------------------------------
        method_name (str) | The name of the outlier detection method to use.
        save_plot (bool)  | If True, generates and displays plots showing detected outliers. Default is False.
        
        
        Return (type)          | DESCRIPTION
        ------------------------------------------------
        bad_ind (np.ndarray)  | Indices of the detected outliers.
        

        Raises
        ------
        ValueError
            If the specified method is not recognized.
        """
        
        if method_name not in self.detection_methods:
            raise ValueError(f"Method {method_name} not recognized.")
        
        r_t = data['r_time'] 
        r_h = data['r_h']
        r_param = data['r_param']
        r_error = data['r_error']
        
        
        # Convert date and time to datetime objects
        r_time, date_of_day = self._to_datetime(data['r_time'])
        
        
        
        # Mask for filtered values. False outside and True inside interval
        mask = np.any((r_h >= 140) & (r_h <= 300), axis=1)
        
        
        # Perform PCA on Ne and errors
        pca_r_param = self.pca(r_param[mask,:])
        pca_r_error = self.pca(r_error[mask,:])
        
        
        outliers = self.detection_methods[method_name](pca_r_param)
        outliers_err = self.detection_methods[method_name](pca_r_error)
        
        # Find indices of minutes (rows) where any outlier is detected
        minutes_with_outliers = np.any(outliers, axis=0)
        minutes_with_outliers_err = np.any(outliers_err, axis=0)
        
        outlier_indices = np.where(minutes_with_outliers)[0]
        outlier_indices_err = np.where(minutes_with_outliers_err)[0]
        
        bad_ind = np.intersect1d(outlier_indices, outlier_indices_err)
        
        # bad_ind = outlier_indices
        
        # if len(bad_ind) == 0:
            # bad_ind = outlier_indices
        
        
        # Check if bad_ind is still empty
        if len(bad_ind) == 0:
            print(f"No outliers detected in day: {date_of_day}")
            return bad_ind
        
        # fig, ax = plt.subplots(1, 2)
        
        # ax[0].set_title('Electron Density')
        # ax[0].scatter(pca_r_param[0, :], pca_r_param[1, :], zorder=0)
        # ax[0].scatter(pca_r_param[0, outlier_indices], pca_r_param[1, outlier_indices], zorder=1, color="red")
        
        # ax[1].set_title('Error')
        # ax[1].scatter(pca_r_error[0, :], pca_r_error[1, :], zorder=0)
        # ax[1].scatter(pca_r_error[0, outlier_indices_err], pca_r_error[1, outlier_indices_err], zorder=1, color="red")
        # plt.show()
        
        
        
        # Option for the user to view plot of outliers
        if save_plot:
            num_plots = len(bad_ind) + 1  # One for the "Bad Samples" plot, plus one for each bad index
            
            fig, ax = plt.subplots(1, num_plots, figsize=(5 * num_plots, 5))
            
            # First plot: "Bad Samples"
            ax[0].set_title(f'Outliers in day: {date_of_day}')
            ax[0].scatter(pca_r_param[0, :], pca_r_param[1, :], zorder=0)
            ax[0].scatter(pca_r_param[0, bad_ind], pca_r_param[1, bad_ind], zorder=1, color="red")
            
            # Subsequent plots: one for each bad index
            for num_ax in range(1, num_plots):
                ax[num_ax].set_title(f'Sample Index: {bad_ind[num_ax-1]}')
                ax[num_ax].plot(r_param[:, bad_ind[num_ax-1]], r_h.flatten(), label=f'Index {bad_ind[num_ax-1]}')
                ax[num_ax].set_xscale('log')
                ax[num_ax].legend()
            
            plt.tight_layout()
            plt.show()
        
        
        
        return bad_ind
    
    
    
    def _to_datetime(self, time_array: np.ndarray):
        
        # Convert time arrays to datetime objects
        time_converted = np.array([datetime(year, month, day, hour, minute) 
                                   for year, month, day, hour, minute, second in time_array])
        
        # Date of 
        date_str = time_converted[0].strftime('%Y-%m-%d')
        
        return time_converted, date_str
        
        
        
        
    def return_outliers(self):
        """
        Returns self.dataset_outliers
        """
        return self.dataset_outliers
        
        
    




















