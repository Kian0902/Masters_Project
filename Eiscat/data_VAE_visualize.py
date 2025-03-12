# -*- coding: utf-8 -*-
"""
Created on Wed Mar 12 13:47:37 2025

@author: Kian Sartipzadeh
"""



import torch
import numpy as np
import matplotlib.pyplot as plt


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def visualize_latent_space(model, dataset, batch_size=256):
    """
    Visualizes the latent space by encoding the dataset and plotting the 2D latent codes.

    Args:
        model: Trained VAE model.
        dataset: Dataset to encode (e.g. your full StoreDataset).
        batch_size: Batch size for DataLoader (default: 256).
    """
    
    # data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    latent_codes = []
    
    data_loader = dataset
    
    model.eval()
    with torch.no_grad():
        for data in data_loader:
            data = data.to(device)
            
            mu, _ = model.encode(data)
            latent_codes.append(mu.cpu().numpy())
    
    # Concatenate all batches into one array
    latent_codes = np.concatenate(latent_codes, axis=0)

    
    plt.figure(figsize=(8, 6))
    plt.scatter(latent_codes[:, 0], latent_codes[:, 1], alpha=0.6, edgecolors='k')
    plt.xlabel('Latent Dim 1')
    plt.ylabel('Latent Dim 2')
    plt.title('Latent Space Visualization')
    plt.grid(False)
    plt.show()




# # Example usage:
# # Visualize the latent space using the entire dataset (A is your StoreDataset)
# visualize_latent_space(model, val_loader)



