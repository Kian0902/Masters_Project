# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 15:46:29 2024

@author: Kian Sartipzadeh
"""

import torch
import numpy as np
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_model(model, test_loader, loss_function):
    model.eval()
    total_test_loss = 0.0
    predicted_outputs = []
    true_outputs = []

    with torch.no_grad():
        for test_inputs, test_targets in test_loader:
            test_inputs, test_targets = test_inputs.to(device), test_targets.to(device)
            
            z = torch.linspace(90, 400, 27).to(device)
            test_outputs = model(test_inputs, z)
            
            outputs_np = test_outputs.cpu().numpy()
            targets_np = test_targets.cpu().numpy()
            
            predicted_outputs.append(outputs_np)
            true_outputs.append(targets_np)
            
            test_loss = loss_function(test_outputs, test_targets)
            total_test_loss += test_loss.item()

    avg_test_loss = total_test_loss / len(test_loader)
    predicted_outputs = np.concatenate(predicted_outputs, axis=0)
    true_outputs = np.concatenate(true_outputs, axis=0)

    predicted_outputs_flat = predicted_outputs.flatten()
    true_outputs_flat = true_outputs.flatten()

    r2 = r2_score(true_outputs_flat, predicted_outputs_flat)

    tolerance = 0.02
    relative_error = np.abs(predicted_outputs_flat - true_outputs_flat) / np.abs(true_outputs_flat)
    accuracy = np.mean(relative_error < tolerance) * 100

    return avg_test_loss, r2, accuracy, predicted_outputs, true_outputs


def plot_results(predicted_outputs, true_outputs, num_plots=None):
    z = np.linspace(90, 400, 27)
    print(num_plots)
    
    if not num_plots:
        num_plots=len(predicted_outputs)
    
    plt.figure(figsize=(10, 6))
    for i in range(num_plots):
        plt.plot(predicted_outputs[i], z, label='Predicted', color="C1")
        plt.plot(true_outputs[i], z, label='True', color="C0")
        plt.xlabel('Log Electron Density')
        plt.ylabel('Altitude (km)')
        plt.legend()
        plt.show()
