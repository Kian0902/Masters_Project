# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 12:57:30 2024

@author: Kian Sartipzadeh
"""



import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

sns.set(style="dark", context=None, palette=None)

def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def huber_loss(y_true, y_pred, delta=1.0):
    error = y_true - y_pred
    is_small_error = np.abs(error) <= delta
    squared_loss = 0.5 * error**2
    linear_loss = delta * (np.abs(error) - 0.5 * delta)
    return np.where(is_small_error, squared_loss, linear_loss).mean()

# Define the true value and a range of predicted values
y_true = 0
y_pred_range = np.linspace(-1.6, 1.6, 100)

# Calculate MSE, MAE, and Huber loss for each predicted value
mse_values = [mse(y_true, y_pred) for y_pred in y_pred_range]
mae_values = [mae(y_true, y_pred) for y_pred in y_pred_range]
huber_values = [huber_loss(y_true, y_pred) for y_pred in y_pred_range]

fig, ax = plt.subplots(figsize=(9, 6))

ax.plot(y_pred_range, mse_values, label='MSE')
ax.plot(y_pred_range, mae_values, label='MAE')
ax.plot(y_pred_range, huber_values, label='Huber')
ax.set_xlabel('Predicted Value', fontsize=15)
ax.set_ylabel('Loss', fontsize=15)
ax.set_title('Loss Functions Comparison', fontsize=15)
ax.grid()
ax.legend()
plt.show()
