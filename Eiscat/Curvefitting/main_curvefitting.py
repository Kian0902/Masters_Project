# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 14:53:48 2024

@author: Kian Sartipzadeh
"""



import os
import pickle
from curvefitting_models import CurvefittingChapman
from curvefitting_evaluation import CurvefittingEvaluation


# Importing processed dataset
custom_file_path = "X_avg"
with open(custom_file_path, 'rb') as f:
    dataset = pickle.load(f)




key_choise = list(dataset.keys())[:3]
X = {key: dataset[key] for key in key_choise}




# # # m = 'scipy'
# m = 'lmfit'
# # # m = 'NN'

# A = CurvefittingChapman(X)
# A.batch_detection(model_name=m, H_initial=[5, 10, 25, 40], save_plot=False)
# A.save_curvefits(custom_file_path + "_" + m + "_curvefits")

# x = A.return_curvefits()


def import_file(file_name):
    with open(file_name, 'rb') as f:
        dataset = pickle.load(f)
    
    return dataset


# # Import files
file_org = "X_avg"
file_fit = "X_avg_lmfit_curvefits"

# Get datasets
X_org = import_file(file_org)
X_fit = import_file(file_fit)

key_choise = list(X_fit.keys())[:]


x_org = {key: X_org[key] for key in key_choise}
x_fit = {key: X_fit[key] for key in key_choise}



E = CurvefittingEvaluation(x_org, x_fit)
E.batch_detection(eval_method="Normalized Residuals", show_plot=True, save_plot=False)
# E.residual_norm(x_org, x_fit, show_plot=True, save_plot=True)
# E.chi_square(x_org, x_fit, show_plot=True)










