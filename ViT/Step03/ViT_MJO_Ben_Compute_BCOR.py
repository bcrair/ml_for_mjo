#!/usr/bin/env python
# coding: utf-8

# ### CNN for MJO Compute Bivariate Correlation - V1
# ### Ben Crair
# ### March 21st


#imports
import matplotlib.pyplot as plt
import numpy as np
import torch
from netCDF4 import Dataset
import csv
import os

leadTm = 15
directory = "/home/bec32/project/ML_for_MJO/ViT/Ensemble_Results/Result_03/"
num_ensembles = 1
num_rounds = 3
num_prediction_points = 1528 + 2036 * 2

def torch_bcorr(x, y):
    pred_rmm1 = x[:,0]
    pred_rmm2 = x[:,1]
    target_rmm1 = y[:,0]
    target_rmm2 = y[:,1]
    
    return torch.sum(target_rmm1 * pred_rmm1 + target_rmm2 * pred_rmm2) / (torch.sqrt(torch.sum(target_rmm1 ** 2 + target_rmm2**2)) * torch.sqrt(torch.sum(pred_rmm1 ** 2 + pred_rmm2**2)))

sum_train_preds = np.zeros((9676, 2)).astype(np.float32)
sum_train_targets = np.zeros((9676, 2)).astype(np.float32)
sum_test_preds = np.zeros((5600, 2)).astype(np.float32)
sum_test_targets = np.zeros((5600, 2)).astype(np.float32)


for i in range(num_ensembles):
    train_preds = np.zeros((0, 2)).astype(np.float32)
    train_targets = np.zeros((0, 2)).astype(np.float32)
    test_preds = np.zeros((0, 2)).astype(np.float32)
    test_targets = np.zeros((0, 2)).astype(np.float32)
    for j in range(num_rounds):
        in1 = Dataset(directory + "ViTMJO_Ben_OBS_leadTm"+str(leadTm)+"_ensm"+str(i+1)+"_rund"+str(j+1)+"_Prdct_Train.nc")
        in2 = Dataset(directory + "ViTMJO_Ben_OBS_leadTm"+str(leadTm)+"_ensm"+str(i+1)+"_rund"+str(j+1)+"_Prdct_Test.nc")
    
        temp_train_pred = np.array(in1.variables["mjoTrainPred"] [:])
        temp_train_target = np.array(in1.variables["mjoTrainTagt"] [:])
        temp_test_pred = np.array(in2.variables["mjoTestPred"] [:])
        temp_test_target = np.array(in2.variables["mjoTestTagt"] [:])
        train_preds = np.vstack((train_preds, temp_train_pred))
        train_targets = np.vstack((train_targets, temp_train_target))
        test_preds = np.vstack((test_preds, temp_test_pred))
        test_targets = np.vstack((test_targets, temp_test_target))

    sum_train_preds = sum_train_preds + train_preds
    sum_train_targets = sum_train_targets + train_targets
    sum_test_preds = sum_test_preds + test_preds
    sum_test_targets = sum_test_targets + test_targets

avg_train_preds = sum_train_preds / num_ensembles
avg_train_targets = sum_train_targets / num_ensembles
avg_test_preds = sum_test_preds / num_ensembles
avg_test_targets = sum_test_targets / num_ensembles

train_corr = torch_bcorr(torch.tensor(avg_train_preds), torch.tensor(avg_train_targets))
print(f"Train Correlation: {train_corr}")

test_corr = torch_bcorr(torch.tensor(avg_test_preds), torch.tensor(avg_test_targets))
print(f"Test Correlation: {test_corr}")

filename = directory + "BCOR_by_leadTm.csv"
new_row = [leadTm, train_corr.item(), test_corr.item()]
file_exists = os.path.isfile(filename)

with open(filename, 'a', newline='') as csvfile:
    writer = csv.writer(csvfile)

    if not file_exists:
        writer.writerow(['leadTm', 'train_corr', 'test_corr'])

    writer.writerow(new_row)
