# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 12:26:34 2020

@author: Thanh Tung Khuat

This script is to execute other machine learning algorithms on real testing sets

Algorithms:
    - Decision trees
    - Gaussian Naive Bayes
    - Linear Discriminant Analysis
    - Random Forest
    - AdaBoost
    - XGBoost
    - LightGBM
    - Extra-trees classifier
    - Support vector machines
    - K-nearest neighbors
"""
import sys, os
from os.path import dirname
root_path = dirname(dirname(os.getcwd()))
sys.path.insert(0, root_path) # insert root directory to environmental variables

import numpy as np
import pandas as pd
import math
import time

from GFMM.onlinegfmm_opt import OnlineGFMM

from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
# We can use ParameterSampler to run random search for RandomHyperboxes or other GFMM models
from sklearn.model_selection import ParameterSampler
from sklearn.metrics import confusion_matrix

if __name__ == '__main__':
    
    path = "/data/tkhuat/gfmm/Dataset/Electronic_data/"
    training_file = "New_TrainingDataset_30features.csv"
    
    testing_file_1 = "RealProcessDataset_F30.csv"
    testing_file_2 = "Simulation1Dataset_F30.csv"
    testing_file_3 = "Simulation2Dataset_F30.csv"
    
    path_save = "/data/tkhuat/gfmm/Experiment/New_Electronic_data/other_ml_realdataset/"
    if not os.path.exists(path_save):
        os.makedirs(path_save)
                
    file_acc_realprocess = "onlngfmm_realprocess_f30_predicted_results.csv"
    file_acc_sim1 = "onlngfmm_simulation1_f30_predicted_results.csv"
    file_acc_sim2 = "onlngfmm_simulation2_f30_predicted_results.csv"
    
    
    df_train = pd.read_csv(path + training_file, header=None)
    df_test_realprocess = pd.read_csv(path + testing_file_1, header=None)
    df_test_sim1 = pd.read_csv(path + testing_file_2, header=None)
    df_test_sim2 = pd.read_csv(path + testing_file_3, header=None)
    
    X_tr = df_train.iloc[:, :-1].values
    y_tr = df_train.iloc[:, -1].values
    
    X_test_realprocess = df_test_realprocess.values 
    X_test_sim1 = df_test_sim1.values
    X_test_sim2 = df_test_sim2.values
    
    # For OnlnGFMM
    print("Onln-GFMM")
    start_time = time.perf_counter()
    
    onln_gfmm_clf = OnlineGFMM(1, 0.1, 0.1, False, 'min', False)
    
    onln_gfmm_clf.fit(X_tr, X_tr, y_tr)
    pred_onlngfmm_realprocess = onln_gfmm_clf.predict(X_test_realprocess, X_test_realprocess, None, True)
    pred_onlngfmm_sim1 = onln_gfmm_clf.predict(X_test_sim1, X_test_sim1, None, True)
    pred_onlngfmm_sim2 = onln_gfmm_clf.predict(X_test_sim2, X_test_sim2, None, True)
    
    pred_final_realprocess_result = pred_onlngfmm_realprocess.reshape(-1, 1)
    pred_final_sim1_result = pred_onlngfmm_sim1.reshape(-1, 1)
    pred_final_sim2_result = pred_onlngfmm_sim2.reshape(-1, 1)
    
    open(path_save + file_acc_realprocess, 'w').close() # make existing file empty        
    with open(path_save + file_acc_realprocess, 'a') as f_handle:
        f_handle.writelines('Onln-GFMM 0.1\n')
        np.savetxt(f_handle, pred_final_realprocess_result, fmt='%s', delimiter=', ')
        
    open(path_save + file_acc_sim1, 'w').close() # make existing file empty        
    with open(path_save + file_acc_sim1, 'a') as f_handle:
        f_handle.writelines('Onln-GFMM 0.1\n')
        np.savetxt(f_handle, pred_final_sim1_result, fmt='%s', delimiter=', ')
    
    open(path_save + file_acc_sim2, 'w').close() # make existing file empty        
    with open(path_save + file_acc_sim2, 'a') as f_handle:
        f_handle.writelines('Onln-GFMM 0.1\n')
        np.savetxt(f_handle, pred_final_sim2_result, fmt='%s', delimiter=', ')
        
    

