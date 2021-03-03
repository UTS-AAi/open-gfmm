# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 10:47:01 2020

@author: Thanh Tung Khuat

This is a script to run Onln-GFMM on electronic data using grid-search and 5-fold cross-validation
"""
import sys, os
from os.path import dirname
root_path = dirname(dirname(os.getcwd()))
sys.path.insert(0, root_path) # insert root directory to environmental variables

import time
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from GFMM.onlinegfmm_opt import OnlineGFMM
from functionhelper.membershipcalc import extract_mem_for_classes
from sklearn.model_selection import StratifiedKFold

if __name__ == '__main__':
    path = "/data/tkhuat/gfmm/Dataset/Electronic_data/"
    training_file = "New_TrainingDataset_30features.csv"
    testing_file = "New_TestingDataset_30features.csv"
    
    path_save = "/data/tkhuat/gfmm/Experiment/New_Electronic_data/opt_onlngfmm"
    if not os.path.exists(path_save):
        os.makedirs(path_save)
        
    file_acc_name = "acc_opt_onln_gfmm.csv"
    file_mem_name = "wrong_mem_onln_gfmm_best_theta.csv"
    
    df_train = pd.read_csv(path + training_file, header=None)
    df_test = pd.read_csv(path + testing_file, header=None)
    
    X_tr = df_train.iloc[:, :-1].values
    y_tr = df_train.iloc[:, -1].values
    
    X_test = df_test.iloc[:, :-1].values
    y_test = df_test.iloc[:, -1].values
    
    best_theta = 0
    best_acc = 0
    num_split = 5
    skf = StratifiedKFold(n_splits=num_split, random_state=0, shuffle=True)
    start_time = time.perf_counter()
    for theta in [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]:
        tmp_acc = 0
        for train_index, val_index in skf.split(X_tr, y_tr):
            X_tr_fold, X_val_fold = X_tr[train_index], X_tr[val_index]
            y_tr_fold, y_val_fold = y_tr[train_index], y_tr[val_index]
            
            classifier = OnlineGFMM(1, theta, theta, False, 'min', False)
            classifier.fit(X_tr_fold, X_tr_fold, y_tr_fold)
                
            predicted_class = classifier.predict(X_val_fold, X_val_fold, y_val_fold, True)
            tmp_acc = tmp_acc + accuracy_score(y_val_fold, predicted_class)
        
        tmp_acc /= num_split
        if tmp_acc >= best_acc:
            best_acc = tmp_acc
            best_theta = theta
    
    opt_time = time.perf_counter() - start_time
                
    classifier = OnlineGFMM(1, best_theta, best_theta, False, 'min', False)
    classifier.fit(X_tr, X_tr, y_tr)
    opt_num_box = len(classifier.classId)
    training_time = classifier.elapsed_training_time
    
    y_pred = classifier.predict(X_test, X_test, y_test, True)
    opt_acc = accuracy_score(y_test, y_pred)
    
    id_nonexact = np.nonzero(y_test != y_pred)[0]
    extracted_mem = extract_mem_for_classes(classifier.V, classifier.W, classifier.classId, X_test[id_nonexact], X_test[id_nonexact])
    id_real = id_nonexact + 1
    saved_extracted_mem = np.hstack((id_real.reshape(-1, 1), extracted_mem))
    
    open(path_save + file_mem_name, 'w').close() # make existing file empty        
    with open(path_save + file_mem_name, 'a') as f_handle:
        f_handle.writelines('Best theta = %.2f \n'%best_theta)
        f_handle.writelines('testing sample index, membership of class 1, membership of class 2\n')
        np.savetxt(f_handle, saved_extracted_mem, fmt='%s', delimiter=', ')
        
    # Save accuracy
    open(path_save + file_acc_name, 'w').close() # make existing file empty        
    with open(path_save + file_acc_name, 'a') as f_handle:
        f_handle.writelines('best theta = %.2f, accuracy = %.6f, no of hyperboxes = %d, training time = %.2f, optimisation time = %.2f'%(best_theta, opt_acc, opt_num_box, training_time, opt_time))
       

