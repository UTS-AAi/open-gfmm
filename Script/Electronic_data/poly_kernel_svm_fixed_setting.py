# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 21:37:38 2020

@author: Thanh Tung Khuat

This class contains scripts of functions to test Polynomial kernerl SVM with different values of degree 
"""
import sys, os
from os.path import dirname
root_path = dirname(dirname(os.getcwd()))
sys.path.insert(0, root_path) # insert root directory to environmental variables

import numpy as np
import pandas as pd
import math
import time

from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix

from sklearn.svm import SVC

if __name__ == '__main__':
    path = "/data/tkhuat/gfmm/Dataset/Electronic_data/"
    training_file = "TrainingDataset_10features.csv"
    testing_file = "TestingDataset_10features.csv"
    
    path_save = "/data/tkhuat/gfmm/Experiment/Electronic_data/poly_svm_10/"
    if not os.path.exists(path_save):
        os.makedirs(path_save)
        
    file_acc_name = "acc_svm.csv"
    file_prob_name = "wrong_prob_svm_"
    theta_txt = ['d1.csv', 'd2.csv', 'd3.csv', 'd4.csv', 'd5.csv', 'd6.csv', 'd7.csv', 'd8.csv', 'd9.csv', 'd10.csv']
    
    df_train = pd.read_csv(path + training_file, header=None)
    df_test = pd.read_csv(path + testing_file, header=None)
    
    X_tr = df_train.iloc[:, :-1].values
    y_tr = df_train.iloc[:, -1].values
    
    X_test = df_test.iloc[:, :-1].values
    y_test = df_test.iloc[:, -1].values
    
    acc_results = []
    
    # Save accuracy
    open(path_save + file_acc_name, 'w').close() # make existing file empty        
    with open(path_save + file_acc_name, 'a') as f_handle:
        f_handle.writelines('degree, accuracy, training time, confusion matrix\n')
    
    for id_, deg in enumerate([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]):
        print("degree = ", deg)
        
        classifier = SVC(C=2, kernel='poly', degree=deg, gamma='auto', coef0=1, probability=True)
        start_time = time.perf_counter()
        classifier.fit(X_tr, y_tr)
        tmp_training_time = time.perf_counter() - start_time
            
        predicted_class = classifier.predict(X_test)
            
        tmp_acc = accuracy_score(y_test, predicted_class)
        cf_mat = confusion_matrix(y_test, predicted_class)
        
        id_nonexact = np.nonzero(y_test != predicted_class)[0]
        extracted_prob = classifier.predict_proba(X_test[id_nonexact])
        id_real = id_nonexact + 1
        saved_extracted_prob = np.hstack((id_real.reshape(-1, 1), extracted_prob))
        
        open(path_save + file_prob_name + theta_txt[id_], 'w').close() # make existing file empty        
        with open(path_save + file_prob_name + theta_txt[id_], 'a') as f_handle:
            f_handle.writelines('testing sample index, prob prediction of class 1, prob prediction of class 2\n')
            np.savetxt(f_handle, saved_extracted_prob, fmt='%s', delimiter=', ')
            
        # Save accuracy
        with open(path_save + file_acc_name, 'a') as f_handle:
            f_handle.writelines('%d, %.6f, %.3f, %s \n'%(deg, tmp_acc, tmp_training_time, str(cf_mat)))
