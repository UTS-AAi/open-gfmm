# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 23:42:23 2020

@author: Thanh Tung Khuat

This class contains scripts of functions to test Polynomial kernerl SVM with parameter tuning
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

from sklearn.svm import SVC

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

scv = StratifiedKFold(n_splits=5, random_state=0, shuffle=True)


def find_opt_params_poly_svm(Xtr, ytr):
    svm_tuned_parameters = {'kernel': ['poly'], 'gamma': [2**-15, 2**-13, 2**-11, 2**-9, 2**-7, 2**-5, 2**-3, 2**-1, 2**1, 2**3], 'C': [2**-5, 2**-3, 2**-1, 2**1, 2**3, 2**5, 2**7], 'degree':range(1, 11), 'coef0':[0.001, 0.01, 0.1, 1, 10, 100]}
    
    svc_clf = SVC(random_state=0)
    
    svc_clf_random_search = RandomizedSearchCV(svc_clf, svm_tuned_parameters, cv=scv,
                                  scoring="accuracy", n_iter=100,
                                  return_train_score=True, random_state=0,
                                  verbose=True,
                                  n_jobs=1)
    
    svc_clf_random_search.fit(Xtr, ytr)
    
    return (svc_clf_random_search.best_params_, svc_clf_random_search.best_score_)

if __name__ == '__main__':
    path = "/data/tkhuat/gfmm/Dataset/Electronic_data/"
    training_file = "TrainingDataset_30features.csv"
    testing_file = "TestingDataset_30features.csv"
    
    path_save = "/data/tkhuat/gfmm/Experiment/Electronic_data/poly_svm/"
    if not os.path.exists(path_save):
        os.makedirs(path_save)
        
    file_acc_name = "acc_opt_poly_svm.csv"
    file_prob_name = "wrong_prob_opt_poly_svm.csv"
    
    df_train = pd.read_csv(path + training_file, header=None)
    df_test = pd.read_csv(path + testing_file, header=None)
    
    X_tr = df_train.iloc[:, :-1].values
    y_tr = df_train.iloc[:, -1].values
    
    X_test = df_test.iloc[:, :-1].values
    y_test = df_test.iloc[:, -1].values
    
    # Save accuracy
    open(path_save + file_acc_name, 'w').close() # make existing file empty        
    with open(path_save + file_acc_name, 'a') as f_handle:
        f_handle.writelines('testing accuracy, validation accuracy, training time, optimisation time, best params \n')
    
    start_time = time.perf_counter()
    best_params_poly, best_val_score = find_opt_params_poly_svm(X_tr, y_tr)
    opt_time = time.perf_counter() - start_time
    
    classifier = SVC(random_state=0, probability=True)
    classifier.set_params(**best_params_poly)
    start_time = time.perf_counter()
    classifier.fit(X_tr, y_tr)
    training_time = time.perf_counter() - start_time
        
    predicted_class = classifier.predict(X_test)
        
    acc_testing_poly = accuracy_score(y_test, predicted_class)
    
    id_nonexact = np.nonzero(y_test != predicted_class)[0]
    extracted_prob = classifier.predict_proba(X_test[id_nonexact])
    id_real = id_nonexact + 1
    saved_extracted_prob = np.hstack((id_real.reshape(-1, 1), extracted_prob))
    
    open(path_save + file_prob_name, 'w').close() # make existing file empty        
    with open(path_save + file_prob_name, 'a') as f_handle:
        f_handle.writelines('testing sample index, prob prediction of class 1, prob prediction of class 2\n')
        np.savetxt(f_handle, saved_extracted_prob, fmt='%s', delimiter=', ')
        
    # Save accuracy
    with open(path_save + file_acc_name, 'a') as f_handle:
        f_handle.writelines('%.6f, %.6f, %.3f, %.3f, %s \n'%(acc_testing_poly, best_val_score, training_time, opt_time, best_params_poly))
