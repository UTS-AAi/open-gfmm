# -*- coding: utf-8 -*-
"""
Created on Sun Nov  8 22:41:28 2020

@author: Thanh Tung Khuat

This is a script to run experiments on electronic data using AGGLO-2 algorithm
"""
import sys, os
from os.path import dirname
root_path = dirname(dirname(os.getcwd()))
sys.path.insert(0, root_path) # insert root directory to environmental variables

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from GFMM.accelbatchgfmm_opt import AccelBatchGFMM
from functionhelper.membershipcalc import extract_mem_for_classes
from sklearn.metrics import confusion_matrix

if __name__ == '__main__':
    path = "/data/tkhuat/gfmm/Dataset/Electronic_data/"
    training_file = "New_TrainingDataset_30features_l1l2.csv"
    testing_file = "New_TestingDataset_30features_l1l2.csv"
    
    path_save = "/data/tkhuat/gfmm/Experiment/New_Electronic_data_l1l2/gfmm_f30/agglo2_f30/"
    if not os.path.exists(path_save):
        os.makedirs(path_save)
        
    file_acc_name = "acc_agglo2_gfmm.csv"
    file_mem_name = "wrong_mem_agglo2_gfmm_"
    file_right_mem_name = "right_mem_agglo2_gfmm_"
    
    theta_txt = ['0_1.csv', '0_15.csv', '0_2.csv', '0_25.csv', '0_3.csv', '0_35.csv', '0_4.csv', '0_45.csv', '0_5.csv', '0_55.csv', '0_6.csv', '0_65.csv', '0_7.csv', '0_75.csv', '0_8.csv', '0_85.csv', '0_9.csv', '0_95.csv']
    
    df_train = pd.read_csv(path + training_file, header=None)
    df_test = pd.read_csv(path + testing_file, header=None)
    
    X_tr = df_train.iloc[:, :-1].values
    y_tr = df_train.iloc[:, -1].values
    
    X_test = df_test.iloc[:, :-1].values
    y_test = df_test.iloc[:, -1].values
    
    open(path_save + file_acc_name, 'w').close() # make existing file empty
    with open(path_save + file_acc_name, 'a') as f_handle:
        f_handle.writelines('sim thres = 0.5, sim measure = short\n')
        f_handle.writelines('theta, accuracy, no of hyperboxes, training time, confusion matrix\n')
    
    for id_, theta in enumerate([0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]):
        print("Theta = ", theta)
        tmp_res = [theta]
        
        classifier = AccelBatchGFMM(gamma = 1, teta = theta, bthres = 0.5, simil = 'short', sing = 'min', isDraw = False, oper = 'min', isNorm = False)
        classifier.fit(X_tr, X_tr, y_tr)
        predicted_class = classifier.predict(X_test, X_test, y_test, True, 0)
        cf_mat = confusion_matrix(y_test, predicted_class)
    
        tmp_acc = accuracy_score(y_test, predicted_class)
        tmp_numbox = len(classifier.classId)
        tmp_trainingtime = classifier.elapsed_training_time
        
        id_nonexact = np.nonzero(y_test != predicted_class)[0]
        extracted_mem = extract_mem_for_classes(classifier.V, classifier.W, classifier.classId, X_test[id_nonexact], X_test[id_nonexact])
        id_real = id_nonexact + 1
        saved_extracted_mem = np.hstack((id_real.reshape(-1, 1), extracted_mem))
        
        id_exact = np.nonzero(y_test == predicted_class)[0]
        extracted_right_mem = extract_mem_for_classes(classifier.V, classifier.W, classifier.classId, X_test[id_exact], X_test[id_exact])
        id_real_right = id_exact + 1
        saved_extracted_right_mem = np.hstack((id_real_right.reshape(-1, 1), extracted_right_mem))
        
        open(path_save + file_mem_name + theta_txt[id_], 'w').close() # make existing file empty        
        with open(path_save + file_mem_name + theta_txt[id_], 'a') as f_handle:
            f_handle.writelines('testing sample index, membership of class 1, membership of class 2\n')
            np.savetxt(f_handle, saved_extracted_mem, fmt='%s', delimiter=', ')
            
        open(path_save + file_right_mem_name + theta_txt[id_], 'w').close() # make existing file empty        
        with open(path_save + file_right_mem_name + theta_txt[id_], 'a') as f_handle:
            f_handle.writelines('testing sample index, membership of class 1, membership of class 2\n')
            np.savetxt(f_handle, saved_extracted_right_mem, fmt='%s', delimiter=', ')
        
        with open(path_save + file_acc_name, 'a') as f_handle:
            f_handle.writelines('%.2f, %.6f, %d, %.3f, %s \n'%(theta, tmp_acc, tmp_numbox, tmp_trainingtime, str(cf_mat)))
            
