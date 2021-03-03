import sys, os
from os.path import dirname
root_path = dirname(dirname(os.getcwd()))
sys.path.insert(0, root_path) # insert root directory to environmental variables

import numpy as np
import pandas as pd
import math
import time

from GFMM.accelbatchgfmm_opt import AccelBatchGFMM

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
                
    file_acc_realprocess = "agglo2_realprocess_f30_predicted_results.csv"
    file_acc_sim1 = "agglo2_simulation1_f30_predicted_results.csv"
    file_acc_sim2 = "agglo2_simulation2_f30_predicted_results.csv"
    
    
    df_train = pd.read_csv(path + training_file, header=None)
    df_test_realprocess = pd.read_csv(path + testing_file_1, header=None)
    df_test_sim1 = pd.read_csv(path + testing_file_2, header=None)
    df_test_sim2 = pd.read_csv(path + testing_file_3, header=None)
    
    X_tr = df_train.iloc[:, :-1].values
    y_tr = df_train.iloc[:, -1].values
    
    X_test_realprocess = df_test_realprocess.values 
    X_test_sim1 = df_test_sim1.values
    X_test_sim2 = df_test_sim2.values
    
    # For AGGLO-2
    print("AGGLO-2")
    start_time = time.perf_counter()
    theta = 0.6
    agglo2_clf = AccelBatchGFMM(gamma = 1, teta = theta, bthres = 0.5, simil = 'short', sing = 'min', isDraw = False, oper = 'min', isNorm = False)
    
    agglo2_clf.fit(X_tr, X_tr, y_tr)
    pred_onlngfmm_realprocess = agglo2_clf.predict(X_test_realprocess, X_test_realprocess, None, True, 0)
    pred_onlngfmm_sim1 = agglo2_clf.predict(X_test_sim1, X_test_sim1, None, True, 0)
    pred_onlngfmm_sim2 = agglo2_clf.predict(X_test_sim2, X_test_sim2, None, True, 0)
    
    pred_final_realprocess_result = pred_onlngfmm_realprocess.reshape(-1, 1)
    pred_final_sim1_result = pred_onlngfmm_sim1.reshape(-1, 1)
    pred_final_sim2_result = pred_onlngfmm_sim2.reshape(-1, 1)
    
    open(path_save + file_acc_realprocess, 'w').close() # make existing file empty        
    with open(path_save + file_acc_realprocess, 'a') as f_handle:
        f_handle.writelines('AGGLO-2\n')
        np.savetxt(f_handle, pred_final_realprocess_result, fmt='%s', delimiter=', ')
        
    open(path_save + file_acc_sim1, 'w').close() # make existing file empty        
    with open(path_save + file_acc_sim1, 'a') as f_handle:
        f_handle.writelines('AGGLO-2\n')
        np.savetxt(f_handle, pred_final_sim1_result, fmt='%s', delimiter=', ')
    
    open(path_save + file_acc_sim2, 'w').close() # make existing file empty        
    with open(path_save + file_acc_sim2, 'a') as f_handle:
        f_handle.writelines('AGGLO-2\n')
        np.savetxt(f_handle, pred_final_sim2_result, fmt='%s', delimiter=', ')
        
    

