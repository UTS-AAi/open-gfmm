# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 12:26:34 2020

@author: Thanh Tung Khuat

This script is to execute other machine learning algorithms using 5-fold cross-validation and random or grid search

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

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
# We can use ParameterSampler to run random search for RandomHyperboxes or other GFMM models
from sklearn.model_selection import ParameterSampler
from sklearn.metrics import confusion_matrix

scv = StratifiedKFold(n_splits=5, random_state=0, shuffle=True)

def find_opt_params_xgboost(Xtr, ytr):
    param_grid_xgboost = {'max_depth': range(4, 21),
                      'subsample': [0.3, 0.4, 0.5, 0.6, 0.7],
                      'colsample_bytree': [0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
                      'eta': [0.025, 0.05, 0.1, 0.2, 0.3],
                      'gamma': [0, 0.1, 0.2, 0.3, 0.4, 1, 1.5, 2],
                      'n_estimators': [30, 50, 70, 100, 150, 200]}
    
    xgboost = XGBClassifier(random_state=0)
    xgboost_random_search = RandomizedSearchCV(xgboost, param_grid_xgboost, cv=scv,
                                  scoring="accuracy", n_iter=100,
                                  return_train_score=True, random_state=0,
                                  verbose=True,
                                  n_jobs=1)
    
    xgboost_random_search.fit(Xtr, ytr)
    
    return (xgboost_random_search.best_params_, xgboost_random_search.best_score_)

def find_opt_params_lightgbm(Xtr, ytr):
    param_grid_lightgbm = {'max_depth': range(4, 21),
                          'min_child_samples': range(4, 31),
                          'subsample': [0.3, 0.4, 0.5, 0.6, 0.7],
                          'colsample_bytree': [0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
                          'learning_rate': [0.025, 0.05, 0.1, 0.2, 0.3],
                          'n_estimators': [30, 50, 70, 100, 150, 200]}
    
    lightgfm_classifier = LGBMClassifier(random_state=0)
    lightgfm_random_search = RandomizedSearchCV(lightgfm_classifier, param_grid_lightgbm, cv=scv,
                                  scoring="accuracy", n_iter=100,
                                  return_train_score=True, random_state=0,
                                  verbose=True,
                                  n_jobs=1)
    
    lightgfm_random_search.fit(Xtr, ytr)
    
    return (lightgfm_random_search.best_params_, lightgfm_random_search.best_score_)
    
def find_opt_params_random_forest(Xtr, ytr):
    param_grid_random_forest = {'max_depth': range(4, 21),
                             'max_features': [0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
                             'min_samples_leaf': range(4, 31),
                             'max_samples': [0.3, 0.4, 0.5, 0.6, 0.7],
                             'n_estimators': [30, 50, 70, 100, 150, 200]}
    randomforest_clf = RandomForestClassifier(random_state=0, bootstrap=True)
    randomforest_random_search = RandomizedSearchCV(randomforest_clf, param_grid_random_forest, cv=scv,
                                  scoring="accuracy", n_iter=100,
                                  return_train_score=True, random_state=0,
                                  verbose=True,
                                  n_jobs=1)
    
    randomforest_random_search.fit(Xtr, ytr)
    
    return (randomforest_random_search.best_params_, randomforest_random_search.best_score_)

def find_opt_params_extra_trees(Xtr, ytr):
    param_grid_extra_trees = {'max_depth': range(4, 21),
                             'max_features': [0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
                             'min_samples_leaf': range(4, 31),
                             'max_samples': [0.3, 0.4, 0.5, 0.6, 0.7],
                             'n_estimators': [30, 50, 70, 100, 150, 200]}
    
    extra_tree_clf = ExtraTreesClassifier(random_state=0, bootstrap=True)
    extra_tree_random_search = RandomizedSearchCV(extra_tree_clf, param_grid_extra_trees, cv=scv,
                                  scoring="accuracy", n_iter=100,
                                  return_train_score=True, random_state=0,
                                  verbose=True,
                                  n_jobs=1)
    
    extra_tree_random_search.fit(Xtr, ytr)
    
    return (extra_tree_random_search.best_params_, extra_tree_random_search.best_score_)

def find_opt_params_adaboost(Xtr, ytr):
    param_grid_adaboost = {'n_estimators': [30, 50, 70, 100, 150, 200],
                       'learning_rate': [0.001, 0.01, 0.1, 0.2, 0.5, 1],
                       'base_estimator__max_depth': range(4, 21),
                       'base_estimator__min_samples_leaf': range(4, 31)}
    adaboost_clf = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(random_state=0), random_state=0)
    adaboost_random_search = RandomizedSearchCV(adaboost_clf, param_grid_adaboost, cv=scv,
                                  scoring="accuracy", n_iter=100,
                                  return_train_score=True, random_state=0,
                                  verbose=True,
                                  n_jobs=1)
    adaboost_random_search.fit(Xtr, ytr)
    
    return (adaboost_random_search.best_params_, adaboost_random_search.best_score_)


def find_opt_params_svm(Xtr, ytr):
    svm_tuned_parameters = [{'kernel': ['rbf'], 'gamma': [2**-15, 2**-13, 2**-11, 2**-9, 2**-7, 2**-5, 2**-3, 2**-1, 2**1, 2**3], 'C': [2**-5, 2**-3, 2**-1, 2**1, 2**3, 2**5, 2**7, 2**9, 2**11, 2**13, 2**15]},
                        {'kernel': ['sigmoid'], 'gamma': [2**-15, 2**-13, 2**-11, 2**-9, 2**-7, 2**-5, 2**-3, 2**-1, 2**1, 2**3], 'C': [2**-5, 2**-3, 2**-1, 2**1, 2**3, 2**5, 2**7, 2**9, 2**11, 2**13, 2**15], 'coef0':[0.001, 0.01, 0.1, 1, 10, 100]},
                        {'kernel': ['linear'], 'C': [2**-5, 2**-3, 2**-1, 2**1, 2**3, 2**5, 2**7, 2**9, 2**11, 2**13, 2**15]}
                      ]
# {'kernel': ['poly'], 'gamma': [2**-15, 2**-13, 2**-11, 2**-9, 2**-7, 2**-5, 2**-3, 2**-1, 2**1, 2**3], 'C': [2**-5, 2**-3, 2**-1, 2**1, 2**3, 2**5, 2**7], 'degree':range(1, 11), 'coef0':[0.001, 0.01, 0.1, 1, 10, 100]}
    
    svc_clf = SVC(random_state=0)
    
    svc_clf_random_search = RandomizedSearchCV(svc_clf, svm_tuned_parameters, cv=scv,
                                  scoring="accuracy", n_iter=100,
                                  return_train_score=True, random_state=0,
                                  verbose=True,
                                  n_jobs=1)
    
    svc_clf_random_search.fit(Xtr, ytr)
    
    return (svc_clf_random_search.best_params_, svc_clf_random_search.best_score_)


def find_opt_params_decision_tree(Xtr, ytr):
    param_grid_decision_tree = {'max_depth': range(4, 21),
                                'min_samples_leaf': range(4, 31)}
    
    decision_tree_clf = DecisionTreeClassifier(random_state=0)
    decision_tree_random_search = RandomizedSearchCV(decision_tree_clf, param_grid_decision_tree, cv=scv,
                                  scoring="accuracy", n_iter=100, random_state=0,
                                  return_train_score=True,
                                  verbose=True,
                                  n_jobs=1)
    
    decision_tree_random_search.fit(Xtr, ytr)
    
    return (decision_tree_random_search.best_params_, decision_tree_random_search.best_score_)

def find_opt_params_knn(Xtr, ytr):
    param_grid_knn = {'n_neighbors' : range(1, 30, 2)}
    
    knn_clf = KNeighborsClassifier()
    knn_grid_search = GridSearchCV(knn_clf, param_grid_knn, cv=scv,
                                  scoring="accuracy",
                                  return_train_score=True,
                                  verbose=True,
                                  n_jobs=1)
    knn_grid_search.fit(Xtr, ytr)
    
    return (knn_grid_search.best_params_, knn_grid_search.best_score_)


if __name__ == '__main__':
    
    path = "/data/tkhuat/gfmm/Dataset/Electronic_data/"
    training_file = "TrainingDataset_30features_l1l2.csv"
    testing_file = "TestingDataset_30features_l1l2.csv"
    
    path_save = "/data/tkhuat/gfmm/Experiment/Electronic_data_l1l2/other_ml_f30/"
    if not os.path.exists(path_save):
        os.makedirs(path_save)
                
    file_acc_otherml = "acc_opt_other_ml.csv"
    file_prob_val_xgboost = "wrong_prob_xgboost.csv"
    file_prob_val_lightgbm = "wrong_prob_lightgbm.csv"
    file_prob_val_randomforest = "wrong_prob_randomforest.csv"
    file_prob_val_extratree = "wrong_prob_extratree.csv"
    file_prob_val_adaboost = "wrong_prob_adaboost.csv"
    file_prob_val_decision_tree = "wrong_prob_decisiontree.csv"
    file_prob_val_knn = "wrong_prob_knn.csv"
    file_prob_val_naivebayses = "wrong_prob_naivebayes.csv"
    file_prob_val_lda = "wrong_prob_lda.csv"
    file_prob_val_svm = "wrong_prob_svm.csv"
    file_important_feature_xgboost = "important_feature_xgboost.csv"
    file_important_feature_lightgbm = "important_feature_lightgbm.csv"
    file_important_feature_randomforest = "important_feature_randomforest.csv"
    file_important_feature_extratree = "important_feature_extratree.csv"
    file_important_feature_adaboost = "important_feature_adaboost.csv"
    file_important_feature_decision_tree = "important_feature_decisiontree.csv"
    file_important_feature_lda = "important_feature_lda.csv"
    
    df_train = pd.read_csv(path + training_file, header=None)
    df_test = pd.read_csv(path + testing_file, header=None)
    
    X_tr = df_train.iloc[:, :-1].values
    y_tr = df_train.iloc[:, -1].values
    
    X_test = df_test.iloc[:, :-1].values
    y_test = df_test.iloc[:, -1].values
    
    feature_name = []

    for i in range(X_tr.shape[1]):
        feature_name.append('F' + str(i+1))
        
    # For decision tree
    print("Decision tree")
    start_time = time.perf_counter()
    best_params_dt, best_val_dt = find_opt_params_decision_tree(X_tr, y_tr)
    opt_time_dt = time.perf_counter() - start_time
    
    decision_tree_clf = DecisionTreeClassifier(random_state=0)
    decision_tree_clf.set_params(**best_params_dt)
    
    start_time = time.perf_counter()
    decision_tree_clf.fit(X_tr, y_tr)
    training_time_dt = time.perf_counter() - start_time
    y_pred_dt = decision_tree_clf.predict(X_test)
    
    acc_dt = accuracy_score(y_test, y_pred_dt)
    cf_mat_dt = confusion_matrix(y_test, y_pred_dt)
    
    # save results for decision tree
    open(path_save + file_acc_otherml, 'w').close() # make existing file empty        
    with open(path_save + file_acc_otherml, 'a') as f_handle:
        f_handle.writelines("Algorithm, Testing Accuraccy, Confusion matrix, Validation accuracy, Training time, Optimisation time, best params \n")
        f_handle.writelines('Decision tree, %.6f, %s, %.6f, %.3f, %.3f, %s \n'%(acc_dt, str(cf_mat_dt), best_val_dt, training_time_dt, opt_time_dt, best_params_dt))
    
    id_nonexact_dt = np.nonzero(y_test != y_pred_dt)[0]
    extracted_prob_dt = decision_tree_clf.predict_proba(X_test[id_nonexact_dt])
    id_real_dt = id_nonexact_dt + 1
    saved_extracted_prob_dt = np.hstack((id_real_dt.reshape(-1, 1), extracted_prob_dt))
    
    open(path_save + file_prob_val_decision_tree, 'w').close() # make existing file empty        
    with open(path_save + file_prob_val_decision_tree, 'a') as f_handle:
        f_handle.writelines('testing sample index, prob predicition of class 1, prob predicition of class 2\n')
        np.savetxt(f_handle, saved_extracted_prob_dt, fmt='%s', delimiter=', ')
        
    dict_feature_score_dt = dict(zip(feature_name, decision_tree_clf.feature_importances_))
    sorted_dict_feature_score_dt = sorted(dict_feature_score_dt.items(), key=lambda x: x[1], reverse=True)
    open(path_save + file_important_feature_decision_tree, 'w').close() # make existing file empty        
    with open(path_save + file_important_feature_decision_tree, 'a') as f_handle:
        np.savetxt(f_handle, sorted_dict_feature_score_dt, fmt='%s', delimiter=', ')
        
        
    # For Gaussian Naive Bayes
    print("Gaussian Naive Bayes")
    gaussian_nb_clf = GaussianNB()
    start_time = time.perf_counter()
    gaussian_nb_clf.fit(X_tr, y_tr)
    training_time_gnb = time.perf_counter() - start_time
    y_pred_gnb = gaussian_nb_clf.predict(X_test)
    
    acc_gnb = accuracy_score(y_test, y_pred_gnb)
    cf_mat_gnb = confusion_matrix(y_test, y_pred_gnb)
    
    # save results for gaussian naive bayes
    with open(path_save + file_acc_otherml, 'a') as f_handle:
        f_handle.writelines('Gaussian naive bayes, %.6f, %s, -, %.3f, 0, \n'%(acc_gnb, str(cf_mat_gnb), training_time_gnb))
    
    id_nonexact_gnb = np.nonzero(y_test != y_pred_gnb)[0]
    extracted_prob_gnb = gaussian_nb_clf.predict_proba(X_test[id_nonexact_gnb])
    id_real_gnb = id_nonexact_gnb + 1
    saved_extracted_prob_gnb = np.hstack((id_real_gnb.reshape(-1, 1), extracted_prob_gnb))
    
    open(path_save + file_prob_val_naivebayses, 'w').close() # make existing file empty        
    with open(path_save + file_prob_val_naivebayses, 'a') as f_handle:
        f_handle.writelines('testing sample index, prob predicition of class 1, prob predicition of class 2\n')
        np.savetxt(f_handle, saved_extracted_prob_gnb, fmt='%s', delimiter=', ')
        
    # For Linear Discrinant Analysis
    print("Linear Discrinant Analysis")
    lda_clf = LinearDiscriminantAnalysis()
    
    start_time = time.perf_counter()
    lda_clf.fit(X_tr, y_tr)
    training_time_lda = time.perf_counter() - start_time
    y_pred_lda = lda_clf.predict(X_test)
    
    acc_lda = accuracy_score(y_test, y_pred_lda)
    cf_mat_lda = confusion_matrix(y_test, y_pred_lda)
    
    # save results for LDA
    with open(path_save + file_acc_otherml, 'a') as f_handle:
        f_handle.writelines('Linear Discriminant Analysis, %.6f, %s, -, %.3f, 0, \n'%(acc_lda, str(cf_mat_lda), training_time_lda))
    
    id_nonexact_lda = np.nonzero(y_test != y_pred_lda)[0]
    extracted_prob_lda = lda_clf.predict_proba(X_test[id_nonexact_lda])
    id_real_lda = id_nonexact_lda + 1
    saved_extracted_prob_lda = np.hstack((id_real_lda.reshape(-1, 1), extracted_prob_lda))
    
    open(path_save + file_prob_val_lda, 'w').close() # make existing file empty        
    with open(path_save + file_prob_val_lda, 'a') as f_handle:
        f_handle.writelines('testing sample index, prob predicition of class 1, prob predicition of class 2\n')
        np.savetxt(f_handle, saved_extracted_prob_lda, fmt='%s', delimiter=', ')
        
    dict_feature_score_lda = dict(zip(feature_name, lda_clf.coef_[0]))
    sorted_dict_feature_score_lda = sorted(dict_feature_score_lda.items(), key=lambda x: x[1], reverse=True)
    open(path_save + file_important_feature_lda, 'w').close() # make existing file empty        
    with open(path_save + file_important_feature_lda, 'a') as f_handle:
        np.savetxt(f_handle, sorted_dict_feature_score_lda, fmt='%s', delimiter=', ')

    # For KNN
    print("K-nearest neighbors")
    start_time = time.perf_counter()
    best_params_knn, best_val_knn = find_opt_params_knn(X_tr, y_tr)
    opt_time_knn = time.perf_counter() - start_time
    
    knn_clf = KNeighborsClassifier()
    knn_clf.set_params(**best_params_knn)
    
    start_time = time.perf_counter()
    knn_clf.fit(X_tr, y_tr)
    training_time_knn = time.perf_counter() - start_time
    y_pred_knn = knn_clf.predict(X_test)
    
    acc_knn = accuracy_score(y_test, y_pred_knn)
    cf_mat_knn = confusion_matrix(y_test, y_pred_knn)
    
    # save results for KNN
    with open(path_save + file_acc_otherml, 'a') as f_handle:
        f_handle.writelines('K-nearest neighbours, %.6f, %s, %.6f, %.3f, %.3f, %s \n'%(acc_knn, str(cf_mat_knn), best_val_knn, training_time_knn, opt_time_knn, best_params_knn))
    
    id_nonexact_knn = np.nonzero(y_test != y_pred_knn)[0]
    extracted_prob_knn = knn_clf.predict_proba(X_test[id_nonexact_knn])
    id_real_knn = id_nonexact_knn + 1
    saved_extracted_prob_knn = np.hstack((id_real_knn.reshape(-1, 1), extracted_prob_knn))
    
    open(path_save + file_prob_val_knn, 'w').close() # make existing file empty        
    with open(path_save + file_prob_val_knn, 'a') as f_handle:
        f_handle.writelines('testing sample index, prob predicition of class 1, prob predicition of class 2\n')
        np.savetxt(f_handle, saved_extracted_prob_knn, fmt='%s', delimiter=', ')

    # For random forest
    print("Random Forest")
    start_time = time.perf_counter()
    best_params_randomforest, best_val_randomforest = find_opt_params_random_forest(X_tr, y_tr)
    opt_time_randomforest = time.perf_counter() - start_time
    
    randomforest_clf = RandomForestClassifier(random_state=0, bootstrap=True)
    randomforest_clf.set_params(**best_params_randomforest)
    
    start_time = time.perf_counter()
    randomforest_clf.fit(X_tr, y_tr)
    training_time_randomforest = time.perf_counter() - start_time
    y_pred_randomforest = randomforest_clf.predict(X_test)
    
    acc_randomforest = accuracy_score(y_test, y_pred_randomforest)
    cf_mat_randomforest = confusion_matrix(y_test, y_pred_randomforest)
    
    ## save results for random forest
    with open(path_save + file_acc_otherml, 'a') as f_handle:
        f_handle.writelines('Random Forest, %.6f, %s, %.6f, %.3f, %.3f, %s \n'%(acc_randomforest, str(cf_mat_randomforest), best_val_randomforest, training_time_randomforest, opt_time_randomforest, best_params_randomforest))
    
    id_nonexact_randomforest = np.nonzero(y_test != y_pred_randomforest)[0]
    extracted_prob_randomforest = randomforest_clf.predict_proba(X_test[id_nonexact_randomforest])
    id_real_randomforest = id_nonexact_randomforest + 1
    saved_extracted_prob_randomforest = np.hstack((id_real_randomforest.reshape(-1, 1), extracted_prob_randomforest))
    
    open(path_save + file_prob_val_randomforest, 'w').close() # make existing file empty        
    with open(path_save + file_prob_val_randomforest, 'a') as f_handle:
        f_handle.writelines('testing sample index, prob predicition of class 1, prob predicition of class 2\n')
        np.savetxt(f_handle, saved_extracted_prob_randomforest, fmt='%s', delimiter=', ')
        
    dict_feature_score_randomforest = dict(zip(feature_name, randomforest_clf.feature_importances_))
    sorted_dict_feature_score_randomforest = sorted(dict_feature_score_randomforest.items(), key=lambda x: x[1], reverse=True)
    open(path_save + file_important_feature_randomforest, 'w').close() # make existing file empty        
    with open(path_save + file_important_feature_randomforest, 'a') as f_handle:
        np.savetxt(f_handle, sorted_dict_feature_score_randomforest, fmt='%s', delimiter=', ')
        
    # For extra trees
    print("Extra Trees")
    start_time = time.perf_counter()
    best_params_extratree, best_val_extratree = find_opt_params_extra_trees(X_tr, y_tr)
    opt_time_extratree = time.perf_counter() - start_time
    
    extratree_clf = ExtraTreesClassifier(random_state=0, bootstrap=True)
    extratree_clf.set_params(**best_params_extratree)
    
    start_time = time.perf_counter()
    extratree_clf.fit(X_tr, y_tr)
    training_time_extratree = time.perf_counter() - start_time
    y_pred_extratree = extratree_clf.predict(X_test)
    
    acc_extratree = accuracy_score(y_test, y_pred_extratree)
    cf_mat_extratree = confusion_matrix(y_test, y_pred_extratree)
    
    ## save results for extra trees
    with open(path_save + file_acc_otherml, 'a') as f_handle:
        f_handle.writelines('Extra Trees, %.6f, %s, %.6f, %.3f, %.3f, %s \n'%(acc_extratree, str(cf_mat_extratree), best_val_extratree, training_time_extratree, opt_time_extratree, best_params_extratree))
    
    id_nonexact_extratree = np.nonzero(y_test != y_pred_extratree)[0]
    extracted_prob_extratree = extratree_clf.predict_proba(X_test[id_nonexact_extratree])
    id_real_extratree = id_nonexact_extratree + 1
    saved_extracted_prob_extratree = np.hstack((id_real_extratree.reshape(-1, 1), extracted_prob_extratree))
    
    open(path_save + file_prob_val_extratree, 'w').close() # make existing file empty        
    with open(path_save + file_prob_val_extratree, 'a') as f_handle:
        f_handle.writelines('testing sample index, prob predicition of class 1, prob predicition of class 2\n')
        np.savetxt(f_handle, saved_extracted_prob_extratree, fmt='%s', delimiter=', ')
        
    dict_feature_score_extratree = dict(zip(feature_name, extratree_clf.feature_importances_))
    sorted_dict_feature_score_extratree = sorted(dict_feature_score_extratree.items(), key=lambda x: x[1], reverse=True)
    open(path_save + file_important_feature_extratree, 'w').close() # make existing file empty        
    with open(path_save + file_important_feature_extratree, 'a') as f_handle:
        np.savetxt(f_handle, sorted_dict_feature_score_extratree, fmt='%s', delimiter=', ')
        
    # For LightGBM
    print("LightGBM")
    start_time = time.perf_counter()
    best_params_lightgbm, best_val_lightgbm = find_opt_params_lightgbm(X_tr, y_tr)
    opt_time_lightgbm = time.perf_counter() - start_time
    
    lightgbm_clf = LGBMClassifier(random_state=0)
    lightgbm_clf.set_params(**best_params_lightgbm)
    
    start_time = time.perf_counter()
    lightgbm_clf.fit(X_tr, y_tr)
    training_time_lightgbm = time.perf_counter() - start_time
    y_pred_lightgbm = lightgbm_clf.predict(X_test)
    
    acc_lightgbm = accuracy_score(y_test, y_pred_lightgbm)
    cf_mat_lightgbm = confusion_matrix(y_test, y_pred_lightgbm)
    
    ## save results for LightGBM
    with open(path_save + file_acc_otherml, 'a') as f_handle:
        f_handle.writelines('Light Gradient Boosted Trees, %.6f, %s, %.6f, %.3f, %.3f, %s \n'%(acc_lightgbm, str(cf_mat_lightgbm), best_val_lightgbm,  training_time_lightgbm, opt_time_lightgbm, best_params_lightgbm))
    
    id_nonexact_lightgbm = np.nonzero(y_test != y_pred_lightgbm)[0]
    extracted_prob_lightgbm = lightgbm_clf.predict_proba(X_test[id_nonexact_lightgbm])
    id_real_lightgbm = id_nonexact_lightgbm + 1
    saved_extracted_prob_lightgbm = np.hstack((id_real_lightgbm.reshape(-1, 1), extracted_prob_lightgbm))
    
    open(path_save + file_prob_val_lightgbm, 'w').close() # make existing file empty        
    with open(path_save + file_prob_val_lightgbm, 'a') as f_handle:
        f_handle.writelines('testing sample index, prob predicition of class 1, prob predicition of class 2\n')
        np.savetxt(f_handle, saved_extracted_prob_lightgbm, fmt='%s', delimiter=', ')
        
    dict_feature_score_lightgbm = dict(zip(feature_name, lightgbm_clf.feature_importances_))
    sorted_dict_feature_score_lightgbm = sorted(dict_feature_score_lightgbm.items(), key=lambda x: x[1], reverse=True)
    open(path_save + file_important_feature_lightgbm, 'w').close() # make existing file empty        
    with open(path_save + file_important_feature_lightgbm, 'a') as f_handle:
        np.savetxt(f_handle, sorted_dict_feature_score_lightgbm, fmt='%s', delimiter=', ')
        
    # For XGBoost
    print("XGBoost")
    start_time = time.perf_counter()
    best_params_xgboost, best_val_xgboost = find_opt_params_xgboost(X_tr, y_tr)
    opt_time_xgboost = time.perf_counter() - start_time
    
    xgboost_clf = XGBClassifier(random_state=0)
    xgboost_clf.set_params(**best_params_xgboost)
    
    start_time = time.perf_counter()
    xgboost_clf.fit(X_tr, y_tr)
    training_time_xgboost = time.perf_counter() - start_time
    y_pred_xgboost = xgboost_clf.predict(X_test)
    
    acc_xgboost = accuracy_score(y_test, y_pred_xgboost)
    cf_mat_xgboost = confusion_matrix(y_test, y_pred_xgboost)
    
    ## save results for XGBoost
    with open(path_save + file_acc_otherml, 'a') as f_handle:
        f_handle.writelines('XGBoost, %.6f, %s, %.6f, %.3f, %.3f, %s \n'%(acc_xgboost, str(cf_mat_xgboost), best_val_xgboost, training_time_xgboost, opt_time_xgboost, best_params_xgboost))
    
    id_nonexact_xgboost = np.nonzero(y_test != y_pred_xgboost)[0]
    extracted_prob_xgboost = xgboost_clf.predict_proba(X_test[id_nonexact_xgboost])
    id_real_xgboost = id_nonexact_xgboost + 1
    saved_extracted_prob_xgboost = np.hstack((id_real_xgboost.reshape(-1, 1), extracted_prob_xgboost))
    
    open(path_save + file_prob_val_xgboost, 'w').close() # make existing file empty        
    with open(path_save + file_prob_val_xgboost, 'a') as f_handle:
        f_handle.writelines('testing sample index, prob predicition of class 1, prob predicition of class 2\n')
        np.savetxt(f_handle, saved_extracted_prob_xgboost, fmt='%s', delimiter=', ')
        
    dict_feature_score_xgboost = dict(zip(feature_name, xgboost_clf.feature_importances_))
    sorted_dict_feature_score_xgboost = sorted(dict_feature_score_xgboost.items(), key=lambda x: x[1], reverse=True)
    open(path_save + file_important_feature_xgboost, 'w').close() # make existing file empty        
    with open(path_save + file_important_feature_xgboost, 'a') as f_handle:
        np.savetxt(f_handle, sorted_dict_feature_score_xgboost, fmt='%s', delimiter=', ')
        
    # For AdaBoost
    print("AdaBoost")
    start_time = time.perf_counter()
    best_params_adaboost, best_val_adaboost = find_opt_params_adaboost(X_tr, y_tr)
    opt_time_adaboost = time.perf_counter() - start_time
    
    adaboost_clf = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(random_state=0), random_state=0)
    adaboost_clf.set_params(**best_params_adaboost)
    
    start_time = time.perf_counter()
    adaboost_clf.fit(X_tr, y_tr)
    training_time_adaboost = time.perf_counter() - start_time
    y_pred_adaboost = adaboost_clf.predict(X_test)
    
    acc_adaboost = accuracy_score(y_test, y_pred_adaboost)
    cf_mat_adaboost = confusion_matrix(y_test, y_pred_adaboost)
    
    ## save results for AdaBoost
    with open(path_save + file_acc_otherml, 'a') as f_handle:
        f_handle.writelines('AdaBoost, %.6f, %s, %.6f, %.3f, %.3f, %s \n'%(acc_adaboost, str(cf_mat_adaboost), best_val_adaboost, training_time_adaboost, opt_time_adaboost, best_params_adaboost))
    
    id_nonexact_adaboost = np.nonzero(y_test != y_pred_adaboost)[0]
    extracted_prob_adaboost = adaboost_clf.predict_proba(X_test[id_nonexact_adaboost])
    id_real_adaboost = id_nonexact_adaboost + 1
    saved_extracted_prob_adaboost = np.hstack((id_real_adaboost.reshape(-1, 1), extracted_prob_adaboost))
    
    open(path_save + file_prob_val_adaboost, 'w').close() # make existing file empty        
    with open(path_save + file_prob_val_adaboost, 'a') as f_handle:
        f_handle.writelines('testing sample index, prob predicition of class 1, prob predicition of class 2\n')
        np.savetxt(f_handle, saved_extracted_prob_adaboost, fmt='%s', delimiter=', ')
        
    dict_feature_score_adaboost = dict(zip(feature_name, adaboost_clf.feature_importances_))
    sorted_dict_feature_score_adaboost = sorted(dict_feature_score_adaboost.items(), key=lambda x: x[1], reverse=True)
    open(path_save + file_important_feature_adaboost, 'w').close() # make existing file empty        
    with open(path_save + file_important_feature_adaboost, 'a') as f_handle:
        np.savetxt(f_handle, sorted_dict_feature_score_adaboost, fmt='%s', delimiter=', ')
        
    # For SVM
    print("Support vector machines")
    start_time = time.perf_counter()
    best_params_svm, best_val_svm = find_opt_params_svm(X_tr, y_tr)
    opt_time_svm = time.perf_counter() - start_time
    
    svm_clf = SVC(probability=True)
    svm_clf.set_params(**best_params_svm)
    
    start_time = time.perf_counter()
    svm_clf.fit(X_tr, y_tr)
    training_time_svm = time.perf_counter() - start_time
    y_pred_svm = svm_clf.predict(X_test)
    
    acc_svm = accuracy_score(y_test, y_pred_svm)
    cf_mat_sv = confusion_matrix(y_test, y_pred_svm)
    
    ## save results for SVM
    with open(path_save + file_acc_otherml, 'a') as f_handle:
        f_handle.writelines('Support vector machines, %.6f, %s, %.6f, %.3f, %.3f, %s \n'%(acc_svm, str(cf_mat_sv), best_val_svm, training_time_svm, opt_time_svm, best_params_svm))
    
    id_nonexact_svm = np.nonzero(y_test != y_pred_svm)[0]
    extracted_prob_svm = svm_clf.predict_proba(X_test[id_nonexact_svm])
    id_real_svm = id_nonexact_svm + 1
    saved_extracted_prob_svm = np.hstack((id_real_svm.reshape(-1, 1), extracted_prob_svm))
    
    open(path_save + file_prob_val_svm, 'w').close() # make existing file empty        
    with open(path_save + file_prob_val_svm, 'a') as f_handle:
        f_handle.writelines('testing sample index, prob predicition of class 1, prob predicition of class 2\n')
        np.savetxt(f_handle, saved_extracted_prob_svm, fmt='%s', delimiter=', ')
        
    

