# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 10:01:05 2019

@author: Thanh Tung Khuat

This is script to run random boxes algorithms
"""

import sys, os
from os.path import dirname
root_path = dirname(dirname(os.getcwd()))
sys.path.insert(0, root_path) # insert root directory to environmental variables

import numpy as np
import math

from EFMN.efmnnclassification import EFMNNClassification
from EFMN.knefmnnclassification import KNEFMNNClassification
from EFMN.rfmnnclassification import RFMNNClassification
from FMNN.fmnnclassification import FMNNClassification
from GFMM.onlineagglogfmm import OnlineAggloGFMM
from GFMM.faster_accelbatchgfmm import AccelBatchGFMM

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from rotation_forest import RotationForestClassifier

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier

from functionhelper.preprocessinghelper import loadDataset
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold

if __name__ == '__main__':
    theta = 0.1
    theta_higher = 0.1
    
    str_theta = "_0_1"
    
    save_efmm_result_folder_path = root_path + '\\Experiment\\randombox\\no_pruning\\cv_random_box\\efmm\\'
    save_knefmm_result_folder_path = root_path + '\\Experiment\\randombox\\no_pruning\\cv_random_box\\knefmm\\'
    save_rfmnn_result_folder_path = root_path + '\\Experiment\\randombox\\no_pruning\\cv_random_box\\rfmnn\\'
    save_fmnn_result_folder_path = root_path + '\\Experiment\\randombox\\no_pruning\\cv_random_box\\fmnn\\'
    save_agglo2_result_folder_path = root_path + '\\Experiment\\randombox\\no_pruning\\cv_random_box\\agglo2\\'
    save_onln_agglo2_result_folder_path = root_path + '\\Experiment\\randombox\\no_pruning\\cv_random_box\\onlnagglo2\\'
    
    save_rf_result_folder_path = root_path + '\\Experiment\\randombox\\no_pruning\\cv_random_box\\random_forest\\'
    save_extra_tree_result_folder_path = root_path + '\\Experiment\\randombox\\no_pruning\\cv_random_box\\extra_tree\\'
    save_gradient_boosting_result_folder_path = root_path + '\\Experiment\\randombox\\no_pruning\\cv_random_box\\gradient_boosting\\'
    save_xgboost_result_folder_path = root_path + '\\Experiment\\randombox\\no_pruning\\cv_random_box\\xgboost\\'
    save_lightgbm_result_folder_path = root_path + '\\Experiment\\randombox\\no_pruning\\cv_random_box\\lightgbm\\'
    save_rotation_forest_result_folder_path = root_path + '\\Experiment\\randombox\\no_pruning\\cv_random_box\\rotation_forest\\'
    
    save_svm_result_folder_path = root_path + '\\Experiment\\randombox\\no_pruning\\cv_random_box\\svm\\'
    save_knn_result_folder_path = root_path + '\\Experiment\\randombox\\no_pruning\\cv_random_box\\knn\\'
    save_lda_result_folder_path = root_path + '\\Experiment\\randombox\\no_pruning\\cv_random_box\\lda\\'
    save_dt_result_folder_path = root_path + '\\Experiment\\randombox\\no_pruning\\cv_random_box\\decision_tree\\'
    save_naive_bayes_result_folder_path = root_path + '\\Experiment\\randombox\\no_pruning\\cv_random_box\\naive_bayes\\'
    
    dataset_path = root_path + '\\Dataset\\train_test\\cv\\'
    
    dataset_names = ['balance_scale', 'banknote_authentication', 'blood_transfusion', 'breast_cancer_wisconsin', 'BreastCancerCoimbra', 'climate_model_crashes', 'connectionist_bench_sonar', 'glass', 'haberman', 'heart', 'hill_valley_without_noise', 'ionosphere', 'movement_libras', 'optical_digit', 'page_blocks', 'pendigits', 'pima_diabetes', 'plant_species_leaves_margin', 'plant_species_leaves_shape', 'plant_species_leaves_texture', 'ringnorm', 'seeds', 'segmentation', 'spambase', 'SPECTF', 'statlog', 'twonorm', 'vehicle', 'vertebral_column', 'vowel', 'waveform', 'wireless_indoor_localization', 'yeast', 'letter', 'musk_v2']
    
    fold_index = np.array(range(40)) + 1
    
    gamma = 1
    n_estimators=100
    bootstrap_sample=True
    bootstrap_feature=False
    class_sample_rate=0.5
    n_jobs=1
    random_state=None
    K_threshold = 5 # K-nearest neighbor
    max_depth = 10
    
    for dt in range(len(dataset_names)):
        #try:
        print('Current dataset: ', dataset_names[dt])
        dataFile = dataset_path + dataset_names[dt] + '.dat'
        
        # Read data file
        foldData, _, foldLabel, _ = loadDataset(dataFile, 1, False)
        
        max_features= int(2 * math.sqrt(foldData.shape[1]))
        
        f1_weighted_efmnn_save = []
        f1_macro_efmnn_save = []
        f1_micro_efmnn_save = []
        
        f1_weighted_knefmnn_save = []
        f1_macro_knefmnn_save = []
        f1_micro_knefmnn_save = []
        
        f1_weighted_rfmnn_save = []
        f1_macro_rfmnn_save = []
        f1_micro_rfmnn_save = []
        
        f1_weighted_fmnn_save = []
        f1_macro_fmnn_save = []
        f1_micro_fmnn_save = []
        
        f1_weighted_rf_save = []
        f1_macro_rf_save = []
        f1_micro_rf_save = []
        
        f1_weighted_agglo2_save = []
        f1_macro_agglo2_save = []
        f1_micro_agglo2_save = []
        
        f1_weighted_onln_002_agglo2_save = []
        f1_macro_onln_002_agglo2_save = []
        f1_micro_onln_002_agglo2_save = []
        
        f1_weighted_onln_005_agglo2_save = []
        f1_macro_onln_005_agglo2_save = []
        f1_micro_onln_005_agglo2_save = []
        
        f1_weighted_extra_tree_save = []
        f1_macro_extra_tree_save = []
        f1_micro_extra_tree_save = []
        
        f1_weighted_gradient_boosting_save = []
        f1_macro_gradient_boosting_save = []
        f1_micro_gradient_boosting_save = []
        
        f1_weighted_xgboost_save = []
        f1_macro_xgboost_save = []
        f1_micro_xgboost_save = []
        
        f1_weighted_lightgbm_save = []
        f1_macro_lightgbm_save = []
        f1_micro_lightgbm_save = []
        
        f1_weighted_rotation_forest_save = []
        f1_macro_rotation_forest_save = []
        f1_micro_rotation_forest_save = []
        
        f1_weighted_svm_save = []
        f1_macro_svm_save = []
        f1_micro_svm_save = []
        
        f1_weighted_knn_save = []
        f1_macro_knn_save = []
        f1_micro_knn_save = []
        
        f1_weighted_lda_save = []
        f1_macro_lda_save = []
        f1_micro_lda_save = []
        
        f1_weighted_decision_tree_save = []
        f1_macro_decision_tree_save = []
        f1_micro_decision_tree_save = []
        
        f1_weighted_naive_bayes_save = []
        f1_macro_naive_bayes_save = []
        f1_micro_naive_bayes_save = []
        
        for it in range(10):
            skf = StratifiedKFold(n_splits=4, random_state=it)
            for train_index, test_index in skf.split(foldData, foldLabel):
                trainingData, testingData = foldData[train_index], foldData[test_index]
                trainingLabel, testingLabel = foldLabel[train_index], foldLabel[test_index]
                
                efmnn = EFMNNClassification(gamma = 1, teta = theta, isDraw = False, isNorm = False)
                efmnn.fit(trainingData, trainingLabel)               
                result_efmnn = efmnn.predict(testingData, testingLabel)
                predicted_efmnn = np.array(result_efmnn.predicted_class, dtype=np.int)                
                f1_weighted_efmnn_save.append(f1_score(testingLabel, predicted_efmnn, average='weighted'))
                f1_macro_efmnn_save.append(f1_score(testingLabel, predicted_efmnn, average='macro'))
                f1_micro_efmnn_save.append(f1_score(testingLabel, predicted_efmnn, average='micro'))                
                
                knefmnn = KNEFMNNClassification(gamma = 1, teta = theta, isDraw = False, isNorm = False)
                knefmnn.fit(trainingData, trainingLabel, K_threshold)                
                result_knefmnn_manhat = knefmnn.predict(testingData, testingLabel)
                predicted_knefmnn = np.array(result_knefmnn_manhat.predicted_class, dtype=np.int)
                f1_weighted_knefmnn_save.append(f1_score(testingLabel, predicted_knefmnn, average='weighted'))
                f1_macro_knefmnn_save.append(f1_score(testingLabel, predicted_knefmnn, average='macro'))
                f1_micro_knefmnn_save.append(f1_score(testingLabel, predicted_knefmnn, average='micro'))                
                
                rfmnn = RFMNNClassification(gamma = 1, teta = theta, isNorm = False)
                rfmnn.fit(trainingData, trainingLabel)                
                result_rfmnn = rfmnn.predict(testingData, testingLabel)
                predicted_rfmnn = np.array(result_rfmnn.predicted_class, dtype=np.int)                
                f1_weighted_rfmnn_save.append(f1_score(testingLabel, predicted_rfmnn, average='weighted'))
                f1_macro_rfmnn_save.append(f1_score(testingLabel, predicted_rfmnn, average='macro'))
                f1_micro_rfmnn_save.append(f1_score(testingLabel, predicted_rfmnn, average='micro'))
                
                fmnn = FMNNClassification(gamma = 1, teta = theta, isDraw = False, isNorm = False)
                fmnn.fit(trainingData, trainingLabel)                
                result_fmnn = fmnn.predict(testingData, testingLabel)
                predicted_fmnn = np.array(result_fmnn.predicted_class, dtype=np.int)               
                f1_weighted_fmnn_save.append(f1_score(testingLabel, predicted_fmnn, average='weighted'))
                f1_macro_fmnn_save.append(f1_score(testingLabel, predicted_fmnn, average='macro'))
                f1_micro_fmnn_save.append(f1_score(testingLabel, predicted_fmnn, average='micro'))
                
                agglo2 = AccelBatchGFMM(gamma = 1, teta = theta, bthres = 0, simil = 'long', sing = 'max', isDraw = False, oper = 'min', isNorm = False)
                agglo2.fit(trainingData, trainingData, trainingLabel)
                result_agglo2 = agglo2.predict(testingData, testingData, testingLabel)
                predicted_agglo2 = np.array(result_agglo2.predicted_class, dtype=np.int)
                f1_weighted_agglo2_save.append(f1_score(testingLabel, predicted_agglo2, average='weighted'))
                f1_macro_agglo2_save.append(f1_score(testingLabel, predicted_agglo2, average='macro'))
                f1_micro_agglo2_save.append(f1_score(testingLabel, predicted_agglo2, average='micro'))
                
                onlnaggo2_002 = OnlineAggloGFMM(gamma = 1, teta_onl = 0.02, teta_agglo = theta, bthres = 0, simil = 'long', sing = 'max', isDraw = False, oper = 'min', isNorm = False)
                onlnaggo2_002.fit(trainingData, trainingData, trainingLabel)
                result_onlnagglo2_002 = onlnaggo2_002.predict(testingData, testingData, testingLabel)
                predicted_onlnagglo2_002 = np.array(result_onlnagglo2_002.predicted_class, dtype=np.int)
                f1_weighted_onln_002_agglo2_save.append(f1_score(testingLabel, predicted_onlnagglo2_002, average='weighted'))
                f1_macro_onln_002_agglo2_save.append(f1_score(testingLabel, predicted_onlnagglo2_002, average='macro'))
                f1_micro_onln_002_agglo2_save.append(f1_score(testingLabel, predicted_onlnagglo2_002, average='micro'))
                
                onlnaggo2_005 = OnlineAggloGFMM(gamma = 1, teta_onl = 0.05, teta_agglo = theta, bthres = 0, simil = 'long', sing = 'max', isDraw = False, oper = 'min', isNorm = False)
                onlnaggo2_005.fit(trainingData, trainingData, trainingLabel)
                result_onlnagglo2_005 = onlnaggo2_005.predict(testingData, testingData, testingLabel)
                predicted_onlnagglo2_005 = np.array(result_onlnagglo2_005.predicted_class, dtype=np.int)
                f1_weighted_onln_005_agglo2_save.append(f1_score(testingLabel, predicted_onlnagglo2_005, average='weighted'))
                f1_macro_onln_005_agglo2_save.append(f1_score(testingLabel, predicted_onlnagglo2_005, average='macro'))
                f1_micro_onln_005_agglo2_save.append(f1_score(testingLabel, predicted_onlnagglo2_005, average='micro'))
                
                random_forest = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, max_features=max_features, random_state=0)
                random_forest.fit(trainingData, trainingLabel)
                rf_result = random_forest.predict(testingData)                
                f1_weighted_rf_save.append(f1_score(testingLabel, rf_result, average='weighted'))
                f1_macro_rf_save.append(f1_score(testingLabel, rf_result, average='macro'))
                f1_micro_rf_save.append(f1_score(testingLabel, rf_result, average='micro'))
                
                extra_tree = ExtraTreesClassifier(n_estimators=n_estimators, max_depth=max_depth, max_features=max_features, random_state=0)
                extra_tree.fit(trainingData, trainingLabel)
                et_result = extra_tree.predict(testingData)
                f1_weighted_extra_tree_save.append(f1_score(testingLabel, et_result, average='weighted'))
                f1_macro_extra_tree_save.append(f1_score(testingLabel, et_result, average='macro'))
                f1_micro_extra_tree_save.append(f1_score(testingLabel, et_result, average='micro'))
                
                gbc = GradientBoostingClassifier(n_estimators=n_estimators, max_depth=max_depth, max_features=max_features, random_state=0, subsample=class_sample_rate)
                gbc.fit(trainingData, trainingLabel)
                gbc_result = gbc.predict(testingData)
                f1_weighted_gradient_boosting_save.append(f1_score(testingLabel, gbc_result, average='weighted'))
                f1_macro_gradient_boosting_save.append(f1_score(testingLabel, gbc_result, average='macro'))
                f1_micro_gradient_boosting_save.append(f1_score(testingLabel, gbc_result, average='micro'))
                
                xgb = XGBClassifier(n_estimators=n_estimators, max_depth=max_depth, objective='binary:logistic', subsample=class_sample_rate, colsample_bytree=max_features/foldData.shape[1], random_state=0)
                xgb.fit(trainingData, trainingLabel)
                xgb_result = xgb.predict(testingData)
                f1_weighted_xgboost_save.append(f1_score(testingLabel, xgb_result, average='weighted'))
                f1_macro_xgboost_save.append(f1_score(testingLabel, xgb_result, average='macro'))
                f1_micro_xgboost_save.append(f1_score(testingLabel, xgb_result, average='micro'))
                
                lgbm = LGBMClassifier(max_depth=max_depth, n_estimators=n_estimators, subsample=class_sample_rate, colsample_bytree=max_features/foldData.shape[1], random_state=0)
                lgbm.fit(trainingData, trainingLabel)
                lgbm_result = lgbm.predict(testingData)
                f1_weighted_lightgbm_save.append(f1_score(testingLabel, lgbm_result, average='weighted'))
                f1_macro_lightgbm_save.append(f1_score(testingLabel, lgbm_result, average='macro'))
                f1_micro_lightgbm_save.append(f1_score(testingLabel, lgbm_result, average='micro'))
                
                rot_forest = RotationForestClassifier(n_estimators=n_estimators, max_depth=max_depth, max_features=max_features, random_state=0)
                rot_forest.fit(trainingData, trainingLabel)
                rot_forest_result = rot_forest.predict(testingData)
                f1_weighted_rotation_forest_save.append(f1_score(testingLabel, rot_forest_result, average='weighted'))
                f1_macro_rotation_forest_save.append(f1_score(testingLabel, rot_forest_result, average='macro'))
                f1_micro_rotation_forest_save.append(f1_score(testingLabel, rot_forest_result, average='micro'))
                
                svm_classifier = SVC(kernel='rbf')
                svm_classifier.fit(trainingData, trainingLabel)
                svm_result = svm_classifier.predict(testingData)
                f1_weighted_svm_save.append(f1_score(testingLabel, svm_result, average='weighted'))
                f1_macro_svm_save.append(f1_score(testingLabel, svm_result, average='macro'))
                f1_micro_svm_save.append(f1_score(testingLabel, svm_result, average='micro'))
                
                knn = KNeighborsClassifier(n_neighbors=K_threshold)
                knn.fit(trainingData, trainingLabel)
                knn_result = knn.predict(testingData)
                f1_weighted_knn_save.append(f1_score(testingLabel, knn_result, average='weighted'))
                f1_macro_knn_save.append(f1_score(testingLabel, knn_result, average='macro'))
                f1_micro_knn_save.append(f1_score(testingLabel, knn_result, average='micro'))
                
                lda = LinearDiscriminantAnalysis()
                lda.fit(trainingData, trainingLabel)
                lda_result = lda.predict(testingData)
                f1_weighted_lda_save.append(f1_score(testingLabel, lda_result, average='weighted'))
                f1_macro_lda_save.append(f1_score(testingLabel, lda_result, average='macro'))
                f1_micro_lda_save.append(f1_score(testingLabel, lda_result, average='micro'))
                
                dtree = DecisionTreeClassifier(max_depth=max_depth, max_features=max_features, random_state=0)
                dtree.fit(trainingData, trainingLabel)
                dtree_result = dtree.predict(testingData)
                f1_weighted_decision_tree_save.append(f1_score(testingLabel, dtree_result, average='weighted'))
                f1_macro_decision_tree_save.append(f1_score(testingLabel, dtree_result, average='macro'))
                f1_micro_decision_tree_save.append(f1_score(testingLabel, dtree_result, average='micro'))
                
                gnb = GaussianNB()
                gnb.fit(trainingData, trainingLabel)
                gnb_result = gnb.predict(testingData)
                f1_weighted_naive_bayes_save.append(f1_score(testingLabel, gnb_result, average='weighted'))
                f1_macro_naive_bayes_save.append(f1_score(testingLabel, gnb_result, average='macro'))
                f1_micro_naive_bayes_save.append(f1_score(testingLabel, gnb_result, average='micro'))
                                
                
        # save random hyperboxes using improve online gfmm
        f1_weighted_efmnn_save = np.array(f1_weighted_efmnn_save)
        f1_macro_efmnn_save = np.array(f1_macro_efmnn_save)
        f1_micro_efmnn_save = np.array(f1_micro_efmnn_save)
        
        f1_weighted_knefmnn_save = np.array(f1_weighted_knefmnn_save)
        f1_macro_knefmnn_save = np.array(f1_macro_knefmnn_save)
        f1_micro_knefmnn_save = np.array(f1_micro_knefmnn_save)
        
        f1_weighted_rfmnn_save = np.array(f1_weighted_rfmnn_save)
        f1_macro_rfmnn_save = np.array(f1_macro_rfmnn_save)
        f1_micro_rfmnn_save = np.array(f1_micro_rfmnn_save)
        
        f1_weighted_fmnn_save = np.array(f1_weighted_fmnn_save)
        f1_macro_fmnn_save = np.array(f1_macro_fmnn_save)
        f1_micro_fmnn_save = np.array(f1_micro_fmnn_save)
        
        f1_weighted_rf_save = np.array(f1_weighted_rf_save)
        f1_macro_rf_save = np.array(f1_macro_rf_save)
        f1_micro_rf_save = np.array(f1_micro_rf_save)
        
        f1_weighted_agglo2_save = np.array(f1_weighted_agglo2_save)
        f1_macro_agglo2_save = np.array(f1_macro_agglo2_save)
        f1_micro_agglo2_save = np.array(f1_micro_agglo2_save)
        
        f1_weighted_onln_002_agglo2_save = np.array(f1_weighted_onln_002_agglo2_save)
        f1_macro_onln_002_agglo2_save = np.array(f1_macro_onln_002_agglo2_save)
        f1_micro_onln_002_agglo2_save = np.array(f1_micro_onln_002_agglo2_save)
        
        f1_weighted_onln_005_agglo2_save = np.array(f1_weighted_onln_005_agglo2_save)
        f1_macro_onln_005_agglo2_save = np.array(f1_macro_onln_005_agglo2_save)
        f1_micro_onln_005_agglo2_save = np.array(f1_micro_onln_005_agglo2_save)
        
        f1_weighted_extra_tree_save = np.array(f1_weighted_extra_tree_save)
        f1_macro_extra_tree_save = np.array(f1_macro_extra_tree_save)
        f1_micro_extra_tree_save = np.array(f1_micro_extra_tree_save)
        
        f1_weighted_gradient_boosting_save = np.array(f1_weighted_gradient_boosting_save)
        f1_macro_gradient_boosting_save = np.array(f1_macro_gradient_boosting_save)
        f1_micro_gradient_boosting_save = np.array(f1_micro_gradient_boosting_save)
        
        f1_weighted_xgboost_save = np.array(f1_weighted_xgboost_save)
        f1_macro_xgboost_save = np.array(f1_macro_xgboost_save)
        f1_micro_xgboost_save = np.array(f1_micro_xgboost_save)
        
        f1_weighted_lightgbm_save = np.array(f1_weighted_lightgbm_save)
        f1_macro_lightgbm_save = np.array(f1_macro_lightgbm_save)
        f1_micro_lightgbm_save = np.array(f1_micro_lightgbm_save)
        
        f1_weighted_rotation_forest_save = np.array(f1_weighted_rotation_forest_save)
        f1_macro_rotation_forest_save = np.array(f1_macro_rotation_forest_save)
        f1_micro_rotation_forest_save = np.array(f1_micro_rotation_forest_save)
        
        f1_weighted_svm_save = np.array(f1_weighted_svm_save)
        f1_macro_svm_save = np.array(f1_macro_svm_save)
        f1_micro_svm_save = np.array(f1_micro_svm_save)
        
        f1_weighted_knn_save = np.array(f1_weighted_knn_save)
        f1_macro_knn_save = np.array(f1_macro_knn_save)
        f1_micro_knn_save = np.array(f1_micro_knn_save)
        
        f1_weighted_lda_save = np.array(f1_weighted_lda_save)
        f1_macro_lda_save = np.array(f1_macro_lda_save)
        f1_micro_lda_save = np.array(f1_micro_lda_save)
        
        f1_weighted_decision_tree_save = np.array(f1_weighted_decision_tree_save)
        f1_macro_decision_tree_save = np.array(f1_macro_decision_tree_save)
        f1_micro_decision_tree_save = np.array(f1_micro_decision_tree_save)
        
        f1_weighted_naive_bayes_save = np.array(f1_weighted_naive_bayes_save)
        f1_macro_naive_bayes_save = np.array(f1_macro_naive_bayes_save)
        f1_micro_naive_bayes_save = np.array(f1_micro_naive_bayes_save)
        
        # defile location to save data
        filename_efmnn = save_efmm_result_folder_path + dataset_names[dt] + str_theta + '.csv'
        filename_knefmnn = save_knefmm_result_folder_path + dataset_names[dt] + str_theta + '.csv'
        filename_rfmnn = save_rfmnn_result_folder_path + dataset_names[dt] + str_theta + '.csv'
        filename_fmnn = save_fmnn_result_folder_path + dataset_names[dt] + str_theta + '.csv'
        filename_random_forest = save_rf_result_folder_path + dataset_names[dt] + str_theta + '.csv'
        filename_rotation_forest = save_rotation_forest_result_folder_path + dataset_names[dt] + str_theta + '.csv'
        filename_extra_tree = save_extra_tree_result_folder_path + dataset_names[dt] + str_theta + '.csv'
        filename_xgboost = save_xgboost_result_folder_path + dataset_names[dt] + str_theta + '.csv'
        filename_lgb = save_lightgbm_result_folder_path + dataset_names[dt] + str_theta + '.csv'
        filename_gradient_boosting = save_gradient_boosting_result_folder_path + dataset_names[dt] + str_theta + '.csv'
        filename_svm = save_svm_result_folder_path + dataset_names[dt] + str_theta + '.csv'
        filename_decision_tree = save_dt_result_folder_path + dataset_names[dt] + str_theta + '.csv'
        filename_knn = save_knn_result_folder_path + dataset_names[dt] + str_theta + '.csv'
        filename_naive_bayes = save_naive_bayes_result_folder_path + dataset_names[dt] + str_theta + '.csv'
        filename_lda = save_lda_result_folder_path + dataset_names[dt] + str_theta + '.csv'
        filename_agglo2 = save_agglo2_result_folder_path + dataset_names[dt] + str_theta + '.csv'
        filename_onlnagglo2_002 = save_onln_agglo2_result_folder_path + dataset_names[dt] + 'ol002' + str_theta + '.csv'
        filename_onlnagglo2_005 = save_onln_agglo2_result_folder_path + dataset_names[dt] + 'ol005' + str_theta + '.csv'
        
        # prepare data to save
        data_efmnn = np.hstack((fold_index.reshape(-1, 1), f1_weighted_efmnn_save.reshape(-1, 1), f1_macro_efmnn_save.reshape(-1, 1), f1_micro_efmnn_save.reshape(-1, 1)))
        data_knefmnn = np.hstack((fold_index.reshape(-1, 1), f1_weighted_knefmnn_save.reshape(-1, 1), f1_macro_knefmnn_save.reshape(-1, 1), f1_micro_knefmnn_save.reshape(-1, 1)))        
        data_rfmnn = np.hstack((fold_index.reshape(-1, 1), f1_weighted_rfmnn_save.reshape(-1, 1), f1_macro_rfmnn_save.reshape(-1, 1), f1_micro_rfmnn_save.reshape(-1, 1)))
        data_fmnn = np.hstack((fold_index.reshape(-1, 1), f1_weighted_fmnn_save.reshape(-1, 1), f1_macro_fmnn_save.reshape(-1, 1), f1_micro_fmnn_save.reshape(-1, 1)))
        data_random_forest = np.hstack((fold_index.reshape(-1, 1), f1_weighted_rf_save.reshape(-1, 1), f1_macro_rf_save.reshape(-1, 1), f1_micro_rf_save.reshape(-1, 1)))
        data_rotation_forest = np.hstack((fold_index.reshape(-1, 1), f1_weighted_rotation_forest_save.reshape(-1, 1), f1_macro_rotation_forest_save.reshape(-1, 1), f1_micro_rotation_forest_save.reshape(-1, 1)))
        data_extra_tree = np.hstack((fold_index.reshape(-1, 1), f1_weighted_extra_tree_save.reshape(-1, 1), f1_macro_extra_tree_save.reshape(-1, 1), f1_micro_extra_tree_save.reshape(-1, 1)))
        data_xgboost = np.hstack((fold_index.reshape(-1, 1), f1_weighted_xgboost_save.reshape(-1, 1), f1_macro_xgboost_save.reshape(-1, 1), f1_micro_xgboost_save.reshape(-1, 1)))
        data_lightgbm = np.hstack((fold_index.reshape(-1, 1), f1_weighted_lightgbm_save.reshape(-1, 1), f1_macro_lightgbm_save.reshape(-1, 1), f1_micro_lightgbm_save.reshape(-1, 1)))
        data_gradient_boosting = np.hstack((fold_index.reshape(-1, 1), f1_weighted_gradient_boosting_save.reshape(-1, 1), f1_macro_gradient_boosting_save.reshape(-1, 1), f1_micro_gradient_boosting_save.reshape(-1, 1)))
        data_svm = np.hstack((fold_index.reshape(-1, 1), f1_weighted_svm_save.reshape(-1, 1), f1_macro_svm_save.reshape(-1, 1), f1_micro_svm_save.reshape(-1, 1)))
        data_knn = np.hstack((fold_index.reshape(-1, 1), f1_weighted_knn_save.reshape(-1, 1), f1_macro_knn_save.reshape(-1, 1), f1_micro_knn_save.reshape(-1, 1)))
        data_decision_tree = np.hstack((fold_index.reshape(-1, 1), f1_weighted_decision_tree_save.reshape(-1, 1), f1_macro_decision_tree_save.reshape(-1, 1), f1_micro_decision_tree_save.reshape(-1, 1)))
        data_naive_bayes = np.hstack((fold_index.reshape(-1, 1), f1_weighted_naive_bayes_save.reshape(-1, 1), f1_macro_naive_bayes_save.reshape(-1, 1), f1_micro_naive_bayes_save.reshape(-1, 1)))
        data_lda = np.hstack((fold_index.reshape(-1, 1), f1_weighted_lda_save.reshape(-1, 1), f1_macro_lda_save.reshape(-1, 1), f1_micro_lda_save.reshape(-1, 1)))
        data_agglo2 = np.hstack((fold_index.reshape(-1, 1), f1_weighted_agglo2_save.reshape(-1, 1), f1_macro_agglo2_save.reshape(-1, 1), f1_micro_agglo2_save.reshape(-1, 1)))
        data_onlnagglo2_002 = np.hstack((fold_index.reshape(-1, 1), f1_weighted_onln_002_agglo2_save.reshape(-1, 1), f1_macro_onln_002_agglo2_save.reshape(-1, 1), f1_micro_onln_002_agglo2_save.reshape(-1, 1)))
        data_onlnagglo2_005 = np.hstack((fold_index.reshape(-1, 1), f1_weighted_onln_005_agglo2_save.reshape(-1, 1), f1_macro_onln_005_agglo2_save.reshape(-1, 1), f1_micro_onln_005_agglo2_save.reshape(-1, 1)))
        
        open(filename_efmnn, 'w').close() # make existing file empty        
        with open(filename_efmnn,'a') as f_handle:
            f_handle.writelines('Fold, F1-weighted avg, F1-macro avg, F1-micro\n')
            np.savetxt(f_handle, data_efmnn, fmt='%s', delimiter=', ')
            
        open(filename_knefmnn, 'w').close() # make existing file empty       
        with open(filename_knefmnn,'a') as f_handle:
            f_handle.writelines('Fold, F1-weighted avg, F1-macro avg, F1-micro\n')
            np.savetxt(f_handle, data_knefmnn, fmt='%s', delimiter=', ')
            
        open(filename_rfmnn, 'w').close() # make existing file empty        
        with open(filename_rfmnn,'a') as f_handle:
            f_handle.writelines('Fold, F1-weighted avg, F1-macro avg, F1-micro\n')
            np.savetxt(f_handle, data_rfmnn, fmt='%s', delimiter=', ')
            
        open(filename_fmnn, 'w').close() # make existing file empty
        with open(filename_fmnn,'a') as f_handle:
            f_handle.writelines('Fold, F1-weighted avg, F1-macro avg, F1-micro\n')
            np.savetxt(f_handle, data_fmnn, fmt='%s', delimiter=', ')
            
        open(filename_random_forest, 'w').close() # make existing file empty
        with open(filename_random_forest,'a') as f_handle:
            f_handle.writelines('Fold, F1-weighted avg, F1-macro avg, F1-micro\n')
            np.savetxt(f_handle, data_random_forest, fmt='%s', delimiter=', ')
            
        open(filename_rotation_forest, 'w').close() # make existing file empty
        with open(filename_rotation_forest,'a') as f_handle:
            f_handle.writelines('Fold, F1-weighted avg, F1-macro avg, F1-micro\n')
            np.savetxt(f_handle, data_rotation_forest, fmt='%s', delimiter=', ')
            
        open(filename_extra_tree, 'w').close() # make existing file empty
        with open(filename_extra_tree,'a') as f_handle:
            f_handle.writelines('Fold, F1-weighted avg, F1-macro avg, F1-micro\n')
            np.savetxt(f_handle, data_extra_tree, fmt='%s', delimiter=', ')
            
        open(filename_xgboost, 'w').close() # make existing file empty
        with open(filename_xgboost,'a') as f_handle:
            f_handle.writelines('Fold, F1-weighted avg, F1-macro avg, F1-micro\n')
            np.savetxt(f_handle, data_xgboost, fmt='%s', delimiter=', ')
            
        open(filename_lgb, 'w').close() # make existing file empty
        with open(filename_lgb,'a') as f_handle:
            f_handle.writelines('Fold, F1-weighted avg, F1-macro avg, F1-micro\n')
            np.savetxt(f_handle, data_lightgbm, fmt='%s', delimiter=', ')
            
        open(filename_gradient_boosting, 'w').close() # make existing file empty
        with open(filename_gradient_boosting,'a') as f_handle:
            f_handle.writelines('Fold, F1-weighted avg, F1-macro avg, F1-micro\n')
            np.savetxt(f_handle, data_gradient_boosting, fmt='%s', delimiter=', ')
            
        open(filename_svm, 'w').close() # make existing file empty
        with open(filename_svm,'a') as f_handle:
            f_handle.writelines('Fold, F1-weighted avg, F1-macro avg, F1-micro\n')
            np.savetxt(f_handle, data_svm, fmt='%s', delimiter=', ')
            
        open(filename_knn, 'w').close() # make existing file empty
        with open(filename_knn,'a') as f_handle:
            f_handle.writelines('Fold, F1-weighted avg, F1-macro avg, F1-micro\n')
            np.savetxt(f_handle, data_knn, fmt='%s', delimiter=', ')
            
        open(filename_lda, 'w').close() # make existing file empty
        with open(filename_lda,'a') as f_handle:
            f_handle.writelines('Fold, F1-weighted avg, F1-macro avg, F1-micro\n')
            np.savetxt(f_handle, data_lda, fmt='%s', delimiter=', ')
            
        open(filename_naive_bayes, 'w').close() # make existing file empty
        with open(filename_naive_bayes,'a') as f_handle:
            f_handle.writelines('Fold, F1-weighted avg, F1-macro avg, F1-micro\n')
            np.savetxt(f_handle, data_naive_bayes, fmt='%s', delimiter=', ')
            
        open(filename_decision_tree, 'w').close() # make existing file empty
        with open(filename_decision_tree,'a') as f_handle:
            f_handle.writelines('Fold, F1-weighted avg, F1-macro avg, F1-micro\n')
            np.savetxt(f_handle, data_decision_tree, fmt='%s', delimiter=', ')
            
        open(filename_agglo2, 'w').close() # make existing file empty
        with open(filename_agglo2,'a') as f_handle:
            f_handle.writelines('Fold, F1-weighted avg, F1-macro avg, F1-micro\n')
            np.savetxt(f_handle, data_agglo2, fmt='%s', delimiter=', ')
            
        open(filename_onlnagglo2_002, 'w').close() # make existing file empty
        with open(filename_onlnagglo2_002,'a') as f_handle:
            f_handle.writelines('Fold, F1-weighted avg, F1-macro avg, F1-micro\n')
            np.savetxt(f_handle, data_onlnagglo2_002, fmt='%s', delimiter=', ')
            
        open(filename_onlnagglo2_005, 'w').close() # make existing file empty
        with open(filename_onlnagglo2_005,'a') as f_handle:
            f_handle.writelines('Fold, F1-weighted avg, F1-macro avg, F1-micro\n')
            np.savetxt(f_handle, data_onlnagglo2_005, fmt='%s', delimiter=', ')
            
            
        #        except:
#            pass
        
    print('---Finish---')

