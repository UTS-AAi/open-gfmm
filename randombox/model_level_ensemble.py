# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 14:09:02 2019

@author: Thanh Tung Khuat

This is the implementation of model-based combination of GFMM models

"""
import sys, os
sys.path.insert(0, os.path.pardir)

import numpy as np
import math
import time
import random
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from sklearn.utils import resample
from functionhelper.preprocessinghelper import normalize
from GFMM.improvedonlinegfmm import ImprovedOnlineGFMM
from GFMM.faster_onlinegfmm import OnlineGFMM
from GFMM.faster_accelbatchgfmm import AccelBatchGFMM
from functionhelper.preprocessinghelper import loadDataset, string_to_boolean
from randombox.utils import find_max_mode
from sklearn.metrics import accuracy_score, classification_report
from functionhelper.membershipcalc import memberG
from randombox.decision_ensemble import DecisionLevelCombination
from GFMM.classification import predict_with_probability, predict_with_manhattan

def get_num_cpu_cores():
    num_cores = multiprocessing.cpu_count()
    return num_cores

class ModelLevelCombination(DecisionLevelCombination):
    """
    An ensemble of hyperbox-based classifier.
        
    Parameters:
    ----------
        n_estimators : integer, optional (default=100)
            The number of classifiers in the model.
    
        theta : float, optional (default = 0.2)
            The maximum hyperbox size of each base model
            
        higher_level_theta: float, optinal (default=0.2)
            The maximum hyperbox size to combine at the model level
            
        bootstrap_sample : boolean, optional (default=False)
            Whether bootstrap samples are used when building hyperbox-based models.
            If False, the 'class_sample_rate' * total_sample samples is used 
            to build each base model.
            
        class_sample_rate: float, optional (default=0.4)
            Percentage of samples for each class used for training
            
        n_jobs : int or None, optional (default=None)
            The number of jobs to run in parallel for both `fit` and `predict`.
            'None' means 1, '-1' means using all processors.
    
        random_state : int, RandomState instance or None, optional (default=None)
            If int, random_state is the seed used by the random number generator;
            If RandomState instance, random_state is the random number generator;
            If None, the random number generator is the RandomState instance used
            by `np.random`.
            
        gamma: double, optional (default = 1)
            The speed of decreasing of membership function
            
        isNorm: boolean, optional (default=False)
            + True: normalize the input data, False: non-normalize the input data
            
        norm_range: a list with two elements, optional (default=[0, 1])
            The range to normalize the input data
    """
    
    def __init__(self,
                 n_estimators=100,
                 theta=0.2,
                 higher_level_theta=0.2,
                 bootstrap_sample=False,
                 class_sample_rate=0.4,
                 n_jobs=None,
                 gamma = 1,
                 random_state=None,
                 isNorm = False,
                 norm_range = [0, 1]):
        
        super(ModelLevelCombination, self).__init__(n_estimators=n_estimators, theta=theta, bootstrap_sample = bootstrap_sample, gamma=gamma, n_jobs=n_jobs, random_state=random_state, isNorm=isNorm, norm_range=norm_range)
        self.higher_level_theta = higher_level_theta
        
    #! public methods
    
    def fit_base_learner(self, X_l, X_u, patClassId=None, base_learer_type='iol-gfmm'):
        super(ModelLevelCombination, self).fit(X_l, X_u, patClassId, base_learer_type)
    
    def predict_voting_before_model_combination(self, Xl_Test, Xu_Test, patClassIdTest):
        accuracy, num_wrong_samples, predicted_classes = super(ModelLevelCombination, self).predict_voting(Xl_Test, Xu_Test, patClassIdTest)
        
        return (accuracy, num_wrong_samples, predicted_classes)
    
    def predict_based_mem_before_model_combination(self, Xl_Test, Xu_Test, patClassIdTest):
        accuracy, num_wrong_samples, predicted_classes = super(ModelLevelCombination, self).predict_based_mem(Xl_Test, Xu_Test, patClassIdTest)
        
        return (accuracy, num_wrong_samples, predicted_classes)
    
    def fit_model_combination(self, X_l, X_u, patClassId=None, base_learer_type='iol-gfmm'):
        """
        Training the classifier
        
         Xl             Input data lower bounds (rows = objects, columns = features)
         Xu             Input data upper bounds (rows = objects, columns = features)
         patClassId     Input data class labels (crisp). patClassId[i] = 0 corresponds to an unlabeled item
                        If patClassId = None => the last column of both X_l and X_u contains classes of training samples
        
         base_learer    Selected base learner
                         + 'iol-gfmm': GFMM using improved online learning
                         + 'agglo2': GFMM using Agglomerative learning v2
                         + otherwise: GFMM using Manhattan distance
        """
        
        # combine base learners stored in self.list_learners
        # results: V, W, classId, counter, sub_selected_feature
        self.select_learner = base_learer_type
        num_estimators = len(self.list_learners)
        self.V = []
        self.W = []
        self.classId = []
        
        if base_learer_type == 'iol-gfmm':
            self.counter = []
        
        for i in range(num_estimators):
            self.V.extend(self.list_learners[i].V)
            self.W.extend(self.list_learners[i].W)
            self.classId.extend(self.list_learners[i].classId)
            if base_learer_type == 'iol-gfmm':
                self.counter.extend(self.list_learners[i].counter)
            
        #convert to numpy array
        self.classId = np.array(self.classId)
        self.counter = np.array(self.counter)
        self.V = np.array(self.V)
        self.W = np.array(self.W)
        
        # relearning to merge hyperboxes
        if base_learer_type == 'iol-gfmm':
            classifier = ImprovedOnlineGFMM(gamma=self.gamma, teta=self.higher_level_theta, sigma = 0, isDraw = False, isNorm=False)
            classifier.fit(self.V, self.W, self.classId, self.counter)
        elif base_learer_type == 'agglo2':
            classifier = AccelBatchGFMM(self.gamma, self.higher_level_theta, bthres = 0, simil = 'long', sing = 'max', isDraw = False, isNorm = False)
            classifier.fit(self.V, self.W, self.classId)
        else:
            classifier = OnlineGFMM(gamma = self.gamma, teta=self.higher_level_theta, tMin=self.higher_level_theta, isDraw = False, isNorm=False)            
            classifier.fit(self.V, self.W, self.classId)
         
        # save results
        self.V = classifier.V
        self.W = classifier.W
        self.classId = classifier.classId
        if base_learer_type == 'iol-gfmm':
            self.counter = classifier.counter
        
    def predict_model_combination(self, Xl_Test, Xu_Test, patClassIdTest):
        if len(self.mins) > 0:
            noSamples = Xl_Test.shape[0]
            Xl_Test = self.loLim + (self.hiLim - self.loLim) * (Xl_Test - np.ones((noSamples, 1)) * self.mins) / (np.ones((noSamples, 1)) * (self.maxs - self.mins))
            Xu_Test = self.loLim + (self.hiLim - self.loLim) * (Xu_Test - np.ones((noSamples, 1)) * self.mins) / (np.ones((noSamples, 1)) * (self.maxs - self.mins))

            if Xl_Test.min() < self.loLim or Xu_Test.min() < self.loLim or Xl_Test.max() > self.hiLim or Xu_Test.max() > self.hiLim:
                print('Test sample falls outside', self.loLim, '-', self.hiLim, 'interval')
                print('Number of original samples = ', noSamples)

                # only keep samples within the interval loLim-hiLim
                indXl_good = np.where((Xl_Test >= self.loLim).all(axis = 1) & (Xl_Test <= self.hiLim).all(axis = 1))[0]
                indXu_good = np.where((Xu_Test >= self.loLim).all(axis = 1) & (Xu_Test <= self.hiLim).all(axis = 1))[0]
                indKeep = np.intersect1d(indXl_good, indXu_good)

                Xl_Test = Xl_Test[indKeep, :]
                Xu_Test = Xu_Test[indKeep, :]

                print('Number of kept samples =', Xl_Test.shape[0])
                #return

        # do classification
        result = None

        if Xl_Test.shape[0] > 0:
            if self.select_learner == 'iol-gfmm':
                result = predict_with_probability(self.V, self.W, self.classId, self.counter, Xl_Test, Xu_Test, patClassIdTest, self.gamma)
            else:
                result = predict_with_manhattan(self.V, self.W, self.classId, Xl_Test, Xu_Test, patClassIdTest, self.gamma)

        return result
        
    
if __name__ == '__main__':
    """
    INPUT parameters from command line
    
    arg1: + 1 - training and testing datasets are located in separated files
          + 2 - training and testing datasets are located in the same files
    arg2: path to file containing the training dataset (arg1 = 1) or both training and testing datasets (arg1 = 2)
    arg3: + path to file containing the testing dataset (arg1 = 1)
          + percentage of the training dataset in the input file
    arg4: + path to file containing the validation dataset
    arg5: + Maximum size of hyperboxes (theta, default: 0.2)
    arg6: + Number of estimators (default: 100)
    arg7: boolean, optional (default=True)
        Whether bootstrap samples are used when building hyperbox-based models.
        If False, the 'class_sample_rate' * total_sample samples is used 
        to build each base model.
            
    arg8: float, optional (default=0.5)
        Percentage of samples for each class used for training
            
    arg9: int or None, optional (default=None)
        The number of jobs to run in parallel for both `fit` and `predict`.
        'None' means 1, '-1' means using all processors.
        
    arg10: Selected algorithms for each base learner:
        + 'iol-gfmm': GFMM using improved online learning
        + otherwise: GFMM using Manhattan distance
    arg11: + gamma value (default: 1)
    arg12: + Minimum accuracy of each hyperbox used for the pruning process
    arg13: + do normalization of datasets or not? True: Normilize, False: No normalize (default: False)
    arg14: + range of input values after normalization (default: [0, 1])
    """
    # Init default parameters
#    if len(sys.argv) < 6:
#        theta = 0.2
#    else:
#        theta = float(sys.argv[5])
#
#    if len(sys.argv) < 7:
#        n_estimators = 100
#    else:
#        n_estimators = int(sys.argv[6])
#
#    if len(sys.argv) < 8:
#        bootstrap_sample = True
#    else:
#        bootstrap_sample = string_to_boolean(sys.argv[7])
#
#    if len(sys.argv) < 9:
#        class_sample_rate = 0.5
#    else:
#        class_sample_rate = float(sys.argv[8])
#
#    if len(sys.argv) < 10:
#        n_jobs = 1
#    else:
#        try:
#            n_jobs = int(sys.argv[9])
#        except:
#            n_jobs = 1
#           
#    if len(sys.argv) < 11:
#        selected_alg = 'iol-gfmm'
#    else:
#        selected_alg = sys.argv[10]
#  
#    if len(sys.argv) < 12:
#        gamma = 1
#    else:
#        gamma = float(sys.argv[11])
    
#    if len(sys.argv) < 13:
#        min_accuracy_pruning = 0.5
#    else:
#        min_accuracy_pruning = float(sys.argv[12])
        
#    if len(sys.argv) < 14:
#        isNorm = True
#    else:
#        isNorm = string_to_boolean(sys.argv[13])
#
#    if len(sys.argv) < 15:
#        norm_range = [0, 1]
#    else:
#        norm_range = ast.literal_eval(sys.argv[14])
#
#    start_t = time.perf_counter()
#    if sys.argv[1] == '1':
#        training_file = sys.argv[2]
#        testing_file = sys.argv[3]
#
#        # Read training file
#        Xtr, X_tmp, patClassIdTr, pat_tmp = loadDataset(training_file, 1, False)
#        # Read testing file
#        X_tmp, Xtest, pat_tmp, patClassIdTest = loadDataset(testing_file, 0, False)
#
#    else:
#        dataset_file = sys.argv[2]
#        percent_Training = float(sys.argv[3])
#        Xtr, Xtest, patClassIdTr, patClassIdTest = loadDataset(dataset_file, percent_Training, False)
#    
#    validation_file = sys.argv[4]
#    
#    if (not validation_file) == True:
#        # empty validation file
#        print('no pruning')
#        isPruning = False
#    else:
#        print('pruning')
#        isPruning = True
#        Xval, _, patClassIdVal, _ = loadDataset(validation_file, 1, False)  
      
    
    isPruning = False
    training_file = "C:\\Hyperbox-based-ML\\Dataset\\train_test\\training_testing_data\\balance_scale_dps_test.dat"
    testing_file = "C:\\Hyperbox-based-ML\\Dataset\\train_test\\training_testing_data\\balance_scale_dps_test.dat"
    validation_file = "C:\\Hyperbox-based-ML\\Dataset\\train_test\\training_testing_data\\balance_scale_dps_val.dat"
    gamma = 1
    theta = 0.5
    isNorm = False
    norm_range = [0, 1]
    n_estimators=5
    bootstrap_sample=False
    class_sample_rate=1
    n_jobs=1
    random_state=None
    selected_alg = 'iol-gfmm'
    gamma = 1
    # Read training file
    Xtr, _, patClassIdTr, _ = loadDataset(training_file, 1, False)
    # Read testing file
    _, Xtest, _, patClassIdTest = loadDataset(testing_file, 0, False)
    
    # Read validation file
    Xval, _, patClassIdVal, _ = loadDataset(validation_file, 1, False)
    
    classifier = ModelLevelCombination(n_estimators=n_estimators, theta=theta, bootstrap_sample=bootstrap_sample, class_sample_rate=class_sample_rate, n_jobs=n_jobs, random_state=None, gamma = gamma)
    classifier.fit(Xtr, Xtr, patClassIdTr, selected_alg)
    
    # Testing
    print("-- Before pruning --")
    print("Number of hyperboxes = ", classifier.get_complexity())
    accuracy, num_wrong_samples, predicted_class = classifier.predict_test(Xtest, Xtest, patClassIdTest)
    print("Testing error = ", 100 - 100*accuracy_score(patClassIdTest, predicted_class))
    print(classification_report(patClassIdTest, predicted_class))
    
    # Do pruning
#    classifier.pruning_val(Xval, Xval, patClassIdVal)
#    
#    # Testing after pruning
#    print("-- After pruning --")
#    print("Number of hyperboxes = ", classifier.get_complexity())
#    accuracy, num_wrong_samples, predicted_class = classifier.predict_test(Xtest, Xtest, patClassIdTest)
#    print("Testing error = ", 100 - 100*accuracy_score(patClassIdTest, predicted_class))
#    print(classification_report(patClassIdTest, predicted_class))
#    
        