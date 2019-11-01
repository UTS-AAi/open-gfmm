# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 14:09:02 2019

@author: Thanh Tung Khuat

This is the implementation of random hyperboxes and superboxes

"""
import sys, os
sys.path.insert(0, os.path.pardir)

import numpy as np
import math
import random
import ast
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from sklearn.utils import resample
from functionhelper.preprocessinghelper import normalize
from randombox.improvedonlgfmm_baselearner import IOL_GFMM_BaseLearner
from randombox.onlgfmm_baselearner import GFMM_BaseLearner
from functionhelper.preprocessinghelper import loadDataset, string_to_boolean
from randombox.utils import find_max_mode
from sklearn.metrics import accuracy_score, classification_report

def get_num_cpu_cores():
    num_cores = multiprocessing.cpu_count()
    return num_cores

class RandomHyperboxClassifier(object):
    """
    A random hyperbox-based classifier.
        
    A random hyperboxes classifier is a meta estimator that fits a number of hyperbox-based
    classifiers (i.e., variants of GFMM) on various sub-samples and sub-features 
    of the dataset and uses averaging to improve the predictive accuracy and control over-fitting.
    The sub-sample size is always the same as the original
    input sample size but the samples are drawn with replacement if
    `bootstrap_sample=True` (default).
    
    Parameters:
    ----------
        n_estimators : integer, optional (default=100)
            The number of classifiers in the model.
    
        theta : float, optional (default = 0.2)
            The maximum hyperbox size of each base model
            
        max_features : int, float, string or None, optional (default="auto")
            The maximum number of features to consider when building each base model:
            - If int, then consider 'max_features' features.
            - If float, then 'max_features' is a fraction and
              `int(max_features * n_features)` features are considered.
            - If "auto", then 'max_features=sqrt(n_features)'.
            - If "sqrt", then 'max_features=sqrt(n_features)' (same as "auto").
            - If "log2", then 'max_features=log2(n_features)'.
            - If None, then `max_features=n_features`.
            
        bootstrap_sample : boolean, optional (default=True)
            Whether bootstrap samples are used when building hyperbox-based models.
            If False, the 'class_sample_rate' * total_sample samples is used 
            to build each base model.
            
        class_sample_rate: float, optional (default=0.4)
            Percentage of samples for each class used for training
            
        bootstrap_feature : boolean, optional (default=True)
            Whether bootstrap features are used when building hyperbox-based models.
            If False, a uniform random number (num_features) in the rage of [1, max_features] is generated,
            and then the algorithm choices num_features features randomly to build each base model.
        
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
                 max_features='auto',
                 bootstrap_sample=True,
                 bootstrap_feature=True,
                 class_sample_rate=0.4,
                 n_jobs=None,
                 gamma = 1,
                 random_state=None,
                 isNorm = False,
                 norm_range = [0, 1]):
        
        self.n_estimators = n_estimators
        self.theta = theta
        self.max_features = max_features
        self.bootstrap_sample = bootstrap_sample
        self.bootstrap_feature = bootstrap_feature
        self.gamma = gamma
        
        if n_jobs is None:
            self.n_jobs = 1
        elif n_jobs == -1:
            self.n_jobs = get_num_cpu_cores()
        else:
            self.n_jobs = n_jobs
        
        self.random_state = random_state
        self.class_sample_rate = class_sample_rate
        self.isNorm = isNorm
        self.loLim = norm_range[0]
        self.hiLim = norm_range[1]
        self.mins = []
        self.maxs = []
     
    def _dataPreprocessing(self, X_l, X_u):
        """
        Preprocess data: delete constant dimensions, Normalize input samples if needed

        INPUT:
            X_l          Input data lower bounds (rows = objects, columns = features)
            X_u          Input data upper bounds (rows = objects, columns = features)

        OUTPUT
            X_l, X_u were preprocessed
        """

        # delete constant dimensions
        #X_l, X_u = delete_const_dims(X_l, X_u)

        # Normalize input samples if needed
        if X_l.min() < self.loLim or X_u.min() < self.loLim or X_u.max() > self.hiLim or X_l.max() > self.hiLim:
            self.mins = X_l.min(axis = 0) # get min value of each feature
            self.maxs = X_u.max(axis = 0) # get max value of each feature
            X_l = normalize(X_l, [self.loLim, self.hiLim])
            X_u = normalize(X_u, [self.loLim, self.hiLim])
        else:
            self.isNorm = False
            self.mins = []
            self.maxs = []

        return (X_l, X_u)
    
    def _get_random_sample(self, inputData, inputClass=None, random_state=None):
        """
        Get a random subsample for training data using class_sample_rate for each class
        Parameters:
            inputData: numpy array
                Contain features and/or label of input data, the label is located in the last column
            inputClass: numpy array, optional (default=None)
                Contain classes of input samples
        Return:
            (Xtr, Clstr):
                Xtr: training set with features
                Clstr: label of training samples
                selected_id: Indices of selected samples
        """
        if inputClass is not None:
            inputData = np.hstack((inputData, inputClass.reshape(-1, 1)))
        
        labels = inputData[:, -1]
        class_lb = np.unique(labels)
        
        selected_id = []
        for i in range(len(class_lb)):
            if isinstance(random_state, int):
                random_state_i = random_state + i
            if random_state is None:
                random_state_i = None
            id_class_i = np.nonzero(labels == class_lb[i])[0]
            num_sample_class_i = len(id_class_i)
            num_ele = int(self.class_sample_rate * num_sample_class_i + 0.5)
            selected_idx_cls_i = resample(range(num_sample_class_i), replace=False, n_samples=num_ele, random_state=random_state_i)
            selected_id.extend(id_class_i[selected_idx_cls_i])
        
        np.random.shuffle(selected_id)
        
        Xtr = inputData[selected_id, 0:-1]
        Clstr = inputData[selected_id, -1]
        
        return (Xtr, Clstr, selected_id)
    
    def _get_bootstrap_sample(self, inputData, inputClass=None, random_state=None):
        """
        Get a bootstrap sample with the same size as inputData to build training data
        Parameters:
            inputData: numpy matrix
                Contain features and/or label of input data, the label is located in the last column
            inputClass: numpy array, optional (default=None)
                Contain classes of input samples
        Return:
            (Xtr, Clstr):
                Xtr: training set with features
                Clstr: label of training samples
                idSubsample: Indices of selected samples
        """
        n_samples = inputData.shape[0]
        
        if inputClass is not None:
            inputData = np.hstack((inputData, inputClass.reshape(-1, 1)))
        
        idSubsample = resample(range(n_samples), replace=True, n_samples=n_samples, random_state=random_state)
        
        Xtr = inputData[idSubsample, 0:-1]
        Clstr = inputData[idSubsample, -1]
        
        return (Xtr, Clstr, idSubsample)
    
        
    def _set_max_feature(self, n_features):
        """
        Set the value for self.max_features
            Parameters:
                n_features: The number of features of input data
        """
        if (self.max_features is 'auto') or (self.max_features is 'sqrt'):
            self.max_features = int(math.sqrt(n_features))
        elif self.max_features == 'log2':
            self.max_features = int(math.log2(n_features))
        elif self.max_features is None:
            self.max_features = n_features
        elif isinstance(self.max_features, float):
            self.max_features = int(self.max_features * n_features)
            
    def _get_random_subfeature(self, Xtr, random_state=None):
        """
        Get a subset of features to rebuild a training set
        
            Parameters:
                Xtr: numpy array
                    A matrix which each row is a training sample without label
                    
            Return:
                (Xtr_subf, subf)
                Xtr_subf: numpy array
                    A matrix whose columns are a subset of columns in Xtr
                subf: vector
                    A array storing a subset of selected features
        """
        num_features = Xtr.shape[1]
        subf = resample(range(num_features), replace=self.bootstrap_feature, n_samples=self.max_features, random_state=random_state)
        subf = np.unique(subf) # remove duplicated indices and sort in ascending
        
        Xtr_subf = Xtr[:, subf]
        
        return (Xtr_subf, subf)
    
    def _get_random_subfeature_uniform(self, Xtr, random_state=None):
        """
        Get a subset of features to rebuild a training set
        
            Parameters:
                Xtr: numpy array
                    A matrix which each row is a training sample without label
                    
            Return:
                (Xtr_subf, subf)
                Xtr_subf: numpy array
                    A matrix whose columns are a subset of columns in Xtr
                subf: vector
                    A array storing a subset of selected features
        """
        num_features = Xtr.shape[1]
        num_selected_features = random.choice(np.array(range(self.max_features)) + 1)
        subf = resample(range(num_features), replace=False, n_samples=num_selected_features, random_state=random_state)
        
        Xtr_subf = Xtr[:, subf]
        
        return (Xtr_subf, subf)
    
    def _slicing_data_using_subselected_features(self, inputData, subf):
        """
        This function is to return a dataset from inputData, in which indices are 
        saved in subf
        
            Parameters:
                inputData: Input data (rows = objects, columns = features)
                subf: indices of selected features
                
            Return:
                sliced_data: inputData were sliced
        """
        return inputData[:, subf]
    
    def _baselearner_building(self, Xl_full, Xu_full, n_learners=1, base_learer_type='iol-gfmm', random_state=None):
        """
        This function is to build base learners
        
            Parameters:
                Xl_full, Xu_full: numpy matrix
                    Lower and upper bounds of training data including labels in the last column
                n_learners: int, optional (default=1)
                    The number of learners is built
                base_learer_type: string, optional (default='iol-gfmm')
                    + 'iol-gfmm': using GFMM with improved learing algorithm
                    + else: using GFMM with Manhattan distance
                    
            Return:
                A list of built classifiers with n_learners elements
        """
        list_learners = []
        for i in range(n_learners):
            if isinstance(random_state, int):
                rd_state = random_state + i
            else:
                rd_state = random_state
            
            if self.bootstrap_sample:
                Xl_tr, clsTr, selected_id = self._get_bootstrap_sample(inputData=Xl_full, random_state=rd_state)
                Xu_tr = Xu_full[selected_id, 0:-1]
            else:
                Xl_tr, clsTr, selected_id = self._get_random_sample(inputData=Xl_full, random_state=rd_state)
                Xu_tr = Xu_full[selected_id, 0:-1]
            
            if self.bootstrap_feature == True:
                Xl_tr, subf = self._get_random_subfeature(Xtr=Xl_tr, random_state=rd_state)
            else:
                Xl_tr, subf = self._get_random_subfeature_uniform(Xtr=Xl_tr, random_state=rd_state)
            
            Xu_tr = self._slicing_data_using_subselected_features(Xu_tr, subf)
            
            if base_learer_type == 'iol-gfmm':
                learner = IOL_GFMM_BaseLearner(gamma=self.gamma, theta=self.theta, sub_selected_feature=subf)
                learner.fit(Xl_tr, Xu_tr, clsTr)
            else:
                learner = GFMM_BaseLearner(gamma = self.gamma, theta=self.theta, theta_min=self.theta, sub_selected_feature=subf)
                learner.fit(Xl_tr, Xu_tr, clsTr)
            list_learners.append(learner)
        
        return list_learners
    
    def _predict_test(self, Xl_Test, Xu_Test, patClassIdTest, lst_learners):
        """
        Predictive results of base learned stored in lst_learners
            
            Parameters:
                lst_learners: save list of base learners
                Xl_Test, Xu_Test: Lower and upper bounds of the testing set
                patClassIdTest: classes of the testing set
        
            Returns:
                Results are save in attribute 'predicted_class' of each element in lst_learners
        """
        n_learners = len(lst_learners)
        
        for i in range(n_learners):
            lst_learners[i].predict(Xl_Test, Xu_Test, patClassIdTest)
            
        return lst_learners
    
    def _pruning_on_val(self, Xl_Val, Xu_Val, patClassIdVal, lst_learners, min_accuracy):
        """
        Perform pruning
            Parameters:
                lst_learners: save list of base learners
                Xl_Val, Xu_Val: Lower and upper bounds of the validation set
                patClassIdVal: classes of the validation set
        """
        n_learners = len(lst_learners)
        
        for i in range(n_learners):
            lst_learners[i].pruning_val(Xl_Val, Xu_Val, patClassIdVal, min_accuracy)
            
        return lst_learners
        
    #! public methods
    
    def fit(self, X_l, X_u, patClassId=None, base_learer_type='iol-gfmm'):
        """
        Training the classifier
        
         Xl             Input data lower bounds (rows = objects, columns = features)
         Xu             Input data upper bounds (rows = objects, columns = features)
         patClassId     Input data class labels (crisp). patClassId[i] = 0 corresponds to an unlabeled item
                        If patClassId = None => the last column of both X_l and X_u contains classes of training samples
        
         base_learer    Selected base learner
                         + 'iol-gfmm': GFMM using improved online learning
                         + otherwise: GFMM using Manhattan distance
        """
        if patClassId is None:
            if self.isNorm:
                patClassId = X_l[:, -1]
                self.patId = X_l[:, -1]
                X_l = X_l[:, 0:-1]
                X_u = X_u[:, 0:-1]
                X_l, X_u = self._dataPreprocessing(X_l, X_u)
                self.X_l = X_l
                self.X_u = X_u
                X_l = np.hstack((X_l, patClassId.reshape(-1, 1)))
                X_u = np.hstack((X_u, patClassId.reshape(-1, 1)))
            else:
                self.X_l = X_l[:, 0:-1]
                self.X_u = X_u[:, 0:-1]
                self.patId = X_l[:, -1]
        else:
            if self.isNorm:
                X_l, X_u = self._dataPreprocessing(X_l, X_u)
            
            self.X_l = X_l
            self.X_u = X_u
            self.patId = patClassId
            # create a training set including labels to use in method _get_random_sample
            X_l = np.hstack((X_l, patClassId.reshape(-1, 1)))
            X_u = np.hstack((X_u, patClassId.reshape(-1, 1)))
        
        
        n_features = X_l.shape[1] - 1 # because we concatenated the classes to features data
        self._set_max_feature(n_features)
       
        #training
        self.list_learners = []
        if self.n_jobs > 1:
            # Parallel execution
            futures = []
            num_est_learners = int(self.n_estimators / self.n_jobs)
            with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
                for i in range(self.n_jobs):
                    if i == self.n_jobs - 1:
                        n_learners = self.n_estimators - (self.n_jobs - 1) * num_est_learners
                    else:
                        n_learners = num_est_learners
                        
                    futures.append(executor.submit(self._baselearner_building, X_l, X_u, n_learners, base_learer_type, self.random_state))
                            
                # Instruct workers to process results as they come, when all are completed
                as_completed(futures) # wait all workers completed:
                for future in futures:
                    self.list_learners.extend(future.result())
        else:
            # Sequential base learner building
            self.list_learners = self._baselearner_building(X_l, X_u, self.n_estimators, base_learer_type)
    
    def predict_test(self, Xl_Test, Xu_Test, patClassIdTest):
        """
        This function is to evaluate the performance of the model
            
            Parameters:
                + Xl_Test, Xu_Test: Lower and upper bounds of the testing set
                + patClassIdTest: classes of the testing set
        
            Returns:
                + Predictive results are saved into attribute 'predicted_class' of each element in self.list_learners
                + Accuracy
                + Number of wrong predicted samples
        """
        if self.n_jobs > 1:
            # Parallel execution
            futures = []
            n_learner_each_job = int(self.n_estimators / self.n_jobs)
            with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
                for i in range(self.n_jobs):
                    start_pos = n_learner_each_job * i
                    if i == self.n_jobs - 1:
                        end_pos = None
                    else:
                        end_pos = n_learner_each_job * (i + 1)
                    futures.append(executor.submit(self._predict_test, Xl_Test, Xu_Test, patClassIdTest, self.list_learners[start_pos:end_pos]))
                    
                as_completed(futures) # wait all workers completed
            
            self.list_learners = []
            for future in futures:
                self.list_learners.extend(future.result())
                
        else:
            self._predict_test(Xl_Test, Xu_Test, patClassIdTest, self.list_learners)
            
        # report result using voting
        n_samples = Xl_Test.shape[0]
        predicted_classes = []
        for i in range(n_samples):
            cls_i = [self.list_learners[j].predicted_class[i] for j in range(self.n_estimators)]
            predicted_classes.append(find_max_mode(cls_i))
            
        num_correct_samples = np.sum(predicted_classes == patClassIdTest)
        num_wrong_samples = n_samples - num_correct_samples
        accuracy = num_correct_samples / n_samples
        
        return (accuracy, num_wrong_samples, predicted_classes)
    
    def pruning_val(self, XlVal, XuVal, patClassIdVal, accuracy_threshold = 0.5):
        """
        pruning handling based on validation (validation routine) with hyperboxes stored in self. V, W, classId
    
          result = pruning_val(XlT,XuT,patClassIdTest)
    
            INPUT
              XlT               Test data lower bounds (rows = objects, columns = features)
              XuT               Test data upper bounds (rows = objects, columns = features)
              patClassIdTest    Test data class labels (crisp)
              accuracy_threshold  The minimum accuracy for each hyperbox
        """
        if self.n_jobs > 1:
            # Parallel execution
            futures = []
            n_learner_each_job = int(self.n_estimators / self.n_jobs)
            with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
                for i in range(self.n_jobs):
                    start_pos = n_learner_each_job * i
                    if i == self.n_jobs - 1:
                        end_pos = None
                    else:
                        end_pos = n_learner_each_job * (i + 1)
                    futures.append(executor.submit(self._pruning_on_val, XlVal, XuVal, patClassIdVal, self.list_learners[start_pos:end_pos], accuracy_threshold))
                    
                as_completed(futures) # wait all workers completed
            
            self.list_learners = []
            for future in futures:
                self.list_learners.extend(future.result())
                
        else:
            self._pruning_on_val(XlVal, XuVal, patClassIdVal, self.list_learners, accuracy_threshold)
         
    def get_complexity(self):
        """
        Get total numbers of hyperboxes in all base learners
        """
        n_boxes = 0
        for i in range(len(self.list_learners)):
            n_boxes = n_boxes + len(self.list_learners[i].classId)
            
        return n_boxes
    
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
    arg7: + Maximum number of features (max_features) considered in each learner (default="auto")
            - If int, then consider 'max_features' features.
            - If float, then 'max_features' is a fraction and
              `int(max_features * n_features)` features are considered.
            - If "auto", then 'max_features=sqrt(n_features)'.
            - If "sqrt", then 'max_features=sqrt(n_features)' (same as "auto").
            - If "log2", then 'max_features=log2(n_features)'.
            - If None, then `max_features=n_features`.
    arg8: boolean, optional (default=True)
        Whether bootstrap samples are used when building hyperbox-based models.
        If False, the 'class_sample_rate' * total_sample samples is used 
        to build each base model.
            
    arg9: float, optional (default=0.5)
        Percentage of samples for each class used for training
            
    arg10: boolean, optional (default=True)
        Whether bootstrap features are used when building hyperbox-based models.
        If False, the whole 'max_features' features is used to build each base model.
    
    arg11: int or None, optional (default=None)
        The number of jobs to run in parallel for both `fit` and `predict`.
        'None' means 1, '-1' means using all processors.
        
    arg12: Selected algorithms for each base learner:
        + 'iol-gfmm': GFMM using improved online learning
        + otherwise: GFMM using Manhattan distance
    arg13: + gamma value (default: 1)
    arg14: + Minimum accuracy of each hyperbox used for the pruning process
    arg15: + do normalization of datasets or not? True: Normilize, False: No normalize (default: False)
    arg16: + range of input values after normalization (default: [0, 1])
    """
    # Init default parameters
    if len(sys.argv) < 6:
        theta = 0.2
    else:
        theta = float(sys.argv[5])

    if len(sys.argv) < 7:
        n_estimators = 100
    else:
        n_estimators = int(sys.argv[6])

    if len(sys.argv) < 8:
        max_features = "auto"
    else:
        try:
            max_features = int(sys.argv[7])
        except:
            try:
                max_features = float(sys.argv[7])
            except:
                max_features = sys.argv[7]
    
    if len(sys.argv) < 9:
        bootstrap_sample = True
    else:
        bootstrap_sample = string_to_boolean(sys.argv[8])

    if len(sys.argv) < 10:
        class_sample_rate = 0.5
    else:
        class_sample_rate = float(sys.argv[9])

    if len(sys.argv) < 11:
        bootstrap_feature = True
    else:
        bootstrap_feature = string_to_boolean(sys.argv[10])
        
    if len(sys.argv) < 12:
        n_jobs = 1
    else:
        try:
            n_jobs = int(sys.argv[11])
        except:
            n_jobs = 1
           
    if len(sys.argv) < 13:
        selected_alg = 'iol-gfmm'
    else:
        selected_alg = sys.argv[12]
  
    if len(sys.argv) < 14:
        gamma = 1
    else:
        gamma = float(sys.argv[13])
    
    if len(sys.argv) < 15:
        min_accuracy_pruning = 0.5
    else:
        min_accuracy_pruning = float(sys.argv[14])
        
    if len(sys.argv) < 16:
        isNorm = True
    else:
        isNorm = string_to_boolean(sys.argv[15])

    if len(sys.argv) < 17:
        norm_range = [0, 1]
    else:
        norm_range = ast.literal_eval(sys.argv[16])

    # print('isDraw = ', isDraw, ' teta = ', teta, ' teta_min = ', teta_min, ' gamma = ', gamma, ' oper = ', oper, ' isNorm = ', isNorm, ' norm_range = ', norm_range)
    if sys.argv[1] == '1':
        training_file = sys.argv[2]
        testing_file = sys.argv[3]

        # Read training file
        Xtr, X_tmp, patClassIdTr, pat_tmp = loadDataset(training_file, 1, False)
        # Read testing file
        X_tmp, Xtest, pat_tmp, patClassIdTest = loadDataset(testing_file, 0, False)

    else:
        dataset_file = sys.argv[2]
        percent_Training = float(sys.argv[3])
        Xtr, Xtest, patClassIdTr, patClassIdTest = loadDataset(dataset_file, percent_Training, False)
    
    validation_file = sys.argv[4]
    
    if (not validation_file) == True:
        # empty validation file
        print('no pruning')
        isPruning = False
    else:
        print('pruning')
        isPruning = True
        Xval, _, patClassIdVal, _ = loadDataset(validation_file, 1, False)  
      
    
#    isPruning = False
#    training_file = "C:\\Hyperbox-based-ML\\Dataset\\train_test\\training_testing_data\\balance_scale_dps_test.dat"
#    testing_file = "C:\\Hyperbox-based-ML\\Dataset\\train_test\\training_testing_data\\balance_scale_dps_test.dat"
#    validation_file = "C:\\Hyperbox-based-ML\\Dataset\\train_test\\training_testing_data\\balance_scale_dps_val.dat"
#    gamma = 1
#    theta = 0.5
#    isNorm = False
#    norm_range = [0, 1]
#    n_estimators=5
#    max_features=4
#    bootstrap_sample=False
#    bootstrap_feature=False
#    class_sample_rate=1
#    n_jobs=1
#    random_state=None
#    selected_alg = 'iol-gfmm'
#    gamma = 1
#    # Read training file
#    Xtr, _, patClassIdTr, _ = loadDataset(training_file, 1, False)
#    # Read testing file
#    _, Xtest, _, patClassIdTest = loadDataset(testing_file, 0, False)
#    
#    # Read validation file
#    Xval, _, patClassIdVal, _ = loadDataset(validation_file, 1, False)
    
    classifier = RandomHyperboxClassifier(n_estimators=n_estimators, theta=theta, max_features=max_features, bootstrap_sample=bootstrap_sample, bootstrap_feature=bootstrap_feature, class_sample_rate=class_sample_rate, n_jobs=n_jobs, random_state=None, gamma = gamma)
    classifier.fit(Xtr, Xtr, patClassIdTr, selected_alg)
    
    # Testing
    print("-- Before pruning --")
    print("Number of hyperboxes = ", classifier.get_complexity())
    accuracy, num_wrong_samples, predicted_class = classifier.predict_test(Xtest, Xtest, patClassIdTest)
    print("Testing error = ", 100 - 100*accuracy_score(patClassIdTest, predicted_class))
    print(classification_report(patClassIdTest, predicted_class))
    
    # Do pruning
    if isPruning == True:
        classifier.pruning_val(Xval, Xval, patClassIdVal)
        
        # Testing after pruning
        print("-- After pruning --")
        print("Number of hyperboxes = ", classifier.get_complexity())
        accuracy, num_wrong_samples, predicted_class = classifier.predict_test(Xtest, Xtest, patClassIdTest)
        print("Testing error = ", 100 - 100*accuracy_score(patClassIdTest, predicted_class))
        print(classification_report(patClassIdTest, predicted_class))