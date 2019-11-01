# -*- coding: utf-8 -*-
"""
Created on Fri Sep 21 21:33:43 2018

@author: Thanh Tung Khuat

Implementation of the refined fuzzy min-max neural network

            RFMNNClassification(gamma, teta, isDraw, isNorm, norm_range)

    INPUT
         V              Hyperbox lower bounds for the model to be updated using new data
         W              Hyperbox upper bounds for the model to be updated using new data
         classId        Hyperbox class labels (crisp)  for the model to be updated using new data
         gamma          Membership function slope (default: 1), datatype: array or scalar
         teta           Maximum hyperbox size (default: 1)
         isDraw         Progress plot flag (default: False)
         isNorm         Do normalization of input training samples or not?
         norm_range     New ranging of input data after normalization
"""

import sys, os
sys.path.insert(0, os.path.pardir) 

import ast
import numpy as np
import time
import matplotlib
matplotlib.use('TkAgg')

from functionhelper.membershipcalc import simpsonMembership
from functionhelper.hyperboxadjustment import is_overlap_general_formulas, hyperbox_contraction_rfmm
from functionhelper.preprocessinghelper import loadDataset, string_to_boolean
from functionhelper.basefmnnclassifier import BaseFMNNClassifier
from functionhelper.baseclassification import predict_rfmm_distance 

class RFMNNClassification(BaseFMNNClassifier):
    
    def __init__(self, gamma = 1, teta = 1, isNorm = False, norm_range = [0, 1], V = np.array([], dtype=np.float64), W = np.array([], dtype=np.float64), classId = np.array([], dtype=np.int16)):
        BaseFMNNClassifier.__init__(self, gamma, teta, False, isNorm, norm_range)
        
        self.V = V
        self.W = W
        self.classId = classId
        
    
    def fit(self, Xh, patClassId):
        """
        Training the classifier
        
         Xh             Input data (rows = objects, columns = features)
         patClassId     Input data class labels (crisp). patClassId[i] = 0 corresponds to an unlabeled item
        
        """
        if self.isNorm == True:
            Xh = self.dataPreprocessing(Xh)
        
        time_start = time.clock()
        
        yX, xX = Xh.shape
        
        # for each input sample
        for i in range(yX):
            classOfX = patClassId[i]
            
            if self.V.size == 0:   # no model provided - starting from scratch
                self.V = np.array([Xh[0]])
                self.W = np.array([Xh[0]])
                self.classId = np.array([patClassId[0]])
                
            else:
                idSameClassOfX = np.nonzero(self.classId == classOfX)[0]
                idDifClassOfX = np.nonzero(self.classId != classOfX)[0]
                # Find all hyperboxes same class with indexOfX
                V_same = self.V[idSameClassOfX]
                
                V_dif = self.V[idDifClassOfX]
                W_dif = self.W[idDifClassOfX]
                
                isCreateNewBox = False
                if len(V_same) > 0:
                    W_same = self.W[idSameClassOfX]
                    
                    b = simpsonMembership(Xh[i], V_same, W_same, self.gamma)
                    
                    max_mem_id = np.argmax(b)
                    # store the index of the winner hyperbox in the list of all hyperboxes of all classes
                    j = idSameClassOfX[max_mem_id]
                
                    if b[max_mem_id] != 1:
                        adjust = False
                
                        # test violation of max hyperbox size and class labels
                        V_cmp = np.minimum(self.V[j], Xh[i])
                        W_cmp = np.maximum(self.W[j], Xh[i])
                        if ((W_cmp - V_cmp) <= self.teta).all() == True:
                            if is_overlap_general_formulas(V_dif, W_dif, V_cmp, W_cmp, False) == False:
                                # adjust the j-th hyperbox
                                self.V[j] = V_cmp
                                self.W[j] = W_cmp
                                adjust = True
                            
                        # if i-th sample did not fit into any existing box, create a new one
                        if not adjust:
                            self.V = np.vstack((self.V, Xh[i]))
                            self.W = np.vstack((self.W, Xh[i]))
                            self.classId = np.append(self.classId, classOfX)
                            isCreateNewBox = True     
                else:
                    # create a new hyperbox
                    self.V = np.vstack((self.V, Xh[i]))
                    self.W = np.vstack((self.W, Xh[i]))
                    self.classId = np.append(self.classId, classOfX)
                    isCreateNewBox = True
                    
                if isCreateNewBox == True and len(V_dif) > 0:
                    is_ovl, hyperbox_ids_overlap, min_overlap_dimensions = is_overlap_general_formulas(V_dif, W_dif, self.V[-1], self.W[-1], True)
                    if is_ovl == True:
                        # convert hyperbox_ids_overlap of hyperboxes with other classes to ids of all existing hyperboxes
                        hyperbox_ids_overlap = idDifClassOfX[hyperbox_ids_overlap]
                        # do contraction for parent hyperboxes with indices contained in hyperbox_ids_overlap
                        self.V, self.W, self.classId = hyperbox_contraction_rfmm(self.V, self.W, self.classId, hyperbox_ids_overlap, -1, min_overlap_dimensions)
                            
        time_end = time.clock()
        self.elapsed_training_time = time_end - time_start
  						
        return self
    
    def predict(self, X_Test, patClassIdTest):
        """
        Perform classification
        
            result = predict(Xl_Test, Xu_Test, patClassIdTest)
        
        INPUT:
            X_Test             Test data (rows = objects, columns = features)
            patClassIdTest	     Test data class labels (crisp)
            
        OUTPUT:
            result        A object with Bunch datatype containing all results as follows:
                          + summis           Number of misclassified objects
                          + misclass         Binary error map
                          + sumamb           Number of objects with maximum membership in more than one class
                          + out              Soft class memberships
                          + mem              Hyperbox memberships
        """
        #X_Test = delete_const_dims(X_Test)
        # Normalize testing dataset if training datasets were normalized
        if len(self.mins) > 0:
            noSamples = X_Test.shape[0]
            X_Test = self.loLim + (self.hiLim - self.loLim) * (X_Test - np.ones((noSamples, 1)) * self.mins) / (np.ones((noSamples, 1)) * (self.maxs - self.mins))          
            
            if X_Test.min() < self.loLim or X_Test.max() > self.hiLim:
                print('Test sample falls outside', self.loLim, '-', self.hiLim, 'interval')
                print('Number of original samples = ', noSamples)
                
                # only keep samples within the interval loLim-hiLim
                indX_Keep = np.where((X_Test >= self.loLim).all(axis = 1) & (X_Test <= self.hiLim).all(axis = 1))[0]
                
                X_Test = X_Test[indX_Keep, :]
                
                print('Number of kept samples =', X_Test.shape[0])
            
        # do classification
        result = None
        
        if X_Test.shape[0] > 0:
            result = predict_rfmm_distance(self.V, self.W, self.classId, X_Test, patClassIdTest, self.gamma)
        
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
    arg5: + Maximum size of hyperboxes (teta, default: 1)
    arg6: + gamma value (default: 1)
    arg7: + do normalization of datasets or not? True: Normilize, False: No normalize (default: True)
    arg8: + range of input values after normalization (default: [0, 1])   
    """
    # Init default parameters
    if len(sys.argv) < 6:
        teta = 1    
    else:
        teta = float(sys.argv[5])
    
    if len(sys.argv) < 7:
        gamma = 1
    else:
        gamma = float(sys.argv[6])

    if len(sys.argv) < 8:
        isNorm = True
    else:
        isNorm = string_to_boolean(sys.argv[7])
    
    if len(sys.argv) < 9:
        norm_range = [0, 1]
    else:
        norm_range = ast.literal_eval(sys.argv[8])
    
#    # print('isDraw = ', isDraw, ' teta = ', teta, ' teta_min = ', teta_min, ' gamma = ', gamma, ' oper = ', oper, ' isNorm = ', isNorm, ' norm_range = ', norm_range)
    start_t = time.perf_counter()
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
#    training_file = "C:\\Hyperbox-based-ML\\Dataset\\train_test\\training_testing_data\\spambase_dps_tr.dat"
#    testing_file = "C:\\Hyperbox-based-ML\\Dataset\\train_test\\training_testing_data\\spambase_dps_test.dat"
#    gamma = 1
#    teta = 0.1
#    isDraw = False
#    oper = 'min'
#    isNorm = False
#    norm_range = [0, 1]
#    # Read training file
#    Xtr, X_tmp, patClassIdTr, pat_tmp = loadDataset(training_file, 1, False)
#    # Read testing file
#    X_tmp, Xtest, pat_tmp, patClassIdTest = loadDataset(testing_file, 0, False)
#   
    
    classifier = RFMNNClassification(gamma, teta, isNorm, norm_range)
    classifier.fit(Xtr, patClassIdTr)
    
    print("Before pruning:")
    print('No boxes = ', len(classifier.classId))
    if isPruning == True:
        classifier.pruning_val(Xval, patClassIdVal)
        print("After pruning:")
        print('Final hyperbox No =', len(classifier.classId))
    
    end_t = time.perf_counter()
    
    print("Reading file + Training and pruning Time = ", end_t - start_t)
    
    # Testing
    print("-- Testing --")
    result = classifier.predict(Xtest, patClassIdTest)
    if result != None:
        print("Number of wrong predicted samples = ", result.summis)
        numTestSample = Xtest.shape[0]
        print("Error Rate = ", np.round(result.summis / numTestSample * 100, 2), "%")