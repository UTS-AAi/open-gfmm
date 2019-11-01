# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 14:09:02 2019

@author: Thanh Tung Khuat

This is the implementation of class related to base learner of random hyperboxes
using GFMM with Manhattan distance version

"""
import sys, os
sys.path.insert(0, os.path.pardir)

import numpy as np
from GFMM.faster_onlinegfmm import OnlineGFMM
from randombox.baselearner import BaseLearner

class GFMM_BaseLearner(OnlineGFMM, BaseLearner):
    """
    This is an extended class of OnlineGFMM to implement a class for a base learner
    of random hyperboxes
    
        Parameters:
            gamma: float, optional (default = 1)
                The speed of decreasing of the membership function
            
            theta: float (in the rage of [0, 1]), optional (default = 0.2)
                The maximum hyperbox size threshold
                
            theta_min: float (in the range of [0, 1] and <= teta), optional (default = teta)
                The minimum value of maximum hyperbox size
                
            sub_selected_feature: numpy array, optional (default = empty)
                The subset of selected features, the training set in the fit method 
                of this classifier was filtered using this subset
    """
    
    
    def __init__(self, gamma = 1, theta = 0.2, theta_min = 0.2, sub_selected_feature = np.array([])):
        super(GFMM_BaseLearner, self).__init__(gamma=gamma, teta=theta, tMin=theta_min)
        self.set_value_sub_selected_feature(sub_selected_feature)
        
    def fit(self, X_l, X_u, patClassId, sub_selected_feature=None):
        """
        Training the classifier

         Xl             Input data lower bounds (rows = objects, columns = features)
         Xu             Input data upper bounds (rows = objects, columns = features)
         patClassId     Input data class labels (crisp). patClassId[i] = 0 corresponds to an unlabeled item

        sub_selected_feature: indices of selected features in the larger dataset used for this class
        """
        if sub_selected_feature is not None:
            self.set_value_sub_selected_feature(sub_selected_feature)
        
        super(GFMM_BaseLearner, self).fit(X_l, X_u, patClassId)
        
    def predict(self, Xl_Test, Xu_Test, patClassIdTest):
        """
        Perform classification

            result = predict(Xl_Test, Xu_Test, patClassIdTest)

        INPUT:
            Xl_Test             Test data lower bounds (rows = objects, columns = features)
            Xu_Test             Test data upper bounds (rows = objects, columns = features)
            patClassIdTest	    Test data class labels (crisp)
            
        Return:
            result        A object with Bunch datatype containing all results as follows:
                          + summis           Number of misclassified objects
                          + misclass         Binary error map
                          + numSampleInBoundary     The number of samples in decision boundary
                          + predicted_class   Predicted class
        """
        # Get testing set used for this base classifier by using sub_selected_feature
        selected_Xl_Test = Xl_Test[:, self.sub_selected_feature]
        selected_Xu_Test = Xu_Test[:, self.sub_selected_feature]
        
        result = super(GFMM_BaseLearner, self).predict(selected_Xl_Test, selected_Xu_Test, patClassIdTest, True)
        self.predicted_class = result.predicted_class
        
        return result
    
    def execute_learner(self, Xl, Xu, patId, gamma=1):
        """
            Return membership, predicted_class
            a boolean vector indicating correct (1), incorrect (0) for each sample
        """
        selected_Xl_Test = Xl[:, self.sub_selected_feature]
        selected_Xu_Test = Xu[:, self.sub_selected_feature]
        
        result = super(GFMM_BaseLearner, self).predict(selected_Xl_Test, selected_Xu_Test, patId, True)
        is_predicted_correct = (~result.misclass).astype(np.int)
        
        return (result.mem_vals, result.predicted_class, is_predicted_correct)
    
    def pruning_val(self, XlVal, XuVal, patClassIdVal, accuracy_threshold = 0.5):
        """
        pruning handling based on validation (validation routine) with hyperboxes stored in self. V, W, classId
    
          result = pruning_val(XlVal, XuVal, patClassIdVal)
    
            INPUT
              XlVal               Validation data lower bounds (rows = objects, columns = features)
              XuVal               Validation data upper bounds (rows = objects, columns = features)
              patClassIdVal       Validation data class labels (crisp)
              accuracy_threshold  The minimum accuracy for each hyperbox
        """
        # Get validaation set used for this base classifier by using sub_selected_feature
        selected_Xl_Val = XlVal[:, self.sub_selected_feature]
        selected_Xu_Val = XuVal[:, self.sub_selected_feature]
        
        super(GFMM_BaseLearner, self).pruning_val(selected_Xl_Val, selected_Xu_Val, patClassIdVal, accuracy_threshold, True)
        