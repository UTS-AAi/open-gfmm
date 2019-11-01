# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 14:09:02 2019

@author: Thanh Tung Khuat

This is the implementation of class related to base learner of random hyperboxes
using GFMM with an improved online learning version

"""
import sys, os
sys.path.insert(0, os.path.pardir)

import numpy as np
import random as rd
from functionhelper.membershipcalc import memberG, memberG_with_selected_features
from functionhelper.bunchdatatype import Bunch
from GFMM.improvedonlinegfmm import ImprovedOnlineGFMM
from randombox.baselearner import BaseLearner

class IOL_GFMM_BaseLearner(ImprovedOnlineGFMM, BaseLearner):
    """
    This is an extended class of ImprovedOnlineGFMM to implement a class for a base learner
    of random hyperboxes
    
        Parameters:
            gamma: float, optional (default = 1)
                The speed of decreasing of the membership function
            
            theta: float (in the rage of [0, 1]), optional (default = 0.2)
                The maximum hyperbox size threshold
                
            sigma: float (in the range of [0, 1]), optional (default = 0)
                The minimum membership value of two aggregated hyperboxes
                
            sub_selected_feature: numpy array, optional (default = empty)
                The subset of selected features, the training set in the fit method 
                of this classifier was filtered using this subset
    """
    
    
    def __init__(self, gamma = 1, theta = 0.2, sigma = 0, sub_selected_feature = np.array([])):
        super(IOL_GFMM_BaseLearner, self).__init__(gamma=gamma, teta=theta)
        self.set_value_sub_selected_feature(sub_selected_feature)
        self.predicted_class = []
        
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
        
        super(IOL_GFMM_BaseLearner, self).fit(X_l, X_u, patClassId)
        
    def execute_learner(self, Xl, Xu, patId, gamma=1):
        """
            Return membership, predicted_class
            a boolean vector indicating correct (1), incorrect (0) for each sample
        """
        selected_Xl_Test = Xl[:, self.sub_selected_feature]
        selected_Xu_Test = Xu[:, self.sub_selected_feature]
        
        result = super(IOL_GFMM_BaseLearner, self).predict(selected_Xl_Test, selected_Xu_Test, patId, True)
        is_predicted_correct = (~np.array(result.misclass, np.bool)).astype(np.int)
        
        return (result.mem_vals, result.predicted_class, is_predicted_correct)
    
    def execute_learner_index(self, Xl, Xu, patId, gamma=1):
        """
            Return membership, predicted_class
            a boolean vector indicating correct (1), incorrect (0) for each sample
        """
        selected_Xl_Test = Xl[:, self.sub_selected_feature]
        selected_Xu_Test = Xu[:, self.sub_selected_feature]
        
        result = super(IOL_GFMM_BaseLearner, self).predict(selected_Xl_Test, selected_Xu_Test, patId, True)
        is_predicted_correct = (~np.array(result.misclass, np.bool)).astype(np.int)
        index_winner = result.id_winner
        
        useful_id_box = index_winner * is_predicted_correct
        
        unique_useful_id_box, counts_score = np.unique(useful_id_box, return_counts=True)
        
        return (unique_useful_id_box, counts_score)
     
        
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
        
        result = super(IOL_GFMM_BaseLearner, self).predict(selected_Xl_Test, selected_Xu_Test, patClassIdTest, True)
        self.predicted_class = result.predicted_class
        
        return result
    
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
        
        super(IOL_GFMM_BaseLearner, self).pruning_val(selected_Xl_Val, selected_Xu_Val, patClassIdVal, accuracy_threshold, True)
        
    def find_weight_hyperbox(self, XlVal, XuVal, patClassIdVal):
        """
            Find accuracy of each hyperbox
        """
        if len(XlVal.shape) == 1:
            XlVal = XlVal.reshape(1, -1)
        if len(XuVal.shape) == 1:
            XuVal = XuVal.reshape(1, -1)
            
        # Get validaation set used for this base classifier by using sub_selected_feature
        selected_Xl_Val = XlVal[:, self.sub_selected_feature]
        selected_Xu_Val = XuVal[:, self.sub_selected_feature]
            
        #initialization
        yX = selected_Xl_Val.shape[0]
        correct_prediction = np.zeros(yX)
        correct_vec = np.zeros(self.V.shape[0])
        incorrect_vec = np.zeros(self.V.shape[0])
        weight_feature = np.zeros(self.V.shape[1])
        
        for i in range(yX):
            mem, f_w = memberG_with_selected_features(selected_Xl_Val[i, :], selected_Xu_Val[i, :], self.V, self.W, self.gamma) # calculate memberships for all hyperboxes
            bmax = mem.max()	                                          # get max membership value
            maxVind = np.nonzero(mem == bmax)[0]                         # get indexes of all hyperboxes with max membership

            cls_same_mem = np.unique(self.classId[maxVind])
            if len(cls_same_mem) > 1:
                id_box_with_one_sample = np.nonzero(self.counter[maxVind] == 1)[0]
                if len(id_box_with_one_sample) == 0:
                    sum_prod_denum = (mem[maxVind] * self.counter[maxVind]).sum()
                    if sum_prod_denum == 0:
                        cls_val = rd.choice(cls_same_mem)
                        pre_id_cls = np.nonzero(self.classId[maxVind] == cls_val)[0]
                    else:
                        max_prob = -1
                        pre_id_cls = None
                        for c in cls_same_mem:
                            id_cls = np.nonzero(self.classId[maxVind] == c)[0]
                            sum_pro_num = (mem[maxVind[id_cls]] * self.counter[maxVind[id_cls]]).sum()
                            tmp = sum_pro_num / sum_prod_denum
                            
                            if tmp > max_prob or (tmp == max_prob and pre_id_cls is not None and self.counter[maxVind[id_cls]].sum() > self.counter[maxVind[pre_id_cls]].sum()):
                                max_prob = tmp
                                cls_val = c
                                pre_id_cls = id_cls
                            
                    if cls_val == patClassIdVal[i]:
                        correct_vec[maxVind[pre_id_cls]] = correct_vec[maxVind[pre_id_cls]] + 1
                        correct_prediction[i] = 1
                        
                        tmp_set = set()
                        for jj in maxVind[pre_id_cls]:
#                            print(f_w[jj])
#                            print("\n")
                            tmp_set |= set(f_w[jj])
                            
                        weight_feature[list(tmp_set)] = weight_feature[list(tmp_set)] + bmax
                           
                    else:
                        incorrect_vec[maxVind[pre_id_cls]] = incorrect_vec[maxVind[pre_id_cls]] + 1
                else:
                    #print('One sample')
                    cls_val = self.classId[maxVind[id_box_with_one_sample[0]]]
                    if cls_val == patClassIdVal[i]:
                        correct_vec[maxVind[id_box_with_one_sample]] = correct_vec[maxVind[id_box_with_one_sample]] + 1
                        correct_prediction[i] = 1
                        
                        tmp_set = set()
                        for jj in maxVind[id_box_with_one_sample]:
                            tmp_set |= set(f_w[jj])
                            
                        weight_feature[list(tmp_set)] = weight_feature[list(tmp_set)] + bmax
                        
                    else:
                        incorrect_vec[maxVind[id_box_with_one_sample]] = incorrect_vec[maxVind[id_box_with_one_sample]] + 1
                                  
            else:
                if self.classId[maxVind[0]] == patClassIdVal[i]:
                    correct_vec[maxVind] = correct_vec[maxVind] + 1
                    correct_prediction[i] = 1
                    tmp_set = set()
                    for jj in maxVind:
                        try:
                            tmp_set |= set(f_w[jj])
                        except:
                            print(f_w[jj])
                        
                    weight_feature[list(tmp_set)] = weight_feature[list(tmp_set)] + bmax
                    
                else:
                    incorrect_vec[maxVind] = incorrect_vec[maxVind] + 1
                    
        # divide by the number of samples
        self.weight = correct_vec / (correct_vec + incorrect_vec)
        min_val = np.nanmean(self.weight) # np.nanmin(self.weight) - 0.01
        self.weight = np.nan_to_num(self.weight, nan=min_val)
        
        weight_feature = weight_feature.sum(axis = 0) / yX
        
        return (self.weight, correct_prediction, weight_feature)
        