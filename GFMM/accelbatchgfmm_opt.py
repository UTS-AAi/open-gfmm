# -*- coding: utf-8 -*-
"""
Created on Mon Sep 10 09:42:14 2018

@author: Thanh Tung Khuat

    Accelerated Batch GFMM classifier (training core). This version uses lemma to accelerate the learning algorithm
        
        AccelBatchGFMM(gamma, teta, bthres, simil, sing, isDraw, oper, isNorm, norm_range)
  
    INPUT:
        gamma           Membership function slope (default: 1)
        teta            Maximum hyperbox size (default: 1)
        bthres          Similarity threshold for hyperbox concatenation (default: 0.5)
        simil           Similarity measure: 'short', 'long' or 'mid' (default: 'mid')
        sing            Use 'min' or 'max' (default) memberhsip in case of assymetric similarity measure (simil='mid')
        isDraw          Progress plot flag (default: False)
        oper            Membership calculation operation: 'min' or 'prod' (default: 'min')
        isNorm          Do normalization of input training samples or not?
        norm_range      New ranging of input data after normalization, for example: [0, 1]
        cardin      Input hyperbox cardinalities
        clusters    Identifiers of objects in each input hyperbox 
        
    ATTRIBUTES:
        V               Hyperbox lower bounds
        W               Hyperbox upper bounds
        classId         Hyperbox class labels (crisp)
        # Comment 2 attributes to accelerate the code because of non-using now
        cardin          Hyperbox cardinalities (the number of training samples is covered by corresponding hyperboxes)
        clusters        Identifiers of input objects in each hyperbox (indexes of training samples covered by corresponding hyperboxes)

"""

import sys, os
sys.path.insert(0, os.path.pardir)
from pathlib import Path
import ast
import numpy as np
import time
import matplotlib
try:
    matplotlib.use('TkAgg')
except:
    pass

from functionhelper import UNLABELED_CLASS
from functionhelper.preprocessinghelper import loadDataset, string_to_boolean
from functionhelper.drawinghelper import drawbox
from functionhelper.hyperboxadjustment import isOverlap, modifiedIsOverlap, extendedModifiedIsOverlap
from GFMM.basebatchlearninggfmm import BaseBatchLearningGFMM
from functionhelper.membershipcalc import asym_similarity_one_many, memberG
from GFMM.classification import predict_with_probability, predict, predict_with_weighted_manhattan, predict_with_manhattan
import random
from functionhelper.measurehelper import manhattan_distance
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold

class AccelBatchGFMM(BaseBatchLearningGFMM):
    
    def __init__(self, gamma = 1, teta = 1, bthres = 0.5, simil = 'mid', sing = 'max', isDraw = False, oper = 'min', isNorm = False, norm_range = [0, 1], cardin = np.array([], dtype=np.int64), clusters = np.array([], dtype=object)):
        BaseBatchLearningGFMM.__init__(self, gamma, teta, isDraw, oper, isNorm, norm_range)
        
        self.bthres = bthres
        self.simil = simil
        self.sing = sing
        self.cardin = cardin
    
    
    def fit(self, X_l, X_u, patClassId, num_pat=None):  
        """
            Xl              Input data lower bounds (rows = objects, columns = features)
            Xu              Input data upper bounds (rows = objects, columns = features)
            patClassId      Input data class labels (crisp)
            num_pat         Save the number of samples in hyperboxes [X_l, X_u]
        """
        
        if self.isNorm == True:
            X_l, X_u = self.dataPreprocessing(X_l, X_u)
            
        time_start = time.perf_counter()
         
        self.V = X_l
        self.W = X_u
        self.classId = patClassId
        self.num_potential_hyperbox = 0
        
        yX, xX = X_l.shape
        
        if len(self.cardin) == 0:
            if num_pat is None:
                self.cardin = np.ones(yX)
            else:
                self.cardin = np.array(num_pat)
        
        if self.isDraw:
            mark_col = np.array(['r', 'g', 'b', 'y', 'c', 'm', 'k'])
            drawing_canvas = self.initializeCanvasGraph("GFMM - Faster AGGLO-2", xX)
                
            # plot initial hyperbox
            Vt, Wt = self.pcatransform()
            color_ = np.empty(len(self.classId), dtype = object)
            for c in range(len(self.classId)):
                color_[c] = mark_col[self.classId[c]]
            boxes = drawbox(Vt, Wt, drawing_canvas, color_)
            self.delay()
            hyperboxes = list(boxes)
            
        # training
        isTraining = True
        threshold = max(self.bthres, 1 - self.gamma * self.teta)
        while isTraining:
            isTraining = False
            
            k = 0 # input pattern index
            while k < len(self.classId):
                idx_same_classes = (self.classId == self.classId[k]) | (self.classId == UNLABELED_CLASS) | ((self.classId != self.classId[k]) & (self.classId[k] == UNLABELED_CLASS))
                idx_same_classes[k] = False # remove element in the position k
                idex = np.arange(len(self.classId))
                idex = idex[idx_same_classes] # keep the indices of elements retained
                V_same_class = self.V[idx_same_classes]
                W_same_class = self.W[idx_same_classes]
                
                if self.simil == 'short':
                    b = memberG(self.W[k], self.V[k], V_same_class, W_same_class, self.gamma, self.oper)
                elif self.simil == 'long':
                    b = memberG(self.V[k], self.W[k], W_same_class, V_same_class, self.gamma, self.oper)
                else:
                    #b = memberG(self.V[k], self.W[k], V_same_class, W_same_class, self.gamma, self.oper)
                    b = asym_similarity_one_many(self.V[k], self.W[k], V_same_class, W_same_class, self.gamma, self.sing, self.oper)
                
                indB = np.argsort(b)[::-1]
                idex = idex[indB]
                sortB = b[indB]
                
                maxB = sortB[sortB >= threshold]	# apply membership threshold
                
                if len(maxB) > 0:
                    idexmax = idex[sortB >= threshold]
                    
                    pairewise_maxb = np.concatenate((np.minimum(k, idexmax)[:, np.newaxis], np.maximum(k,idexmax)[:, np.newaxis], maxB[:, np.newaxis]), axis=1)

                    for i in range(pairewise_maxb.shape[0]):
                        # calculate new coordinates of k-th hyperbox by including pairewise_maxb(i,1)-th box, scrap the latter and leave the rest intact
                        # agglomorate pairewise_maxb(i, 0) and pairewise_maxb(i, 1) by adjusting pairewise_maxb(i, 0)
                        # remove pairewise_maxb(i, 1) by getting newV from 1 -> pairewise_maxb(i, 0) - 1, new coordinates for pairewise_maxb(i, 0), from pairewise_maxb(i, 0) + 1 -> pairewise_maxb(i, 1) - 1, pairewise_maxb(i, 1) + 1 -> end
                        self.num_potential_hyperbox = self.num_potential_hyperbox + 1
                        row1 = int(pairewise_maxb[i, 0])
                        row2 = int(pairewise_maxb[i, 1])
                        newV = np.concatenate((self.V[:row1], np.minimum(self.V[row1], self.V[row2]).reshape(1, -1), self.V[row1 + 1:row2], self.V[row2 + 1:]), axis=0)
                        newW = np.concatenate((self.W[:row1], np.maximum(self.W[row1], self.W[row2]).reshape(1, -1), self.W[row1 + 1:row2], self.W[row2 + 1:]), axis=0)
                        newClassId = np.concatenate((self.classId[:row2], self.classId[row2 + 1:]))
                        if (newClassId[row1] == UNLABELED_CLASS):
                            newClassId[row1] = self.classId[row2]
               
                        # adjust the hyperbox if no overlap and maximum hyperbox size is not violated
                        # position of adjustment is pairewise_maxb[i, 0] in new bounds
                        if ((((newW[row1] - newV[row1]) <= self.teta).all() == True) and (not extendedModifiedIsOverlap(newV, newW, row1, newClassId))):
                            self.V = newV
                            self.W = newW
                            self.classId = newClassId
                            
                            self.cardin[row1] = self.cardin[row1] + self.cardin[row2]
                            self.cardin = np.append(self.cardin[0:row2], self.cardin[row2 + 1:])
                        
                            isTraining = True
                            
                            if k != pairewise_maxb[i, 0]: # position pairewise_maxb[i, 1] (also k) is removed, so next step should start from pairewise_maxb[i, 1]
                                k = k - 1
                                
                            if self.isDraw:
                                try:
                                    hyperboxes[int(pairewise_maxb[i, 1])].remove()
                                    hyperboxes[int(pairewise_maxb[i, 0])].remove()
                                except:
                                    print("No remove old hyperbox")
                                
                                Vt, Wt = self.pcatransform()
                                
                                box_color = 'k'
                                if self.classId[int(pairewise_maxb[i, 0])] < len(mark_col):
                                    box_color = mark_col[self.classId[int(pairewise_maxb[i, 0])]]
                                
                                box = drawbox(np.asmatrix(Vt[int(pairewise_maxb[i, 0])]), np.asmatrix(Wt[int(pairewise_maxb[i, 0])]), drawing_canvas, box_color)
                                self.delay()
                                hyperboxes[int(pairewise_maxb[i, 0])] = box[0]
                                hyperboxes.remove(hyperboxes[int(pairewise_maxb[i, 1])])
                                
                            break # if hyperbox adjusted there's no need to look at other hyperboxes
                            
                        
                k = k + 1
        
        time_end = time.perf_counter()
        self.elapsed_training_time = time_end - time_start
         
        return self
    
    def predict(self, Xl_Test, Xu_Test, patClassIdTest=None, newVer = True, ruleNewVer = 1):
        """
        Perform classification

            result = predict(Xl_Test, Xu_Test, patClassIdTest)

        INPUT:
            Xl_Test             Test data lower bounds (rows = objects, columns = features)
            Xu_Test             Test data upper bounds (rows = objects, columns = features)
            patClassIdTest	    Test data class labels (crisp)
            newVer              + True: Using an additional criterion
                                + False: No use of an additional criterion
            ruleNewVer          The version used to find the final hyperbox among many winners
                                + 1: using a prob formula with the cardinality to support the classification process
                                + 2: using weighted manhattan distance
                                + other: using manhattan distance

        OUTPUT:
            predicted classes for testing samples
        """
        #Xl_Test, Xu_Test = delete_const_dims(Xl_Test, Xu_Test)
        # Normalize testing dataset if training datasets were normalized
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
        self.predicted_class = None
        if Xl_Test.shape[0] > 0:
            if newVer:
                if ruleNewVer == 1:
                    result = predict_with_probability(self.V, self.W, self.classId, self.cardin, Xl_Test, Xu_Test, patClassIdTest, self.gamma)
                elif ruleNewVer == 2:
                    result = predict_with_weighted_manhattan(self.V, self.W, self.classId, self.cardin, Xl_Test, Xu_Test, patClassIdTest, self.gamma)
                else:
                    result = predict_with_manhattan(self.V, self.W, self.classId, Xl_Test, Xu_Test, patClassIdTest, self.gamma)
            else:
                result = predict(self.V, self.W, self.classId, Xl_Test, Xu_Test, patClassIdTest, self.gamma)
                
            self.predicted_class = np.array(result.predicted_class, np.int)

        return self.predicted_class  
    
    def pruning_val(self, XlT, XuT, patClassIdTest, accuracy_threshold = 0.5, newVerPredict = True, newRuleManyWinners=1, keepZeroValBox = False, deleted_fully_contained = False):
        """
        pruning handling based on validation (validation routine) with hyperboxes stored in self. V, W, classId
    
          result = pruning_val(XlT,XuT,patClassIdTest)
    
            INPUT
              XlT               Test data lower bounds (rows = objects, columns = features)
              XuT               Test data upper bounds (rows = objects, columns = features)
              patClassIdTest    Test data class labels (crisp)
              accuracy_threshold  The minimum accuracy for each hyperbox
              newVerPredict     + True: using an additional rule to choose the right hyperbox
                                + False: No using probability formula for prediction
              newRuleManyWinners  regulate which rule is used to choose the final hyperbox among many winner hyperboxes
                                + 1: Using a probability formula for prediction in addition to fuzzy membership
                                + 2: Using the weighted Manhattan distance
                                + others: Using the Manhattan distance
              keepZeroValBox    Whether to keep the hyperboxes which do not join the prediction process on the validation set
                                + True: Keep them
                                + False: Pruning based on the predictive results on the validation set
                                
        """
        # Delete all hyperboxes contained in hyperboxes representing other classes
        random.seed(0)
        if deleted_fully_contained == True:
            num_classes = np.unique(self.classId)
            # find hyperbox index needing to be removed and set flag = False
            keep_id_non_contained = np.ones(len(self.classId), dtype=bool)
            for cls_ele in num_classes:
                # Filter all hyperboxes representing class cls_ele and other classes
                ids_box_cls_ele = np.nonzero(self.classId == cls_ele)[0]
                ids_box_non_cls_ele = np.nonzero(self.classId != cls_ele)[0]
                # Compute membership function
                for id_box_cls_ele in ids_box_cls_ele:
                    mem_id_cls_ele = memberG(self.V[id_box_cls_ele, :], self.W[id_box_cls_ele, :], self.V[ids_box_non_cls_ele], self.W[ids_box_non_cls_ele], self.gamma, self.oper)
                    if mem_id_cls_ele.max() == 1:
                        keep_id_non_contained[id_box_cls_ele] = False
            
            if (keep_id_non_contained == False).sum() > 0:
                self.V = self.V[keep_id_non_contained]
                self.W = self.W[keep_id_non_contained]
                self.classId = self.classId[keep_id_non_contained]
                self.cardin = self.cardin[keep_id_non_contained]
                    
        #initialization
        yX = XlT.shape[0]
        no_predicted_samples_hyperboxes = np.zeros((len(self.classId), 2))
        # classifications
        for i in range(yX):
            mem = memberG(XlT[i, :], XuT[i, :], self.V, self.W, self.gamma, self.oper) # calculate memberships for all hyperboxes
            bmax = mem.max()	                                          # get max membership value
            maxVind = np.nonzero(mem == bmax)[0]                         # get indexes of all hyperboxes with max membership
            
            if len(maxVind) == 1:
                # Only one hyperbox with the highest membership function
                
                if self.classId[maxVind[0]] == patClassIdTest[i]:
                    no_predicted_samples_hyperboxes[maxVind[0], 0] = no_predicted_samples_hyperboxes[maxVind[0], 0] + 1                 
                else:
                    no_predicted_samples_hyperboxes[maxVind[0], 1] = no_predicted_samples_hyperboxes[maxVind[0], 1] + 1
            else:
                if newVerPredict == True:
                    cls_same_mem = np.unique(self.classId[maxVind])
                    if len(cls_same_mem) > 1:
                        if newRuleManyWinners == 1:
                            is_find_prob_val = True
                            if bmax == 1:
                                id_box_with_one_sample = np.nonzero(self.cardin[maxVind] == 1)[0]
                                if len(id_box_with_one_sample) > 0:
                                    is_find_prob_val = False
                                    id_min = random.choice(maxVind[id_box_with_one_sample])
                            
                            if is_find_prob_val == True:
                                sum_prod_denum = (mem[maxVind] * self.cardin[maxVind]).sum()
                                max_prob = -1
                                pre_id_cls = None
                                for c in cls_same_mem:
                                    id_cls = np.nonzero(self.classId[maxVind] == c)[0]
                                    sum_pro_num = (mem[maxVind[id_cls]] * self.cardin[maxVind[id_cls]]).sum()
                                    tmp = sum_pro_num / sum_prod_denum
                                    
                                    if tmp > max_prob or (tmp == max_prob and pre_id_cls is not None and self.cardin[maxVind[id_cls]].sum() > self.cardin[maxVind[pre_id_cls]].sum()):
                                        max_prob = tmp
                                        pre_id_cls = id_cls
                                        id_min = random.choice(maxVind[id_cls])
                        else:
                            if (XlT[i] == XuT[i]).all() == False:
                                XlT_mat = np.ones((len(maxVind), 1)) * XlT[i]
                                XuT_mat = np.ones((len(maxVind), 1)) * XuT[i]
                                XgT_mat = (XlT_mat + XuT_mat) / 2
                            else:
                                XgT_mat = np.ones((len(maxVind), 1)) * XlT[i]
                            # Find all average points of all hyperboxes with the same membership value
                            avg_point_mat = (self.V[maxVind] +self. W[maxVind]) / 2
                            # compute the manhattan distance from XgT_mat to all average points of all hyperboxes with the same membership value
                            maht_dist = manhattan_distance(avg_point_mat, XgT_mat)
                            if newRuleManyWinners == 2:
                                # Combine with samples
                                weighted_sample = self.cardin[maxVind] / self.cardin[maxVind].sum()
                                weighted_maht_dist = maht_dist / weighted_sample
                                
                                id_min_dist = weighted_maht_dist.argmin()
                            else:
                                id_min_dist = maht_dist.argmin()
                                
                            id_min = maxVind[id_min_dist]
                    else:
                        id_min = random.choice(maxVind)
                else:
                    # More than one hyperbox with highest membership => random choosing
                    id_min = maxVind[np.random.randint(len(maxVind))]
                        
                if self.classId[id_min] != patClassIdTest[i] and patClassIdTest[i] != UNLABELED_CLASS:
                    no_predicted_samples_hyperboxes[id_min, 1] = no_predicted_samples_hyperboxes[id_min, 1] + 1
                else:
                    no_predicted_samples_hyperboxes[id_min, 0] = no_predicted_samples_hyperboxes[id_min, 0] + 1
                    
        # pruning handling based on the validation results
        tmp_no_box = no_predicted_samples_hyperboxes.shape[0]
        accuracy_larger_half = np.zeros(tmp_no_box).astype(np.bool)
        accuracy_larger_half_keep_nojoin = np.zeros(tmp_no_box).astype(np.bool)
        for i in range(tmp_no_box):
            if (no_predicted_samples_hyperboxes[i, 0] + no_predicted_samples_hyperboxes[i, 1] != 0) and no_predicted_samples_hyperboxes[i, 0] / (no_predicted_samples_hyperboxes[i, 0] + no_predicted_samples_hyperboxes[i, 1]) >= accuracy_threshold:
                accuracy_larger_half[i] = True
                accuracy_larger_half_keep_nojoin[i] = True
            if (no_predicted_samples_hyperboxes[i, 0] + no_predicted_samples_hyperboxes[i, 1] == 0):
                accuracy_larger_half_keep_nojoin[i] = True
        
        if keepZeroValBox == True:
            self.V = self.V[accuracy_larger_half_keep_nojoin]
            self.W = self.W[accuracy_larger_half_keep_nojoin]
            self.classId = self.classId[accuracy_larger_half_keep_nojoin]
            self.cardin = self.cardin[accuracy_larger_half_keep_nojoin]
        else:
            # keep one hyperbox for class prunned all
            current_classes = np.unique(self.classId)
            class_tmp = np.unique(self.classId[accuracy_larger_half])
            for c in current_classes:
                if c not in class_tmp:
                    pos = np.nonzero(self.classId == c)[0]
                    id_kept = np.random.randint(len(pos))
                    accuracy_larger_half[pos[id_kept]] = True
            
            V_prun_remove = self.V[accuracy_larger_half]
            W_prun_remove = self.W[accuracy_larger_half]
            classId_prun_remove = self.classId[accuracy_larger_half]
            numSample_prun_remove = self.cardin[accuracy_larger_half]
            
            W_prun_keep = self.W[accuracy_larger_half_keep_nojoin]
            V_prun_keep = self.V[accuracy_larger_half_keep_nojoin]
            classId_prun_keep = self.classId[accuracy_larger_half_keep_nojoin]
            numSample_prun_keep = self.cardin[accuracy_larger_half_keep_nojoin]
            
            if newVerPredict == True:
                if newRuleManyWinners == 1:
                    result_prun_remove = predict_with_probability(V_prun_remove, W_prun_remove, classId_prun_remove, numSample_prun_remove, XlT, XuT, patClassIdTest, self.gamma, self.oper)
                    result_prun_keep_nojoin = predict_with_probability(V_prun_keep, W_prun_keep, classId_prun_keep, numSample_prun_keep, XlT, XuT, patClassIdTest, self.gamma, self.oper)
                elif newRuleManyWinners == 2:
                    result_prun_remove = predict_with_weighted_manhattan(V_prun_remove, W_prun_remove, classId_prun_remove, numSample_prun_remove, XlT, XuT, patClassIdTest, self.gamma, self.oper)
                    result_prun_keep_nojoin = predict_with_weighted_manhattan(V_prun_keep, W_prun_keep, classId_prun_keep, numSample_prun_keep, XlT, XuT, patClassIdTest, self.gamma, self.oper)
                else:
                    result_prun_remove = predict_with_manhattan(V_prun_remove, W_prun_remove, classId_prun_remove, XlT, XuT, patClassIdTest, self.gamma, self.oper)
                    result_prun_keep_nojoin = predict_with_manhattan(V_prun_keep, W_prun_keep, classId_prun_keep, XlT, XuT, patClassIdTest, self.gamma, self.oper)
            else:
                result_prun_remove = predict(V_prun_remove, W_prun_remove, classId_prun_remove, XlT, XuT, patClassIdTest, self.gamma, self.oper)
                result_prun_keep_nojoin = predict(V_prun_keep, W_prun_keep, classId_prun_keep, XlT, XuT, patClassIdTest, self.gamma, self.oper)
            if (result_prun_remove.summis <=  result_prun_keep_nojoin.summis):
                self.V = V_prun_remove
                self.W = W_prun_remove
                self.classId = classId_prun_remove
                self.cardin = numSample_prun_remove
            else:
                self.V = V_prun_keep
                self.W = W_prun_keep
                self.classId = classId_prun_keep
                self.cardin = numSample_prun_keep
            
        
    
if __name__ == '__main__':
    """
    INPUT parameters from command line
    arg1: + 1 - training and testing datasets are located in separated files
          + 2 - training and testing datasets are located in the same files
    arg2: path to file containing the training dataset (arg1 = 1) or both training and testing datasets (arg1 = 2)
    arg3: + path to file containing the testing dataset (arg1 = 1)
          + percentage of the training dataset in the input file
    arg4: + True: drawing hyperboxes during the training process
          + False: no drawing
    arg5: + Maximum size of hyperboxes (teta, default: 1)
    arg6: + gamma value (default: 1)
    arg7: + Similarity threshold (default: 0.5)
    arg8: + Similarity measure: 'short', 'long' or 'mid' (default: 'mid')
    arg9: + operation used to compute membership value: 'min' or 'prod' (default: 'min')
    arg10: + do normalization of datasets or not? True: Normilize, False: No normalize (default: True)
    arg11: + range of input values after normalization (default: [0, 1])   
    arg12: + Use 'min' or 'max' (default) membership in case of assymetric similarity measure (simil='mid')
    """
    # Init default parameters
    # if len(sys.argv) < 5:
    #     isDraw = False
    # else:
    #     isDraw = string_to_boolean(sys.argv[4])
    
    # if len(sys.argv) < 6:
    #     teta = 1    
    # else:
    #     teta = float(sys.argv[5])
    
    # if len(sys.argv) < 7:
    #     gamma = 1
    # else:
    #     gamma = float(sys.argv[6])
    
    # if len(sys.argv) < 8:
    #     bthres = 0.5
    # else:
    #     bthres = float(sys.argv[7])
    
    # if len(sys.argv) < 9:
    #     simil = 'mid'
    # else:
    #     simil = sys.argv[8]
    
    # if len(sys.argv) < 10:
    #     oper = 'min'
    # else:
    #     oper = sys.argv[9]
    
    # if len(sys.argv) < 11:
    #     isNorm = True
    # else:
    #     isNorm = string_to_boolean(sys.argv[10])
    
    # if len(sys.argv) < 12:
    #     norm_range = [0, 1]
    # else:
    #     norm_range = ast.literal_eval(sys.argv[11])
        
    # if len(sys.argv) < 13:
    #     sing = 'max'
    # else:
    #     sing = sys.argv[12]
    
    # start_t = time.perf_counter()        
    # if sys.argv[1] == '1':
    #     training_file = sys.argv[2]
    #     testing_file = sys.argv[3]

    #     # Read training file
    #     Xtr, X_tmp, patClassIdTr, pat_tmp = loadDataset(training_file, 1, False)
    #     # Read testing file
    #     X_tmp, Xtest, pat_tmp, patClassIdTest = loadDataset(testing_file, 0, False)
    
    # else:
    #     dataset_file = sys.argv[2]
    #     percent_Training = float(sys.argv[3])
    #     Xtr, Xtest, patClassIdTr, patClassIdTest = loadDataset(dataset_file, percent_Training, False)
    isPruning = True
    training_file = "C:\\Hyperbox-based-ML\\Dataset\\train_test\\training_testing_data\\spambase_dps_tr.dat"
    testing_file = "C:\\Hyperbox-based-ML\\Dataset\\train_test\\training_testing_data\\spambase_dps_test.dat"
    validation_file = "C:\\Hyperbox-based-ML\\Dataset\\train_test\\training_testing_data\\spambase_dps_val.dat"
    gamma = 1
    #teta = 0.1
    bthres = 0.5
    isDraw = True
    oper = 'min'
    isNorm = False
    norm_range = [0, 1]
    # # Read training file
    # Xtr, X_tmp, patClassIdTr, pat_tmp = loadDataset(training_file, 1, False)
    # # Read testing file
    # X_tmp, Xtest, pat_tmp, patClassIdTest = loadDataset(testing_file, 0, False)
    # # Read validation file
    # _, Xval, _, patClassIdVal = loadDataset(validation_file, 0, False)
    
    # classifier = AccelBatchGFMM(gamma, teta, bthres, simil, sing, isDraw, oper, isNorm, norm_range)
    # start_t = time.perf_counter()
    # classifier.fit(Xtr, Xtr, patClassIdTr)
    # print("Before pruning:")
    # print('V size = ', classifier.V.shape)
    # print('W size = ', classifier.W.shape)
    # newRuleManyWinners = 3
    # if isPruning == True:
    #     classifier.pruning_val(Xval, Xval, patClassIdVal, newRuleManyWinners=newRuleManyWinners, keepZeroValBox = False)
    #     print("After pruning:")
    #     print('V size = ', classifier.V.shape)
    #     print('W size = ', classifier.W.shape)
    # end_t = time.perf_counter()
    
    # print("Training and pruning Time = ", end_t - start_t)
    # # Testing
    # print("-- Testing --")
    # predicted_cls = classifier.predict(Xtest, Xtest, patClassIdTest, True, newRuleManyWinners)
    # if predicted_cls is not None:
    #     print(classification_report(patClassIdTest, predicted_cls))

    path_train_test_file = Path.joinpath(Path.cwd().parent, 'Dataset')
    df_train_path = Path.joinpath(path_train_test_file, 'syn_num_train_ori.csv')
    df_test_path = Path.joinpath(path_train_test_file, 'syn_num_test.csv')
    # Read training file
    Xtr, _, ytr, _ = loadDataset(df_train_path, 1, False)
    # Read testing file
    _, Xtest, _, ytest = loadDataset(df_test_path, 0, False)
    
    rskf = RepeatedStratifiedKFold(n_splits=2, n_repeats=4, random_state=42)
    theta_best = -1
    acc_best = 0
    
    simil = 'mid'
    sing = 'max'
    
    classifier = AccelBatchGFMM(gamma = 1, teta = 0.5, bthres = 0.9, simil = simil, sing = sing, isDraw = True, oper = 'min', isNorm = False)
    print(classifier.get_params())
    # classifier.fit(Xtr, Xtr, ytr)
    # predicted_class = classifier.predict(Xtest, Xtest, ytest, False, 1)
    # print("Accuracy = ", accuracy_score(ytest, predicted_class), ", No hyperboxes = ", len(classifier.classId), ", training time = ", classifier.elapsed_training_time)
    
    # start_opt_time = time.perf_counter()
    # for theta in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5]:
    #     acc_val = 0
    #     for val_index, train_index in rskf.split(Xtr, ytr):
    #         Xtr_tmp = Xtr[train_index]
    #         ytr_tmp = ytr[train_index]
    #         Xval_tmp = Xtr[val_index]
    #         yval_tmp = ytr[val_index]
    #         # Training model
    #         classifier = AccelBatchGFMM(gamma = gamma, teta = theta, bthres = 0, simil = simil, sing = sing, isDraw = False, oper = 'min', isNorm = False)
                                
    #         classifier.fit(Xtr_tmp, Xtr_tmp, ytr_tmp)
            
    #         predicted_class = classifier.predict(Xval_tmp, Xval_tmp, yval_tmp, True, 0)
    #         tmp_val = accuracy_score(yval_tmp, predicted_class)
    #         acc_val = acc_val + tmp_val
        
    #     acc_val = acc_val / 8
    #     #print("acc_best = ", acc_best, ", acc_val = ", acc_val)
    #     if acc_best < acc_val:
    #         acc_best = acc_val
    #         theta_best = theta
            
    # print("Optimisation time = ", time.perf_counter() - start_opt_time)
            
    # # Training a full model on the training data
    # classifier = AccelBatchGFMM(gamma = gamma, teta = theta_best, bthres = 0, simil = simil, sing = sing, isDraw = False, oper = 'min', isNorm = False)
    # classifier.fit(Xtr, Xtr, ytr)
    # predicted_class = classifier.predict(Xtest, Xtest, ytest, True, 0)
    # print("Best Theta = ", theta_best, ": Accuracy = ", accuracy_score(ytest, predicted_class), ", No hyperboxes = ", len(classifier.classId), ", training time = ", classifier.elapsed_training_time)
        