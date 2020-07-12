# -*- coding: utf-8 -*-
"""
Created on Thu Aug 30 22:39:47 2018

@author: Thanh Tung Khuat

Improved version of Online GFMM classifier (training core). Faster version on datasets with high dimensionality
In this implementation, we do not use the branch and bound lemma

    Compare to previous version, in this coding, we find the hyperboxes representing the same label as the input pattern,
    The membership grades are only computed on those hyperboxes. In constrast, in the previous version, we find membership grades for all current hyperboxes and then filter hyperboxes with the same label as the input pattern
    The normal version runs faster on the dataset with low dimensionality. The faster version runs well on the dataset with high dimensionality

     OnlineGFMM(gamma, teta, tMin, isDraw, oper, V, W, classId, isNorm, norm_range)

   INPUT
     V              Hyperbox lower bounds for the model to be updated using new data
     W              Hyperbox upper bounds for the model to be updated using new data
     classId        Hyperbox class labels (crisp)  for the model to be updated using new data
     gamma          Membership function slope (default: 1), datatype: array or scalar
     teta           Maximum hyperbox size (default: 1)
     tMin           Minimum value of Teta
     isDraw         Progress plot flag (default: False)
     oper           Membership calculation operation: 'min' or 'prod' (default: 'min')
     isNorm         Do normalization of input training samples or not?
     norm_range     New ranging of input data after normalization
"""

import sys, os
sys.path.insert(0, os.path.pardir)
#import os,sys,inspect
#currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
#parentdir = os.path.dirname(currentdir)
#sys.path.insert(0, parentdir)

import ast
import numpy as np
import random
import time
import matplotlib
try:
    matplotlib.use('TkAgg')
except:
    pass

from functionhelper import UNLABELED_CLASS
from functionhelper.membershipcalc import memberG
from functionhelper.hyperboxadjustment import hyperboxOverlapTest, hyperboxContraction, directedIsOverlap
from GFMM.classification import predict_with_probability, predict
from functionhelper.drawinghelper import drawbox
from functionhelper.preprocessinghelper import loadDataset, string_to_boolean
from GFMM.basegfmmclassifier import BaseGFMMClassifier
from sklearn.metrics import accuracy_score, classification_report
from functionhelper.measurehelper import manhattan_distance

class ImprovedOnlineGFMM(BaseGFMMClassifier):

    def __init__(self, gamma = 1, teta = 1, sigma = 0, isDraw = False, oper = 'min', isNorm = False, norm_range = [0, 1], V = np.array([]), W = np.array([]), classId = np.array([])):
        BaseGFMMClassifier.__init__(self, gamma, teta, isDraw, oper, isNorm, norm_range)

        self.V = V
        self.W = W
        self.classId = classId
        self.sigma = sigma      


    def fit(self, X_l, X_u, patClassId, num_pat=None):
        """
        Training the classifier

         Xl             Input data lower bounds (rows = objects, columns = features)
         Xu             Input data upper bounds (rows = objects, columns = features)
         patClassId     Input data class labels (crisp). patClassId[i] = 0 corresponds to an unlabeled item
         num_pat        Save the number of samples in hyperboxes [X_l, X_u]
        """
        #print('--Online Learning--')

        if self.isNorm == True:
            X_l, X_u = self.dataPreprocessing(X_l, X_u)
        
        #X_l = X_l.astype(np.float32)
        #X_u = X_u.astype(np.float32)
        
        time_start = time.perf_counter()

        yX, xX = X_l.shape
        teta = self.teta

        mark = np.array(['*', 'o', 'x', '+', '.', ',', 'v', '^', '<', '>', '1', '2', '3', '4', '8', 's', 'p', 'P', 'h', 'H', 'X', 'D', '|', '_'])
        mark_col = np.array(['r', 'g', 'b', 'y', 'c', 'm', 'k'])

        listLines = list()
        listInputSamplePoints = list()

        if self.isDraw:
            drawing_canvas = self.initializeCanvasGraph("GFMM - Online learning", xX)

            if self.V.size > 0:
                # draw existed hyperboxes
                color_ = np.array(['k'] * len(self.classId), dtype = object)
                for c in range(len(self.classId)):
                    if self.classId[c] < len(mark_col):
                        color_[c] = mark_col[self.classId[c]]

                hyperboxes = drawbox(self.V[:, 0:np.minimum(xX,3)], self.W[:, 0:np.minimum(xX,3)], drawing_canvas, color_)
                listLines.extend(hyperboxes)
                self.delay()

        self.misclass = 1
        threshold = 0 # No using lemma for branch and bound
        # for each input sample
        for i in range(yX):
            classOfX = patClassId[i]
            # draw input samples
            if self.isDraw:
                if i == 0 and len(listInputSamplePoints) > 0:
                    # reset input point drawing
                    for point in listInputSamplePoints:
                        point.remove()
                    listInputSamplePoints.clear()

                color_ = 'k'
                if classOfX < len(mark_col):
                    color_ = mark_col[classOfX]

                if (X_l[i, :] == X_u[i, :]).all():
                    marker_ = 'd'
                    if classOfX < len(mark):
                        marker_ = mark[classOfX]

                    if xX == 2:
                        inputPoint = drawing_canvas.plot(X_l[i, 0], X_l[i, 1], color = color_, marker=marker_)
                    else:
                        inputPoint = drawing_canvas.plot([X_l[i, 0]], [X_l[i, 1]], [X_l[i, 2]], color = color_, marker=marker_)

                    #listInputSamplePoints.append(inputPoint)
                else:
                    inputPoint = drawbox(np.asmatrix(X_l[i, 0:np.minimum(xX, 3)]), np.asmatrix(X_u[i, 0:np.minimum(xX, 3)]), drawing_canvas, color_)

                listInputSamplePoints.append(inputPoint[0])
                self.delay()

            if self.V.size == 0:   # no model provided - starting from scratch
                self.V = np.array([X_l[0]])
                self.W = np.array([X_u[0]])
                self.classId = np.array([patClassId[0]])
                if num_pat is None:
                    self.counter = np.array([1]) # save number of samples of each hyperbox
                else:
                    self.counter = np.array([num_pat[0]])
                    
                if self.isDraw == True:
                    # draw hyperbox
                    box_color = 'k'
                    if patClassId[0] < len(mark_col):
                        box_color = mark_col[patClassId[0]]

                    hyperbox = drawbox(np.asmatrix(self.V[0, 0:np.minimum(xX,3)]), np.asmatrix(self.W[0, 0:np.minimum(xX,3)]), drawing_canvas, box_color)
                    listLines.append(hyperbox[0])
                    self.delay()

            else:
                id_lb_sameX = np.nonzero(((self.classId == classOfX) | (self.classId == UNLABELED_CLASS)))[0]
                V_sameX = self.V[id_lb_sameX]                
                
                if len(V_sameX) > 0: 
                    # if we have small number of hyperboxes with low dimension, this operation takes more time compared to computing membership value with all hyperboxes and ignore
                    # hyperboxes with different class (the membership computation on small dimensionality is so rapidly). However, if we have hyperboxes with high dimensionality, the membership computing on all hyperboxes take so long => reduced to only hyperboxes with the
                    # same class will significantly decrease the running time
                    W_sameX = self.W[id_lb_sameX]
                    lb_sameX = self.classId[id_lb_sameX]

                    b = memberG(X_l[i], X_u[i], V_sameX, W_sameX, self.gamma)
                    index = np.argsort(b)[::-1]
                    
                    if b[index[0]] != 1 or (classOfX != lb_sameX[index[0]] and classOfX != UNLABELED_CLASS):
                        adjust = False
                        
                        id_lb_diff = ((self.classId != classOfX) | (self.classId == UNLABELED_CLASS))
                        V_diff = self.V[id_lb_diff]
                        W_diff = self.W[id_lb_diff]
                        
                        indcomp = np.nonzero((W_diff >= V_diff).all(axis = 1))[0] 	# examine only hyperboxes w/o missing dimensions, meaning that in each dimension upper bound is larger than lowerbound
                        no_check_overlap = False
                        if len(indcomp) == 0 or len(V_diff) == 0:
                            no_check_overlap = True
                        else:
                            V_diff = V_diff_save = V_diff[indcomp]
                            W_diff = W_diff_save = W_diff[indcomp]
                        
                        for j in id_lb_sameX[index]:
                            minV_new = np.minimum(self.V[j], X_l[i])
                            maxW_new = np.maximum(self.W[j], X_u[i])
                            
                            # test violation of max hyperbox size and class labels
                            if ((maxW_new - minV_new) <= teta).all() == True:
                                if no_check_overlap == False and classOfX == UNLABELED_CLASS and self.classId[j] == UNLABELED_CLASS:
                                    # remove hyperbox themself
                                    keep_id = (V_diff != self.V[j]).any(1)
                                    V_diff = V_diff[keep_id]
                                    W_diff = W_diff[keep_id]
                                # Test overlap    
                                if no_check_overlap == True or directedIsOverlap(V_diff, W_diff, minV_new, maxW_new) == False:		# overlap test
                                    # adjust the j-th hyperbox
                                    self.V[j] = minV_new
                                    self.W[j] = maxW_new
                                    if num_pat is None:
                                        self.counter[j] = self.counter[j] + 1
                                    else:
                                        self.counter[j] = self.counter[j] + num_pat[i]
                                    
                                    if classOfX != UNLABELED_CLASS and self.classId[j] == UNLABELED_CLASS:
                                        self.classId[j] = classOfX
                                    
                                    if self.isDraw:
                                        # Handle drawing graph
                                        box_color = 'k'
                                        if self.classId[j] < len(mark_col):
                                            box_color = mark_col[self.classId[j]]
    
                                        try:
                                            listLines[j].remove()
                                        except:
                                            pass
    
                                        hyperbox = drawbox(np.asmatrix(self.V[j, 0:np.minimum(xX, 3)]), np.asmatrix(self.W[j, 0:np.minimum(xX, 3)]), drawing_canvas, box_color)
                                        listLines[j] = hyperbox[0]
                                        self.delay()
                                    
                                    adjust = True
                                    break
                                else:
                                    if no_check_overlap == False and classOfX == UNLABELED_CLASS and self.classId[j] == UNLABELED_CLASS:                                   
                                        V_diff = V_diff_save
                                        W_diff = W_diff_save
                                   
    
                        # if i-th sample did not fit into any existing box, create a new one
                        if not adjust:
                            self.V = np.concatenate((self.V, X_l[i].reshape(1, -1)), axis = 0)
                            self.W = np.concatenate((self.W, X_u[i].reshape(1, -1)), axis = 0)
                            self.classId = np.concatenate((self.classId, [classOfX]))
                            if num_pat is None:
                                self.counter = np.concatenate((self.counter, [1]))
                            else:
                                self.counter = np.concatenate((self.counter, [num_pat[i]]))
    
                            if self.isDraw:
                                # handle drawing graph
                                box_color = 'k'
                                if self.classId[-1] < len(mark_col):
                                    box_color = mark_col[self.classId[-1]]
    
                                hyperbox = drawbox(np.asmatrix(X_l[i, 0:np.minimum(xX, 3)]), np.asmatrix(X_u[i, 0:np.minimum(xX, 3)]), drawing_canvas, box_color)
                                listLines.append(hyperbox[0])
                                self.delay()
					else:
						t = 0
                        while (t + 1 < len(index)) and (b[index[t]] == 1) and (self.classId[id_lb_sameX[index[t]]] != classOfX):
                            t = t + 1
                        if b[index[t]] == 1 and self.classId[id_lb_sameX[index[t]]] == classOfX:
                            if num_pat is None:
                                self.counter[id_lb_sameX[index[t]]] = self.counter[id_lb_sameX[index[t]]] + 1
                            else:
                                self.counter[id_lb_sameX[index[t]]] = self.counter[id_lb_sameX[index[t]]] + num_pat[i]
                else:
                    self.V = np.concatenate((self.V, X_l[i].reshape(1, -1)), axis = 0)
                    self.W = np.concatenate((self.W, X_u[i].reshape(1, -1)), axis = 0)
                    self.classId = np.concatenate((self.classId, [classOfX]))
                    
                    if num_pat is None:
                        self.counter = np.concatenate((self.counter, [1]))
                    else:
                        self.counter = np.concatenate((self.counter, [num_pat[i]]))

                    if self.isDraw:
                        # handle drawing graph
                        box_color = 'k'
                        if self.classId[-1] < len(mark_col):
                            box_color = mark_col[self.classId[-1]]

                        hyperbox = drawbox(np.asmatrix(X_l[i, 0:np.minimum(xX, 3)]), np.asmatrix(X_u[i, 0:np.minimum(xX, 3)]), drawing_canvas, box_color)
                        listLines.append(hyperbox[0])
                        self.delay()
        
               
        time_end = time.perf_counter()
        self.elapsed_training_time = time_end - time_start

        return self
    
    
    def predict(self, Xl_Test, Xu_Test, patClassIdTest, newVer = True):
        """
        Perform classification

            result = predict(Xl_Test, Xu_Test, patClassIdTest)

        INPUT:
            Xl_Test             Test data lower bounds (rows = objects, columns = features)
            Xu_Test             Test data upper bounds (rows = objects, columns = features)
            patClassIdTest	     Test data class labels (crisp)
            newVer              + False: Don't use an additional criterion for predicting
                                + True : Using an additional criterion for predicting in the case of the same membership value

        OUTPUT:
            result        A object with Bunch datatype containing all results as follows:
                          + summis           Number of misclassified objects
                          + misclass         Binary error map
                          + numSampleInBoundary     The number of samples in decision boundary
                          + predicted_class   Predicted class
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

        if Xl_Test.shape[0] > 0:
            if newVer:
                result = predict_with_probability(self.V, self.W, self.classId, self.counter, Xl_Test, Xu_Test, patClassIdTest, self.gamma, self.oper)
            else:
                result = predict(self.V, self.W, self.classId, Xl_Test, Xu_Test, patClassIdTest, self.gamma, self.oper)
                
            self.predicted_class = np.array(result.predicted_class, np.int)

        return result
    
    def pruning_val(self, XlT, XuT, patClassIdTest, accuracy_threshold = 0.5, newVerPredict = True):
        """
        pruning handling based on validation (validation routine) with hyperboxes stored in self. V, W, classId
    
          result = pruning_val(XlT,XuT,patClassIdTest)
    
            INPUT
              XlT               Test data lower bounds (rows = objects, columns = features)
              XuT               Test data upper bounds (rows = objects, columns = features)
              patClassIdTest    Test data class labels (crisp)
              accuracy_threshold  The minimum accuracy for each hyperbox
              newVerPredict     + True: using probability formula for prediction in addition to fuzzy membership
                                + False: No using probability formula for prediction
        """
    
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
                        is_find_prob_val = True
                        if bmax == 1:
                            id_box_with_one_sample = np.nonzero(self.counter[maxVind] == 1)[0]
                            if len(id_box_with_one_sample) > 0:
                                is_find_prob_val = False
                                id_min = random.choice(maxVind[id_box_with_one_sample])
                        
                        if is_find_prob_val == True:
                            sum_prod_denum = (mem[maxVind] * self.counter[maxVind]).sum()
                            max_prob = -1
                            pre_id_cls = None
                            for c in cls_same_mem:
                                id_cls = np.nonzero(self.classId[maxVind] == c)[0]
                                sum_pro_num = (mem[maxVind[id_cls]] * self.counter[maxVind[id_cls]]).sum()
                                tmp = sum_pro_num / sum_prod_denum
                                
                                if tmp > max_prob or (tmp == max_prob and pre_id_cls is not None and self.counter[maxVind[id_cls]].sum() > self.counter[maxVind[pre_id_cls]].sum()):
                                    max_prob = tmp
                                    pre_id_cls = id_cls
                                    id_min = random.choice(maxVind[id_cls])
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
        
        # keep one hyperbox for class prunned all
        current_classes = np.unique(self.classId)
        class_tmp = self.classId[accuracy_larger_half]
        class_tmp_keep = self.classId[accuracy_larger_half_keep_nojoin]
        for c in current_classes:
            if c not in class_tmp:
                pos = np.nonzero(self.classId == c)
                id_kept = np.random.randint(len(pos))
                # keep pos[id_kept]
                accuracy_larger_half[pos[id_kept]] = True
            if c not in class_tmp_keep:
                pos = np.nonzero(self.classId == c)
                id_kept = np.random.randint(len(pos))
                accuracy_larger_half_keep_nojoin[pos[id_kept]] = True
        
        V_prun_remove = self.V[accuracy_larger_half]
        W_prun_remove = self.W[accuracy_larger_half]
        classId_prun_remove = self.classId[accuracy_larger_half]
        numSample_prun_remove = self.counter[accuracy_larger_half]
        
        W_prun_keep = self.W[accuracy_larger_half_keep_nojoin]
        V_prun_keep = self.V[accuracy_larger_half_keep_nojoin]
        
        classId_prun_keep = self.classId[accuracy_larger_half_keep_nojoin]
        numSample_prun_keep = self.classId[accuracy_larger_half_keep_nojoin]
        
        if newVerPredict == True:
            result_prun_remove = predict_with_probability(V_prun_remove, W_prun_remove, classId_prun_remove, numSample_prun_remove, XlT, XuT, patClassIdTest, self.gamma, self.oper)
            result_prun_keep_nojoin = predict_with_probability(V_prun_keep, W_prun_keep, classId_prun_keep, numSample_prun_keep, XlT, XuT, patClassIdTest, self.gamma, self.oper)
        else:
            result_prun_remove = predict(V_prun_remove, W_prun_remove, classId_prun_remove, XlT, XuT, patClassIdTest, self.gamma, self.oper)
            result_prun_keep_nojoin = predict(V_prun_keep, W_prun_keep, classId_prun_keep, XlT, XuT, patClassIdTest, self.gamma, self.oper)
        
        if (result_prun_remove.summis <= result_prun_keep_nojoin.summis):
            self.V = V_prun_remove
            self.W = W_prun_remove
            self.classId = classId_prun_remove
            self.counter = numSample_prun_remove
        else:
            self.V = V_prun_keep
            self.W = W_prun_keep
            self.classId = classId_prun_keep
            self.counter = numSample_prun_keep


if __name__ == '__main__':
    """
    INPUT parameters from command line
    arg1: + 1 - training and testing datasets are located in separated files
          + 2 - training and testing datasets are located in the same files
    arg2: path to file containing the training dataset (arg1 = 1) or both training and testing datasets (arg1 = 2)
    arg3: + path to file containing the testing dataset (arg1 = 1)
          + percentage of the training dataset in the input file
    arg4: + path to file containing the validation dataset
    arg5: + True: drawing hyperboxes during the training process
          + False: no drawing
    arg6: + Maximum size of hyperboxes (teta, default: 1)
    arg7: + gamma value (default: 1)
    arg8: + The minimum value of membership to expand the current hyperbox (sigma: default = 1 - teta x gamma)
    arg9: operation used to compute membership value: 'min' or 'prod' (default: 'min')
    arg10: + do normalization of datasets or not? True: Normilize, False: No normalize (default: True)
    arg11: + range of input values after normalization (default: [0, 1])
    """
    # Init default parameters
#    if len(sys.argv) < 6:
#        isDraw = False
#    else:
#        isDraw = string_to_boolean(sys.argv[5])
#
#    if len(sys.argv) < 7:
#        teta = 1
#    else:
#        teta = float(sys.argv[6])
#
#    if len(sys.argv) < 8:
#        gamma = 1
#    else:
#        gamma = float(sys.argv[7])
#        
#    if len(sys.argv) < 9:
#        sigma = 1 - teta * gamma
#    else:
#        sigma = float(sys.argv[8])
#
#    if len(sys.argv) < 10:
#        oper = 'min'
#    else:
#        oper = sys.argv[9]
#
#    if len(sys.argv) < 11:
#        isNorm = True
#    else:
#        isNorm = string_to_boolean(sys.argv[10])
#
#    if len(sys.argv) < 12:
#        norm_range = [0, 1]
#    else:
#        norm_range = ast.literal_eval(sys.argv[11])
#
#    # print('isDraw = ', isDraw, ' teta = ', teta, ' teta_min = ', teta_min, ' gamma = ', gamma, ' oper = ', oper, ' isNorm = ', isNorm, ' norm_range = ', norm_range)
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
    training_file = "C:\\Hyperbox-based-ML\\Dataset\\train_test\\training_testing_data\\spambase_dps_tr.dat"
    testing_file = "C:\\Hyperbox-based-ML\\Dataset\\train_test\\training_testing_data\\spambase_dps_test.dat"
    validation_file = "C:\\Hyperbox-based-ML\\Dataset\\train_test\\training_testing_data\\spambase_dps_val.dat"
    gamma = 1
    teta = 0.1
    sigma = 1 - teta * gamma
    isDraw = False
    oper = 'min'
    isNorm = False
    norm_range = [0, 1]
    # Read training file
    Xtr, X_tmp, patClassIdTr, pat_tmp = loadDataset(training_file, 1, False)
    # Read testing file
    X_tmp, Xtest, pat_tmp, patClassIdTest = loadDataset(testing_file, 0, False)
    
    classifier = ImprovedOnlineGFMM(gamma, teta, sigma, isDraw, oper, isNorm, norm_range)
    classifier.fit(Xtr, Xtr, patClassIdTr)
    print("Before pruning:")
    print('V size = ', classifier.V.shape)
    print('W size = ', classifier.W.shape)
    if isPruning == True:
        #classifier.pruning_val(Xval, Xval, patClassIdVal)
        print("After pruning:")
        print('V size = ', classifier.V.shape)
        print('W size = ', classifier.W.shape)
    end_t = time.perf_counter()
    
    #print("Reading file + Training and pruning Time = ", end_t - start_t)
    
    # Testing
    print("-- Testing --")
    result = classifier.predict(Xtest, Xtest, patClassIdTest)
    if result != None:
        print("Number of wrong predicted samples = ", result.summis)
        numTestSample = Xtest.shape[0]
        print("Error Rate = ", np.round(result.summis / numTestSample * 100, 2), "%")
        predicted_class = np.array(result.predicted_class, np.int64)
        print(classification_report(patClassIdTest, predicted_class))

#    print("-- Testing on Validation file --")
#    result = classifier.predict(Xval, patClassIdVal)
#    if result != None:
#        print("Number of wrong predicted samples = ", result.summis)
#        numTestSample = Xval.shape[0]
#        print("Error Rate = ", np.round(result.summis / numTestSample * 100, 2), "%")