# -*- coding: utf-8 -*-
"""
Created on Mon Sep  3 11:22:08 2018

@author: Thanh Tung Khuat

GFMM Predictor

"""

import numpy as np
import random
from collections import Counter
import operator
from functionhelper.measurehelper import manhattan_distance, min_distance, manhattan_distance_with_missing_value
from functionhelper.membershipcalc import memberG
from functionhelper.bunchdatatype import Bunch
from functionhelper import UNLABELED_CLASS

def predict(V, W, classId, XlT, XuT, patClassIdTest, gama = 1, oper = 'min'):
    """
    GFMM classifier (test routine)

      result = predict(V,W,classId,XlT,XuT,patClassIdTest,gama,oper)

    INPUT
      V                 Tested model hyperbox lower bounds
      W                 Tested model hyperbox upper bounds
      classId	          Input data (hyperbox) class labels (crisp)
      XlT               Test data lower bounds (rows = objects, columns = features)
      XuT               Test data upper bounds (rows = objects, columns = features)
      patClassIdTest    Test data class labels (crisp)
      gama              Membership function slope (default: 1)
      oper              Membership calculation operation: 'min' or 'prod' (default: 'min')

   OUTPUT
      result           A object with Bunch datatype containing all results as follows:
                          + summis           Number of misclassified objects
                          + misclass         Binary error map
                          + predicted_class   Predicted class

    """
    
    if len(XlT.shape) == 1:
        XlT = XlT.reshape(1, -1)
    if len(XuT.shape) == 1:
        XuT = XuT.reshape(1, -1)
        
    #initialization        
    yX = XlT.shape[0]
    misclass = np.zeros(yX)
    predicted_class = np.full(yX, None)
    # classifications
    for i in range(yX):
        mem = memberG(XlT[i, :], XuT[i, :], V, W, gama, oper) # calculate memberships for all hyperboxes
        bmax = mem.max()	                                          # get max membership value
        maxVind = np.nonzero(mem == bmax)[0]                         # get indexes of all hyperboxes with max membership

        if bmax == 0:
            predicted_class[i] = classId[maxVind[0]]
            if predicted_class[i] == patClassIdTest[i]:
                misclass[i] = False
            else:
                misclass[i] = True
        else:
            if len(np.unique(classId[maxVind])) > 1:
                #print('Input is in the boundary')
                misclass[i] = True
            else:
                predicted_class[i] = classId[maxVind[0]]
                if np.any(classId[maxVind] == patClassIdTest[i]) == True or patClassIdTest[i] == UNLABELED_CLASS:
                    misclass[i] = False
                else:
                    misclass[i] = True
                #misclass[i] = ~(np.any(classId[maxVind] == patClassIdTest[i]) | (patClassIdTest[i] == 0))

    # results
    summis = np.sum(misclass).astype(np.int64)

    result = Bunch(summis = summis, misclass = misclass, predicted_class=predicted_class)
    return result

def predict_with_manhattan(V, W, classId, XlT, XuT, patClassIdTest, gama = 1, oper = 'min'):
    """
    GFMM classifier (test routine): Using Manhattan distance in the case of many hyperboxes with different classes having the same maximum membership value

      result = predict(V,W,classId,XlT,XuT,patClassIdTest,gama,oper)

    INPUT
      V                 Tested model hyperbox lower bounds
      W                 Tested model hyperbox upper bounds
      classId	          Input data (hyperbox) class labels (crisp)
      XlT               Test data lower bounds (rows = objects, columns = features)
      XuT               Test data upper bounds (rows = objects, columns = features)
      patClassIdTest    Test data class labels (crisp)
      gama              Membership function slope (default: 1)
      oper              Membership calculation operation: 'min' or 'prod' (default: 'min')

   OUTPUT
      result           A object with Bunch datatype containing all results as follows:
                          + summis           Number of misclassified objects
                          + misclass         Binary error map
                          + numSampleInBoundary     The number of samples in decision boundary
                          + predicted_class   Predicted class

    """
    if len(XlT.shape) == 1:
        XlT = XlT.reshape(1, -1)
    if len(XuT.shape) == 1:
        XuT = XuT.reshape(1, -1)
        
    #initialization
    yX = XlT.shape[0]
    misclass = np.zeros(yX)
    mem_vals = np.zeros(yX)
    numPointInBoundary = 0
    predicted_class = np.full(yX, None)
    # classifications
    for i in range(yX):
        if patClassIdTest[i] == UNLABELED_CLASS:
            misclass[i] = False
        else:          
            mem = memberG(XlT[i, :], XuT[i, :], V, W, gama, oper) # calculate memberships for all hyperboxes
            bmax = mem.max()	                                          # get max membership value
            maxVind = np.nonzero(mem == bmax)[0]                         # get indexes of all hyperboxes with max membership
            mem_vals[i] = bmax
            
#            if bmax == 0:
#                predicted_class[i] = classId[maxVind[0]]
#                if predicted_class[i] == patClassIdTest[i]:
#                    misclass[i] = False
#                else:
#                    misclass[i] = True
#            else:
            if len(np.unique(classId[maxVind])) > 1:
                numPointInBoundary = numPointInBoundary + 1
                #print("Using Manhattan function")
                if (XlT[i] == XuT[i]).all() == False:
                    XlT_mat = np.ones((len(maxVind), 1)) * XlT[i]
                    XuT_mat = np.ones((len(maxVind), 1)) * XuT[i]
                    XgT_mat = (XlT_mat + XuT_mat) / 2
                else:
                    XgT_mat = np.ones((len(maxVind), 1)) * XlT[i]
                # Find all average points of all hyperboxes with the same membership value
                avg_point_mat = (V[maxVind] + W[maxVind]) / 2
                # compute the manhattan distance from XgT_mat to all average points of all hyperboxes with the same membership value
                maht_dist = manhattan_distance(avg_point_mat, XgT_mat)
                #maht_dist = min_distance(avg_point_mat, XgT_mat)
                id_min_dist = maht_dist.argmin()
                
                predicted_class[i] = classId[maxVind[id_min_dist]]
                if classId[maxVind[id_min_dist]] == patClassIdTest[i]:
                    misclass[i] = False
                else:
                    misclass[i] = True
            else:
                predicted_class[i] = classId[maxVind[0]]
                if classId[maxVind[0]] == patClassIdTest[i]:
                    misclass[i] = False
                else:
                    misclass[i] = True
                    #misclass[i] = ~(np.any(classId[maxVind] == patClassIdTest[i]) | (patClassIdTest[i] == 0))

    # results
    summis = np.sum(misclass).astype(np.int64)

    result = Bunch(summis = summis, misclass = misclass, numSampleInBoundary = numPointInBoundary, predicted_class=predicted_class, mem_vals=mem_vals)
    
    return result

def predict_with_manhattan_and_missing_values(V, W, classId, XlT, XuT, patClassIdTest, gama = 1, oper = 'min'):
    """
    GFMM classifier (test routine): Using Manhattan distance in the case of many hyperboxes with different classes having the same maximum membership value

      result = predict(V,W,classId,XlT,XuT,patClassIdTest,gama,oper)

    INPUT
      V                 Tested model hyperbox lower bounds
      W                 Tested model hyperbox upper bounds
      classId	          Input data (hyperbox) class labels (crisp)
      XlT               Test data lower bounds (rows = objects, columns = features)
      XuT               Test data upper bounds (rows = objects, columns = features)
      patClassIdTest    Test data class labels (crisp)
      gama              Membership function slope (default: 1)
      oper              Membership calculation operation: 'min' or 'prod' (default: 'min')

   OUTPUT
      result           A object with Bunch datatype containing all results as follows:
                          + summis           Number of misclassified objects
                          + misclass         Binary error map
                          + numSampleInBoundary     The number of samples in decision boundary
                          + predicted_class   Predicted class

    """
    if len(XlT.shape) == 1:
        XlT = XlT.reshape(1, -1)
    if len(XuT.shape) == 1:
        XuT = XuT.reshape(1, -1)
        
    #initialization
    yX = XlT.shape[0]
    misclass = np.zeros(yX)
    numPointInBoundary = 0
    predicted_class = np.full(yX, None)
    # classifications
    for i in range(yX):
        if patClassIdTest[i] == UNLABELED_CLASS:
            misclass[i] = False
        else:          
            mem = memberG(XlT[i, :], XuT[i, :], V, W, gama, oper) # calculate memberships for all hyperboxes
            bmax = mem.max()	                                          # get max membership value
            maxVind = np.nonzero(mem == bmax)[0]                         # get indexes of all hyperboxes with max membership
    
#            if bmax == 0:
#                predicted_class[i] = classId[maxVind[0]]
#                if predicted_class[i] == patClassIdTest[i]:
#                    misclass[i] = False
#                else:
#                    misclass[i] = True
#            else:
            if len(np.unique(classId[maxVind])) > 1:
                numPointInBoundary = numPointInBoundary + 1
                #print("Using Manhattan function")
                if (XlT[i] == XuT[i]).all() == False:
                    XlT_mat = np.ones((len(maxVind), 1)) * XlT[i]
                    XuT_mat = np.ones((len(maxVind), 1)) * XuT[i]
                    XgT_mat = (XlT_mat + XuT_mat) / 2
                else:
                    XgT_mat = np.ones((len(maxVind), 1)) * XlT[i]
                    
                # compute the manhattan distance from XgT_mat to all average points of all hyperboxes with the same membership value
                maht_dist = manhattan_distance_with_missing_value(V[maxVind], W[maxVind], XgT_mat)
                id_min_dist = maht_dist.argmin()
                
                predicted_class[i] = classId[maxVind[id_min_dist]]
                if classId[maxVind[id_min_dist]] == patClassIdTest[i]:
                    misclass[i] = False
                else:
                    misclass[i] = True
            else:
                predicted_class[i] = classId[maxVind[0]]
                if classId[maxVind[0]] == patClassIdTest[i]:
                    misclass[i] = False
                else:
                    misclass[i] = True
                    #misclass[i] = ~(np.any(classId[maxVind] == patClassIdTest[i]) | (patClassIdTest[i] == 0))

    # results
    summis = np.sum(misclass).astype(np.int64)

    result = Bunch(summis = summis, misclass = misclass, numSampleInBoundary = numPointInBoundary, predicted_class=predicted_class)
    
    return result

def predict_with_probability(V, W, classId, numSamples, XlT, XuT, patClassIdTest, gama = 1, oper = 'min'):
    """
    GFMM classifier (test routine): Using probability formular based on the number of samples in the case of many hyperboxes with different classes having the same maximum membership value

      result = predict(V,W,classId,XlT,XuT,patClassIdTest,gama,oper)

    INPUT
      V                 Tested model hyperbox lower bounds
      W                 Tested model hyperbox upper bounds
      classId	        Input data (hyperbox) class labels (crisp)
      numSamples        Save number of samples of each corresponding hyperboxes contained in V and W
      XlT               Test data lower bounds (rows = objects, columns = features)
      XuT               Test data upper bounds (rows = objects, columns = features)
      patClassIdTest    Test data class labels (crisp)
      gama              Membership function slope (default: 1)
      oper              Membership calculation operation: 'min' or 'prod' (default: 'min')

   OUTPUT
      result           A object with Bunch datatype containing all results as follows:
                          + summis           Number of misclassified objects
                          + misclass         Binary error map
                          + numSampleInBoundary     The number of samples in decision boundary
                          + predicted_class   Predicted class
                          + id_winner         Index of hyperbox used for making prediction for each sample

    """
    if len(XlT.shape) == 1:
        XlT = XlT.reshape(1, -1)
    if len(XuT.shape) == 1:
        XuT = XuT.reshape(1, -1)
        
    #initialization
    yX = XlT.shape[0]
    misclass = np.zeros(yX)
    predicted_class = np.full(yX, None)
    mem_vals = np.zeros(yX)
    id_winner = np.zeros(yX, dtype=np.int64)
    # classifications
    numPointInBoundary = 0
    for i in range(yX):
        if patClassIdTest[i] == UNLABELED_CLASS:
            misclass[i] = False
            id_winner[i] = 0
        else:          
            mem = memberG(XlT[i, :], XuT[i, :], V, W, gama, oper) # calculate memberships for all hyperboxes
            bmax = mem.max()	                                          # get max membership value
            maxVind = np.nonzero(mem == bmax)[0]                         # get indexes of all hyperboxes with max membership
            mem_vals[i] = bmax
#            if bmax == 0:
#                #print('zero maximum membership value')                     # this is probably bad...
#                predicted_class[i] = classId[maxVind[0]]
#                id_winner[i] = maxVind[0]
#                if predicted_class[i] == patClassIdTest[i]:
#                    misclass[i] = False
#                else:
#                    misclass[i] = True
#            else:
            cls_same_mem = np.unique(classId[maxVind])
            if len(cls_same_mem) > 1:
                cls_val = UNLABELED_CLASS
                
                is_find_prob_val = True
                if bmax == 1:
                    id_box_with_one_sample = np.nonzero(numSamples[maxVind] == 1)[0]
                    if len(id_box_with_one_sample) > 0:
                        is_find_prob_val = False
                        id_winner[i] = random.choice(maxVind[id_box_with_one_sample])
                        cls_val = classId[id_winner[i]]
                        
                if is_find_prob_val == True:
                    numPointInBoundary = numPointInBoundary + 1
                    #print('bmax=', bmax)
                    #print("Using probability function")
                    sum_prod_denum = (mem[maxVind] * numSamples[maxVind]).sum()
                    max_prob = -1
                    pre_id_cls = None
                    for c in cls_same_mem:
                        id_cls = np.nonzero(classId[maxVind] == c)[0]
                        sum_pro_num = (mem[maxVind[id_cls]] * numSamples[maxVind[id_cls]]).sum()
                        tmp = sum_pro_num / sum_prod_denum
                        
                        if tmp > max_prob or (tmp == max_prob and pre_id_cls is not None and numSamples[maxVind[id_cls]].sum() > numSamples[maxVind[pre_id_cls]].sum()):
                            max_prob = tmp
                            cls_val = c
                            pre_id_cls = id_cls
                            id_winner[i] = maxVind[id_cls[0]]
              
                predicted_class[i] = cls_val
                if cls_val == patClassIdTest[i]:
                    misclass[i] = False
                else:
                    misclass[i] = True
            else:
                predicted_class[i] = classId[maxVind[0]]
                id_winner[i] = maxVind[0]
                if predicted_class[i] == patClassIdTest[i]:
                    misclass[i] = False
                else:
                    misclass[i] = True
                    #misclass[i] = ~(np.any(classId[maxVind] == patClassIdTest[i]) | (patClassIdTest[i] == 0))

    #print(numPointInBoundary)
    # results
    summis = np.sum(misclass).astype(np.int64)

    result = Bunch(summis = summis, misclass = misclass, numSampleInBoundary = numPointInBoundary, predicted_class=predicted_class, mem_vals=mem_vals, id_winner=id_winner)
    
    return result

def predict_with_probability_weighted(V, W, classId, numSamples, weights, XlT, XuT, patClassIdTest, gama = 1, oper = 'min'):
    """
    GFMM classifier (test routine): Using probability formular based on the number of samples in the case of many hyperboxes with different classes having the same maximum membership value

      result = predict(V,W,classId,XlT,XuT,patClassIdTest,gama,oper)

    INPUT
      V                 Tested model hyperbox lower bounds
      W                 Tested model hyperbox upper bounds
      classId	        Input data (hyperbox) class labels (crisp)
      numSamples        Save number of samples of each corresponding hyperboxes contained in V and W
      weights           The weights of hyperboxes
      XlT               Test data lower bounds (rows = objects, columns = features)
      XuT               Test data upper bounds (rows = objects, columns = features)
      patClassIdTest    Test data class labels (crisp)
      gama              Membership function slope (default: 1)
      oper              Membership calculation operation: 'min' or 'prod' (default: 'min')

   OUTPUT
      result           A object with Bunch datatype containing all results as follows:
                          + summis           Number of misclassified objects
                          + misclass         Binary error map
                          + numSampleInBoundary     The number of samples in decision boundary
                          + predicted_class   Predicted class

    """
    if len(XlT.shape) == 1:
        XlT = XlT.reshape(1, -1)
    if len(XuT.shape) == 1:
        XuT = XuT.reshape(1, -1)
        
    #initialization
    yX = XlT.shape[0]
    misclass = np.zeros(yX)
    predicted_class = np.full(yX, None)
    mem_vals = np.zeros(yX)
    # classifications
    numPointInBoundary = 0
    for i in range(yX):
        if patClassIdTest[i] == UNLABELED_CLASS:
            misclass[i] = False
        else:          
            mem = memberG(XlT[i, :], XuT[i, :], V, W, gama, oper) # calculate memberships for all hyperboxes
            mem = mem * weights
            bmax = mem.max()	                                          # get max membership value
            maxVind = np.nonzero(mem == bmax)[0]                         # get indexes of all hyperboxes with max membership
            mem_vals[i] = bmax
#            if bmax == 0:
#                #print('zero maximum membership value')                     # this is probably bad...
#                predicted_class[i] = classId[maxVind[0]]
#                if predicted_class[i] == patClassIdTest[i]:
#                    misclass[i] = False
#                else:
#                    misclass[i] = True
#            else:
            cls_same_mem = np.unique(classId[maxVind])
            if len(cls_same_mem) > 1:
                cls_val = UNLABELED_CLASS
                
                is_find_prob_val = True
                if bmax == 1:
                    id_box_with_one_sample = np.nonzero(numSamples[maxVind] == 1)[0]
                    if len(id_box_with_one_sample) > 0:
                        is_find_prob_val = False
                        cls_val = classId[int(random.choice(maxVind[id_box_with_one_sample]))]
                
                if is_find_prob_val == True:
                    numPointInBoundary = numPointInBoundary + 1
                    #print('bmax=', bmax)
                    #print("Using probability function")
                    sum_prod_denum = (mem[maxVind] * numSamples[maxVind]).sum()
                    max_prob = -1
                    pre_id_cls = None
                    for c in cls_same_mem:
                        id_cls = np.nonzero(classId[maxVind] == c)[0]
                        sum_pro_num = (mem[maxVind[id_cls]] * numSamples[maxVind[id_cls]]).sum()
                        tmp = sum_pro_num / sum_prod_denum
                        
                        if tmp > max_prob or (tmp == max_prob and pre_id_cls is not None and numSamples[maxVind[id_cls]].sum() > numSamples[maxVind[pre_id_cls]].sum()):
                            max_prob = tmp
                            cls_val = c
                            pre_id_cls = id_cls
                    
                predicted_class[i] = cls_val
                if cls_val == patClassIdTest[i]:
                    misclass[i] = False
                else:
                    misclass[i] = True
            else:
                predicted_class[i] = classId[maxVind[0]]
                if predicted_class[i] == patClassIdTest[i]:
                    misclass[i] = False
                else:
                    misclass[i] = True
                    #misclass[i] = ~(np.any(classId[maxVind] == patClassIdTest[i]) | (patClassIdTest[i] == 0))

    #print(numPointInBoundary)
    # results
    summis = np.sum(misclass).astype(np.int64)

    result = Bunch(summis = summis, misclass = misclass, numSampleInBoundary = numPointInBoundary, predicted_class=predicted_class, mem_vals=mem_vals)
    
    return result

def predict_with_probability_k_voting_new(V, W, classId, weights, XlT, XuT, patClassIdTest, K_threshold = 5, gama = 1, oper = 'min'):
    """
    GFMM classifier (test routine): Using K voting of values in weights for K hyperboxes with the highest membership values

      result = predict(V,W,classId,XlT,XuT,patClassIdTest,gama,oper)

    INPUT
      V                 Tested model hyperbox lower bounds
      W                 Tested model hyperbox upper bounds
      classId	        Input data (hyperbox) class labels (crisp)
      numSamples        Save number of samples of each corresponding hyperboxes contained in V and W
      weights           The weights of hyperboxes
      XlT               Test data lower bounds (rows = objects, columns = features)
      XuT               Test data upper bounds (rows = objects, columns = features)
      patClassIdTest    Test data class labels (crisp)
      gama              Membership function slope (default: 1)
      oper              Membership calculation operation: 'min' or 'prod' (default: 'min')

   OUTPUT
      result           A object with Bunch datatype containing all results as follows:
                          + summis           Number of misclassified objects
                          + misclass         Binary error map
                          + predicted_class   Predicted class

    """
    if len(XlT.shape) == 1:
        XlT = XlT.reshape(1, -1)
    if len(XuT.shape) == 1:
        XuT = XuT.reshape(1, -1)
        
    #initialization
    yX = XlT.shape[0]
    misclass = np.zeros(yX)
    predicted_class = np.full(yX, None)
    # classifications
    for i in range(yX):
        if patClassIdTest[i] == UNLABELED_CLASS:
            misclass[i] = False
        else:          
            mem = memberG(XlT[i, :], XuT[i, :], V, W, gama, oper) # calculate memberships for all hyperboxes
            mem = mem * weights
            sort_id_mem = np.argsort(mem)[::-1]
            selected_id = sort_id_mem[:K_threshold]
            selected_cls = np.unique(classId[selected_id])
            
            if len(selected_cls) == 1:
                predicted_class[i] = selected_cls[0]
                if predicted_class[i] == patClassIdTest[i]:
                    misclass[i] = False
                else:
                    misclass[i] = True
            else:
                # voting based on sum of weights
                max_prob = -1
                max_mem_sum = -1
                for c in selected_cls:
                    id_cls = classId[selected_id] == c
                    cur_prob = np.sum(mem[selected_id[id_cls]])
                    cur_mem = np.max(weights[selected_id[id_cls]])
                    
                    if max_prob < cur_prob:
                        max_prob = cur_prob
                        predicted_class[i] = c
                        max_mem_sum = cur_mem
                    else:
                        if max_prob == cur_prob and max_mem_sum < cur_mem:
                            max_prob = cur_prob
                            predicted_class[i] = c
                            max_mem_sum = cur_mem
                            
                if predicted_class[i] == patClassIdTest[i]:
                    misclass[i] = False
                else:
                    misclass[i] = True
                    
                    #misclass[i] = ~(np.any(classId[maxVind] == patClassIdTest[i]) | (patClassIdTest[i] == 0))

    #print(numPointInBoundary)
    # results
    summis = np.sum(misclass).astype(np.int64)

    result = Bunch(summis = summis, misclass = misclass, predicted_class=predicted_class)
    
    return result

def predictDecisionLevelEnsemble(classifiers, XlT, XuT, patClassIdTest, gama = 1, oper = 'min'):
    """
    Perform classification for a decision level ensemble learning

                result = predictDecisionLevelEnsemble(classifiers, XlT, XuT, patClassIdTest, gama, oper)

    INPUT
        classifiers         An array of classifiers needed to combine, datatype of each element in the array is BaseGFMMClassifier
        XlT                 Test data lower bounds (rows = objects, columns = features)
        XuT                 Test data upper bounds (rows = objects, columns = features)
        patClassIdTest      Test data class labels (crisp)
        gama                Membership function slope (default: 1)
        oper                Membership calculation operation: 'min' or 'prod' (default: 'min')

    OUTPUT
        result              A object with Bunch datatype containing all results as follows:
                                + summis        Number of misclassified samples
                                + misclass      Binary error map for input samples
                                + out           Soft class memberships, rows are testing input patterns, columns are indices of classes
                                + classes       Store class labels corresponding column indices of out
    """
    numClassifier = len(classifiers)

    yX = XlT.shape[0]
    misclass = np.zeros(yX, dtype=np.bool)
    # get all class labels of all base classifiers
    classId = classifiers[0].classId
    for i in range(numClassifier):
        if i != 0:
            classId = np.union1d(classId, classifiers[i].classId)

    classes = np.unique(classId)
    noClasses = len(classes)
    out = np.zeros((yX, noClasses), dtype=np.float64)

    # classification of each testing pattern i
    for i in range(yX):
        for idClf in range(numClassifier):
            # calculate memberships for all hyperboxes of classifier idClf
            mem_tmp = memberG(XlT[i, :], XuT[i, :], classifiers[idClf].V, classifiers[idClf].W, gama, oper)

            for j in range(noClasses):
                # get max membership of hyperboxes with class label j
                same_j_labels = mem_tmp[classifiers[idClf].classId == classes[j]]
                if len(same_j_labels) > 0:
                    mem_max = same_j_labels.max()
                    out[i, j] = out[i, j] + mem_max

        # compute membership value of each class over all classifiers
        out[i, :] = out[i, :] / numClassifier
        # get max membership value for each class with regard to the i-th sample
        maxb = out[i].max()
        # get positions of indices of all classes with max membership
        maxMemInd = out[i] == maxb
        #misclass[i] = ~(np.any(classes[maxMemInd] == patClassIdTest[i]) | (patClassIdTest[i] == 0))
        misclass[i] = np.logical_or((classes[maxMemInd] == patClassIdTest[i]).any(), patClassIdTest[i] == UNLABELED_CLASS) != True

    # count number of missclassified patterns
    summis = np.sum(misclass)

    result = Bunch(summis = summis, misclass = misclass, out = out, classes = classes)
    return result


def predictOnlineOfflineCombination(onlClassifier, offClassifier, XlT, XuT, patClassIdTest, gama = 1, oper = 'min'):
    """
    GFMM online-offline classifier (test routine)

      result = predictOnlineOfflineCombination(onlClassifier, offClassifier, XlT,XuT,patClassIdTest,gama,oper)

    INPUT
      onlClassifier   online classifier with the following attributes:
                        + V: hyperbox lower bounds
                        + W: hyperbox upper bounds
                        + classId: hyperbox class labels (crisp)

      offClassifier   offline classifier with the following attributes:
                        + V: hyperbox lower bounds
                        + W: hyperbox upper bounds
                        + classId: hyperbox class labels (crisp)

      XlT               Test data lower bounds (rows = objects, columns = features)
      XuT               Test data upper bounds (rows = objects, columns = features)
      patClassIdTest    Test data class labels (crisp)
      gama              Membership function slope (default: 1)
      oper              Membership calculation operation: 'min' or 'prod' (default: 'min')

   OUTPUT
      result           A object with Bunch datatype containing all results as follows:
                          + summis           Number of misclassified objects
                          + misclass         Binary error map
                          + out              Soft class memberships

    """

    #initialization
    yX = XlT.shape[0]
    misclass = np.zeros(yX)
    classes = np.union1d(onlClassifier.classId, offClassifier.classId)
    noClasses = classes.size
    mem_onl = np.zeros((yX, onlClassifier.V.shape[0]))
    mem_off = np.zeros((yX, offClassifier.V.shape[0]))
    out = np.zeros((yX, noClasses))

    # classifications
    for i in range(yX):
        mem_onl[i, :] = memberG(XlT[i, :], XuT[i, :], onlClassifier.V, onlClassifier.W, gama, oper) # calculate memberships for all hyperboxes in the online classifier
        bmax_onl = mem_onl[i, :].max()	                                   # get max membership value among hyperboxes in the online classifier
        maxVind_onl = np.nonzero(mem_onl[i,:] == bmax_onl)[0]             # get indexes of all hyperboxes in the online classifier with max membership

        mem_off[i, :] = memberG(XlT[i, :], XuT[i, :], offClassifier.V, offClassifier.W, gama, oper) # calculate memberships for all hyperboxes in the offline classifier
        bmax_off = mem_off[i, :].max()	                                   # get max membership value among hyperboxes in the offline classifier
        maxVind_off = np.nonzero(mem_off[i,:] == bmax_off)[0]                 # get indexes of all hyperboxes in the offline classifier with max membership


        for j in range(noClasses):
            out_onl_mems = mem_onl[i, onlClassifier.classId == classes[j]]            # get max memberships for each class of online classifier
            if len(out_onl_mems) > 0:
                out_onl = out_onl_mems.max()
            else:
                out_onl = 0

            out_off_mems = mem_off[i, offClassifier.classId == classes[j]]            # get max memberships for each class of offline classifier
            if len(out_off_mems) > 0:
                out_off = out_off_mems.max()
            else:
                out_off = 0

            if out_onl > out_off:
                out[i, j] = out_onl
            else:
                out[i, j] = out_off

        if bmax_onl > bmax_off:
            if len(np.unique(onlClassifier.classId[maxVind_onl])) > 1:
                if len(np.unique(offClassifier.classId[maxVind_off])) > 1:
                    misclass[i] = True
                else:
                    misclass[i] = ~(np.any(offClassifier.classId[maxVind_off] == patClassIdTest[i]) | (patClassIdTest[i] == UNLABELED_CLASS))
            else:
                misclass[i] = ~(np.any(onlClassifier.classId[maxVind_onl] == patClassIdTest[i]) | (patClassIdTest[i] == UNLABELED_CLASS))
        else:
            if len(np.unique(offClassifier.classId[maxVind_off])) > 1:
                if len(np.unique(onlClassifier.classId[maxVind_onl])) > 1:
                    misclass[i] = True
                else:
                    misclass[i] = ~(np.any(onlClassifier.classId[maxVind_onl] == patClassIdTest[i]) | (patClassIdTest[i] == UNLABELED_CLASS))
            else:
                misclass[i] = ~(np.any(offClassifier.classId[maxVind_off] == patClassIdTest[i]) | (patClassIdTest[i] == UNLABELED_CLASS))

    # results
    summis = np.sum(misclass).astype(np.int64)

    result = Bunch(summis = summis, misclass = misclass, out = out)
    return result
