# -*- coding: utf-8 -*-
"""
Created on Mon Sep  3 11:22:08 2018

@author: Thanh Tung Khuat

GFMM Predictor

"""

import numpy as np
import random
from copy import deepcopy
from scipy.stats import mode
from collections import Counter
import operator
from functionhelper.measurehelper import manhattan_distance, min_distance, manhattan_distance_with_missing_value
from functionhelper.membershipcalc import memberG, memberG_with_cat_one_hot_features, memberG_with_frequency_based_cat_features, memberG_extended_iol, memberG_with_selected_features, gaussian_memb
from functionhelper.bunchdatatype import Bunch

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

        winner_cls = np.unique(classId[maxVind])
        if len(winner_cls) > 1:
            #print('Input is in the boundary')
            # make random selection
            predicted_class[i] = random.choice(winner_cls)
        else:
            predicted_class[i] = classId[maxVind[0]]
        
        if patClassIdTest is not None:
            if predicted_class[i] == patClassIdTest[i] or patClassIdTest[i] == UNLABELED_CLASS:
                misclass[i] = False
            else:
                misclass[i] = True
                
    # results
    summis = np.sum(misclass).astype(np.int64)

    result = Bunch(summis = summis, misclass = misclass, predicted_class=predicted_class)
    return result

def predict_with_cat_one_hot_feature(V, W, D, classId, XlT, XuT, X_cat, gama = 1, is_use_manhat=False, oper = 'min'):
    """
    predict function for GFMM with categorical features using one-hot encoding method

    INPUT
      V                 Hyperbox lower bounds
      W                 Hyperbox upper bounds
      D                 Hyperbox categorical features
      classId	        Input data (hyperbox) class labels (crisp)
      XlT               Test data lower bounds for numerical feature (rows = objects, columns = features)
      XuT               Test data upper bounds for numerical feature (rows = objects, columns = features)
      X_cat             Categorical features
      patClassIdTest    Test data class labels (crisp)
      gama              Membership function slope (default: 1)
      oper              Membership calculation operation: 'min' or 'prod' (default: 'min')
      is_use_manhat     + True: using manhattan distance with continous feature for points located on the decision boundary
                        + False: select the class randomly for points located on the decision boundary

   OUTPUT
      predicted_class   Predicted class

    """
    
    if XlT is not None:
        if len(XlT.shape) == 1:
            XlT = XlT.reshape(1, -1)
        if len(XuT.shape) == 1:
            XuT = XuT.reshape(1, -1)
            
    if X_cat is not None:
        if len(X_cat.shape) == 1:
            if X_cat.shape[0] > 1:
                X_cat = deepcopy(X_cat).reshape(1, -1)
            else:
                X_tmp = np.zeros((1, 1), dtype=np.object)
                X_tmp.fill(deepcopy(X_cat))
                X_cat = X_tmp
        
    #initialization
    if XlT is not None:       
        yX = XlT.shape[0]
    else:
        yX = X_cat.shape[0]
    
    predicted_class = np.full(yX, None)
    random.seed(0)
    # classifications
    num_ele_on_boundary = 0
    for i in range(yX):
        if (XlT is not None) and (X_cat is not None):
            mem = memberG_with_cat_one_hot_features(XlT[i, :], XuT[i, :], X_cat[i,:], V, W, D, gama, oper) # calculate memberships for all hyperboxes
        else:
            if XlT is not None:
                mem = memberG_with_cat_one_hot_features(XlT[i, :], XuT[i, :], None, V, W, D, gama, oper)
            else:
                mem = memberG_with_cat_one_hot_features(None, None, X_cat[i, :], V, W, D, gama, oper)
            
        bmax = mem.max()	                                          # get max membership value
        maxVind = np.nonzero(mem == bmax)[0]                         # get indexes of all hyperboxes with max membership

        winner_cls = np.unique(classId[maxVind])
        if len(winner_cls) > 1:
            num_ele_on_boundary = num_ele_on_boundary + 1
            if (is_use_manhat == False) or (XlT is None):
                predicted_class[i] = random.choice(winner_cls)
            else:
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
        else:
            predicted_class[i] = classId[maxVind[0]]
    
    print("Number points on the boundary = ", num_ele_on_boundary)
    return predicted_class

def predict_with_cat_feature_freq(V, W, E, F, classId, XlT, XuT, X_cat, sim_vec, gama = 1, is_use_manhat = False, oper = 'min'):
    """
    predict function for GFMM with categorical features using label encoding method and the membership func for cat features is computed based on frequency

    INPUT
      V                 Hyperbox lower bounds of continous features
      W                 Hyperbox upper bounds of continous features
      E                 Hyperbox lower bound of categorical features
      F                 Hyperbox upper bound of categorical features
      sim_vec           A numpy array contains all mapping of similarity values for all cat features
      classId	        Input data (hyperbox) class labels (crisp)
      XlT               Test data lower bounds for numerical feature (rows = objects, columns = features)
      XuT               Test data upper bounds for numerical feature (rows = objects, columns = features)
      X_cat             Categorical features
      patClassIdTest    Test data class labels (crisp)
      gama              Membership function slope (default: 1)
      oper              Membership calculation operation: 'min' or 'prod' (default: 'min')
      is_use_manhat     + True: using manhattan distance with continous feature for points located on the decision boundary
                        + False: select the class randomly for points located on the decision boundary

   OUTPUT
      predicted_class   Predicted class

    """
    
    if XlT is not None:
        if len(XlT.shape) == 1:
            XlT = XlT.reshape(1, -1)
        if len(XuT.shape) == 1:
            XuT = XuT.reshape(1, -1)
            
    if X_cat is not None:
        if len(X_cat.shape) == 1:
            if X_cat.shape[0] > 1:
                X_cat = deepcopy(X_cat).reshape(1, -1)
            else:
                X_tmp = np.zeros((1, 1), dtype=np.object)
                X_tmp.fill(deepcopy(X_cat))
                X_cat = X_tmp
        
    #initialization
    if XlT is not None:       
        yX = XlT.shape[0]
    else:
        yX = X_cat.shape[0]
    
    predicted_class = np.full(yX, None)
    random.seed(0)
    # classifications
    num_ele_on_boundary = 0
    for i in range(yX):
        if (XlT is not None) and (X_cat is not None):
            mem = memberG_with_frequency_based_cat_features(XlT[i, :], XuT[i, :], X_cat[i,:], X_cat[i, :], V, W, E, F, sim_vec, gama, oper) # calculate memberships for all hyperboxes
        else:
            if XlT is not None:
                mem = memberG_with_frequency_based_cat_features(XlT[i, :], XuT[i, :], None, None, V, W, E, F, sim_vec, gama, oper)
            else:
                mem = memberG_with_frequency_based_cat_features(None, None, X_cat[i, :], X_cat[i, :], V, W, E, F, sim_vec, gama, oper)
            
        bmax = mem.max()	                                          # get max membership value
        maxVind = np.nonzero(mem == bmax)[0]                         # get indexes of all hyperboxes with max membership
        winner_cls = np.unique(classId[maxVind])
        if len(winner_cls) > 1:
            num_ele_on_boundary = num_ele_on_boundary + 1
            if (is_use_manhat == False) or (XlT is None):
                predicted_class[i] = random.choice(winner_cls)
            else:
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
        else:
            predicted_class[i] = classId[maxVind[0]]
    
    print("Number of points on the boundary = ", num_ele_on_boundary)
    return predicted_class

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
        mem = memberG(XlT[i, :], XuT[i, :], V, W, gama, oper) # calculate memberships for all hyperboxes
        bmax = mem.max()	                                          # get max membership value
        maxVind = np.nonzero(mem == bmax)[0]                         # get indexes of all hyperboxes with max membership
        mem_vals[i] = bmax
        
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
            if patClassIdTest is not None:
                if classId[maxVind[id_min_dist]] == patClassIdTest[i]:
                    misclass[i] = False
                else:
                    misclass[i] = True
        else:
            predicted_class[i] = classId[maxVind[0]]
            if patClassIdTest is not None:
                if classId[maxVind[0]] == patClassIdTest[i]:
                    misclass[i] = False
                else:
                    misclass[i] = True
                    #misclass[i] = ~(np.any(classId[maxVind] == patClassIdTest[i]) | (patClassIdTest[i] == 0))

    # results
    summis = np.sum(misclass).astype(np.int64)

    result = Bunch(summis = summis, misclass = misclass, numSampleInBoundary = numPointInBoundary, predicted_class=predicted_class, mem_vals=mem_vals)
    
    return result

def predict_with_manhattan_weighted_class(V, W, classId, XlT, XuT, patClassIdTest, weights, gama = 1, oper = 'min'):
    """
    GFMM classifier (test routine): Using Manhattan distance in the case of many hyperboxes with different classes having the same maximum membership value
    We use weights for classes
    
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
      weights           A dictionary contains weights for classes 

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
    
    # create an vector of all weights for all hyperboxes
    mem_weight = []
    for i in classId:
        mem_weight.append(weights[i])
    
    mem_weight = np.array(mem_weight)
        
    #initialization
    yX = XlT.shape[0]
    misclass = np.zeros(yX)
    mem_vals = np.zeros(yX)
    numPointInBoundary = 0
    predicted_class = np.full(yX, None)
    # classifications
    for i in range(yX):
        mem = memberG(XlT[i, :], XuT[i, :], V, W, gama, oper) # calculate memberships for all hyperboxes
        # multiply with weights
        mem = mem * mem_weight
        bmax = mem.max()	                                          # get max membership value
        maxVind = np.nonzero(mem == bmax)[0]                         # get indexes of all hyperboxes with max membership
        mem_vals[i] = bmax
        
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
            if patClassIdTest is not None:
                if classId[maxVind[id_min_dist]] == patClassIdTest[i]:
                    misclass[i] = False
                else:
                    misclass[i] = True
        else:
            predicted_class[i] = classId[maxVind[0]]
            if patClassIdTest is not None:
                if classId[maxVind[0]] == patClassIdTest[i]:
                    misclass[i] = False
                else:
                    misclass[i] = True
                    #misclass[i] = ~(np.any(classId[maxVind] == patClassIdTest[i]) | (patClassIdTest[i] == 0))

    # results
    summis = np.sum(misclass).astype(np.int64)

    result = Bunch(summis = summis, misclass = misclass, numSampleInBoundary = numPointInBoundary, predicted_class=predicted_class, mem_vals=mem_vals)
    
    return result

def predict_with_weighted_manhattan(V, W, classId, numSample, XlT, XuT, patClassIdTest, gama = 1, oper = 'min'):
    """
    GFMM classifier (test routine): Using Manhattan distance weighted by the number of samples included in each winning hyperbox 
    in the case of many hyperboxes with different classes having the same maximum membership value

      result = predict_with_weighted_manhattan(V, W, classId, numSample, XlT, XuT, patClassIdTest, gama, oper)

    INPUT
      V                 Tested model hyperbox lower bounds
      W                 Tested model hyperbox upper bounds
      classId	        Input data (hyperbox) class labels (crisp)
      numSample         Numbers of samples contained in corresponding hyperboxes: an 1-D numpy array
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
        mem = memberG(XlT[i, :], XuT[i, :], V, W, gama, oper) # calculate memberships for all hyperboxes
        bmax = mem.max()	                                          # get max membership value
        maxVind = np.nonzero(mem == bmax)[0]                         # get indexes of all hyperboxes with max membership
        mem_vals[i] = bmax
        
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
            # Combine with samples
            weighted_sample = numSample[maxVind] / numSample[maxVind].sum()
            weighted_maht_dist = maht_dist / weighted_sample
            
            id_min_dist = weighted_maht_dist.argmin()
            
            predicted_class[i] = classId[maxVind[id_min_dist]]
            if patClassIdTest is not None:
                if classId[maxVind[id_min_dist]] == patClassIdTest[i]:
                    misclass[i] = False
                else:
                    misclass[i] = True
        else:
            predicted_class[i] = classId[maxVind[0]]
            if patClassIdTest is not None:
                if classId[maxVind[0]] == patClassIdTest[i]:
                    misclass[i] = False
                else:
                    misclass[i] = True
                    
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
        mem = memberG(XlT[i, :], XuT[i, :], V, W, gama, oper) # calculate memberships for all hyperboxes
        bmax = mem.max()	                                          # get max membership value
        maxVind = np.nonzero(mem == bmax)[0]                         # get indexes of all hyperboxes with max membership

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
            if patClassIdTest is not None:
                if classId[maxVind[id_min_dist]] == patClassIdTest[i]:
                    misclass[i] = False
                else:
                    misclass[i] = True
        else:
            predicted_class[i] = classId[maxVind[0]]
            if patClassIdTest is not None:
                if classId[maxVind[0]] == patClassIdTest[i]:
                    misclass[i] = False
                else:
                    misclass[i] = True
                    #misclass[i] = ~(np.any(classId[maxVind] == patClassIdTest[i]) | (patClassIdTest[i] == 0))

    # results
    summis = np.sum(misclass).astype(np.int64)

    result = Bunch(summis = summis, misclass = misclass, numSampleInBoundary = numPointInBoundary, predicted_class=predicted_class)
    
    return result

def predict_with_manhattan_knn(V, W, classId, XlT, XuT, patClassIdTest, gama = 1, K_threshold = 5, oper = 'min'):
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
            # Find all average points of all hyperboxes with the same membership value
            avg_point_mat = (V[maxVind] + W[maxVind]) / 2
            # compute the manhattan distance from XgT_mat to all average points of all hyperboxes with the same membership value
            maht_dist = manhattan_distance(avg_point_mat, XgT_mat)
            #maht_dist = min_distance(avg_point_mat, XgT_mat)
            if len(maht_dist) < K_threshold:
                K_threshold = len(maht_dist)
                
            # Select K_threshold minimum manhattan distances
            selected_min_dist_id = np.argsort(maht_dist)[:K_threshold]
            selected_class_min_dist = classId[maxVind[selected_min_dist_id]]
            count_dict = Counter(selected_class_min_dist)
            max_val = max(count_dict.values())
            lst_max_dist_cls = [k for k in count_dict if count_dict[k] == max_val]
            
            if len(lst_max_dist_cls) == 1:
                predicted_class[i] = lst_max_dist_cls[0]
            else:
                predicted_class[i] = classId[maxVind[selected_min_dist_id[0]]]
            
            if patClassIdTest is not None:
                if predicted_class[i] == patClassIdTest[i]:
                    misclass[i] = False
                else:
                    misclass[i] = True
            
        else:
            predicted_class[i] = classId[maxVind[0]]
            if patClassIdTest is not None:
                if classId[maxVind[0]] == patClassIdTest[i]:
                    misclass[i] = False
                else:
                    misclass[i] = True
                    #misclass[i] = ~(np.any(classId[maxVind] == patClassIdTest[i]) | (patClassIdTest[i] == 0))

    # results
    summis = np.sum(misclass).astype(np.int64)

    result = Bunch(summis = summis, misclass = misclass, numSampleInBoundary = numPointInBoundary, predicted_class=predicted_class)
    
    return result

def predict_with_manhattan_and_missing_values_knn(V, W, classId, XlT, XuT, patClassIdTest, gama = 1, K_threshold = 5, oper = 'min'):
    """
    GFMM classifier (test routine): Using Manhattan distance in the case of many hyperboxes with different classes having the same maximum membership value
    This function handles missing values and using knn for hyperboxes with the same membership degree
    
      result = predict_with_manhattan_and_missing_values_knn(V,W,classId,XlT,XuT,patClassIdTest,gama,oper)

    INPUT
      V                 Tested model hyperbox lower bounds
      W                 Tested model hyperbox upper bounds
      classId	          Input data (hyperbox) class labels (crisp)
      XlT               Test data lower bounds (rows = objects, columns = features)
      XuT               Test data upper bounds (rows = objects, columns = features)
      patClassIdTest    Test data class labels (crisp)
      gama              Membership function slope (default: 1)
      K_threshold       The number of hyperboxes is considered if there are many winning hyperboxes (default: 5)
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
                if len(maht_dist) < K_threshold:
                    K_threshold = len(maht_dist)
                    
                # Select K_threshold minimum manhattan distances
                selected_min_dist_id = np.argsort(maht_dist)[:K_threshold]
                selected_class_min_dist = classId[maxVind[selected_min_dist_id]]
                count_dict = Counter(selected_class_min_dist)
                max_val = max(count_dict.values())
                lst_max_dist_cls = [k for k in count_dict if count_dict[k] == max_val]
                
                if len(lst_max_dist_cls) == 1:
                    predicted_class[i] = lst_max_dist_cls[0]
                else:
                    # Many classes with the same K mininum distance values => select class with min distance value
                    predicted_class[i] = classId[maxVind[selected_min_dist_id[0]]]
                
                if patClassIdTest is not None:
                    if predicted_class[i] == patClassIdTest[i]:
                        misclass[i] = False
                    else:
                        misclass[i] = True
            else:
                predicted_class[i] = classId[maxVind[0]]
                if patClassIdTest is not None:
                    if classId[maxVind[0]] == patClassIdTest[i]:
                        misclass[i] = False
                    else:
                        misclass[i] = True
                    #misclass[i] = ~(np.any(classId[maxVind] == patClassIdTest[i]) | (patClassIdTest[i] == 0))

    # results
    summis = np.sum(misclass).astype(np.int64)

    result = Bunch(summis = summis, misclass = misclass, numSampleInBoundary = numPointInBoundary, predicted_class=predicted_class)
    
    return result

def predict_with_voting_manhattan_knn(V, W, classId, XlT, XuT, patClassIdTest, gama = 1, K_threshold=5, oper = 'min'):
    """
    GFMM classifier (test routine): Using voting in the case of many hyperboxes with different classes having the same maximum membership value
    if there are many classes with the same voting score, we use voting on K-nearest Manhattan distance values of winning classes

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
        mem = memberG(XlT[i, :], XuT[i, :], V, W, gama, oper) # calculate memberships for all hyperboxes
        bmax = mem.max()	                                          # get max membership value
        maxVind = np.nonzero(mem == bmax)[0]                         # get indexes of all hyperboxes with max membership

        if len(np.unique(classId[maxVind])) > 1:
            # count the number of winining classes
            counter_dict = Counter(classId[maxVind])
            max_vote_cls = max(counter_dict.items(), key=operator.itemgetter(1))
            if len(max_vote_cls) == 1:
                # only one class having the maximum voting
                predicted_class[i] = max_vote_cls[0]
            else:
                #print('Using Manhattan')
                # select the class that hyperbox with the minimum manhatan distance to central point
                id_winining_hyperboxes_same_vote = np.in1d(classId[maxVind], max_vote_cls)
                maxVind = maxVind[id_winining_hyperboxes_same_vote]
            
                if (XlT[i] == XuT[i]).all() == False:
                    XlT_mat = np.ones((len(maxVind), 1)) * XlT[i]
                    XuT_mat = np.ones((len(maxVind), 1)) * XuT[i]
                    XgT_mat = (XlT_mat + XuT_mat) / 2
                else:
                    XgT_mat = np.ones((len(maxVind), 1)) * XlT[i]
                    
                # compute the manhattan distance from XgT_mat to all average points of all hyperboxes with the same membership value
                maht_dist = manhattan_distance_with_missing_value(V[maxVind], W[maxVind], XgT_mat)
                
                if len(maht_dist) < K_threshold:
                    K_threshold = len(maht_dist)
                    
                # Select K_threshold minimum manhattan distances
                selected_min_dist_id = np.argsort(maht_dist)[:K_threshold]
                selected_class_min_dist = classId[maxVind[selected_min_dist_id]]
                count_dict = Counter(selected_class_min_dist)
                max_val = max(count_dict.values())
                lst_max_dist_cls = [k for k in count_dict if count_dict[k] == max_val]
                
                if len(lst_max_dist_cls) == 1:
                    predicted_class[i] = lst_max_dist_cls[0]
                else:
                    # Many classes with the same K mininum distance values => select class with min distance value
                    predicted_class[i] = classId[maxVind[selected_min_dist_id[0]]]
            
            if patClassIdTest is not None:
                if predicted_class[i] == patClassIdTest[i]:
                    misclass[i] = False
                else:
                    misclass[i] = True
        else:
            predicted_class[i] = classId[maxVind[0]]
            if patClassIdTest is not None:
                if classId[maxVind[0]] == patClassIdTest[i]:
                    misclass[i] = False
                else:
                    misclass[i] = True
                    #misclass[i] = ~(np.any(classId[maxVind] == patClassIdTest[i]) | (patClassIdTest[i] == 0))

    # results
    summis = np.sum(misclass).astype(np.int64)

    result = Bunch(summis = summis, misclass = misclass, numSampleInBoundary = numPointInBoundary, predicted_class=predicted_class)
    
    return result

def predict_with_voting(V, W, classId, XlT, XuT, patClassIdTest, gama = 1, oper = 'min'):
    """
    GFMM classifier (test routine): Using voting in the case of many hyperboxes with different classes having the same maximum membership value
    If there are the same voting score, we continue to consider the next membership value for winning classes

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
        mem = memberG(XlT[i, :], XuT[i, :], V, W, gama, oper) # calculate memberships for all hyperboxes
        sorted_mem = np.sort(mem)[::-1]
        max_vote_cls = None
        is_selected_cls = False
        for m in range(len(sorted_mem)):
            bmax = sorted_mem[m]	                                          # get max membership value
            maxVind = np.nonzero(mem == bmax)[0]                         # get indexes of all hyperboxes with max membership
    
            considered_cls = classId[maxVind]
            if max_vote_cls is not None:
                id_considered_cls = np.in1d(considered_cls, max_vote_cls)
                considered_cls = considered_cls[id_considered_cls]
            
            unique_cls = np.unique(considered_cls)
            if len(unique_cls) > 1:
                # count the number of winining classes
                counter_dict = Counter(considered_cls)
                max_vote_cls = max(counter_dict.items(), key=operator.itemgetter(1))
                if len(max_vote_cls) == 1:
                    # only one class having the maximum voting
                    predicted_class[i] = max_vote_cls[0]
                    is_selected_cls = True 
                    break
            elif len(unique_cls) == 1:
                predicted_class[i] = considered_cls[0]
                is_selected_cls = True 
                break
        
        if is_selected_cls == False:
            predicted_class[i] = max_vote_cls[0]
        
        if patClassIdTest is not None:
            if predicted_class[i] == patClassIdTest[i]:
                misclass[i] = False
            else:
                misclass[i] = True
    # results
    summis = np.sum(misclass).astype(np.int64)

    result = Bunch(summis = summis, misclass = misclass, numSampleInBoundary = numPointInBoundary, predicted_class=predicted_class)
    
    return result

def predict_with_k_voting(V, W, classId, XlT, XuT, patClassIdTest, gama = 1, K_threshold=5, oper = 'min'):
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
        mem = memberG(XlT[i, :], XuT[i, :], V, W, gama, oper) # calculate memberships for all hyperboxes
        sorted_mem = np.sort(mem)[::-1]
        max_vote_cls = None
        is_selected_cls = False
        
        selected_id_cls = []
        
        for m in range(K_threshold):
            selected_id_cls.extend(np.nonzero(mem == sorted_mem[m])[0])
            
        considered_cls = classId[selected_id_cls]
        unique_cls = np.unique(considered_cls)
        if len(unique_cls) > 1:
            # count the number of winining classes
            counter_dict = Counter(considered_cls)
            max_vote_cls = max(counter_dict.items(), key=operator.itemgetter(1))
            if len(max_vote_cls) == 1:
                # only one class having the maximum voting
                predicted_class[i] = max_vote_cls[0]
                is_selected_cls = True 
            else:
                # loop to add increase K_threshold with the aim of adding more classes
                for m in range(K_threshold, len(sorted_mem)):
                    extend_id_cls = np.nonzero(mem == sorted_mem[m])[0]
                    extend_cls = classId[extend_id_cls]
                    extend_counter_dict = Counter(extend_cls)
                    for j in extend_counter_dict.keys():
                        if j in counter_dict:
                            counter_dict[j] = counter_dict[j] + extend_counter_dict[j]
                        else:
                            counter_dict[j] = extend_counter_dict[j]
                    max_vote_cls = max(counter_dict.items(), key=operator.itemgetter(1))
                    if len(max_vote_cls) == 1:
                        # only one class having the maximum voting
                        predicted_class[i] = max_vote_cls[0]
                        is_selected_cls = True
                        break
                
        elif len(unique_cls) == 1:
            predicted_class[i] = considered_cls[0]
            is_selected_cls = True
            
        if is_selected_cls == False:
            predicted_class[i] = considered_cls[0]
        
        if patClassIdTest is not None:
            if predicted_class[i] == patClassIdTest[i]:
                misclass[i] = False
            else:
                misclass[i] = True
            
    # results
    summis = np.sum(misclass).astype(np.int64)

    result = Bunch(summis = summis, misclass = misclass, numSampleInBoundary = numPointInBoundary, predicted_class=predicted_class)
    
    return result

def predict_with_voting_counter(V, W, classId, numSamples, XlT, XuT, patClassIdTest, gama = 1, oper = 'min'):
    """
    GFMM classifier (test routine): Using voting in the case of many hyperboxes with different classes having the same maximum membership value
    If there are the same voting score, we continue to consider the next membership value for winning classes

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
        mem = memberG(XlT[i, :], XuT[i, :], V, W, gama, oper) # calculate memberships for all hyperboxes
        bmax = mem.max()	                                          # get max membership value
        maxVind = np.nonzero(mem == bmax)[0]                         # get indexes of all hyperboxes with max membership
        wining_cls = classId[maxVind]
        unique_win_cls = np.unique(wining_cls)
        if len(unique_win_cls) > 1:
            # count the number of winining classes
            wining_num_samples = numSamples[maxVind]
            counter_dict = {}
            for c in unique_win_cls:
                id_cls_c = wining_cls == c
                counter_dict[c] = np.sum(wining_num_samples[id_cls_c])
                
            max_vote_cls = max(counter_dict.items(), key=operator.itemgetter(1))
            
            if len(max_vote_cls) == 1:
                # only one class having the maximum voting
                predicted_class[i] = max_vote_cls[0]
            else:
                #print('Using Manhattan')
                # select the class that hyperbox with the minimum manhatan distance to central point
                id_winining_hyperboxes_same_vote = np.in1d(classId[maxVind], max_vote_cls)
                maxVind = maxVind[id_winining_hyperboxes_same_vote]
            
                if (XlT[i] == XuT[i]).all() == False:
                    XlT_mat = np.ones((len(maxVind), 1)) * XlT[i]
                    XuT_mat = np.ones((len(maxVind), 1)) * XuT[i]
                    XgT_mat = (XlT_mat + XuT_mat) / 2
                else:
                    XgT_mat = np.ones((len(maxVind), 1)) * XlT[i]
                    
                maht_dist = manhattan_distance_with_missing_value(V[maxVind], W[maxVind], XgT_mat)
                id_min_dist = maht_dist.argmin()
                
                predicted_class[i] = classId[maxVind[id_min_dist]]
        else:
            predicted_class[i] = classId[maxVind[0]]
        
        if patClassIdTest is not None:
            if predicted_class[i] == patClassIdTest[i]:
                misclass[i] = False
            else:
                misclass[i] = True
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
        mem = memberG(XlT[i, :], XuT[i, :], V, W, gama, oper) # calculate memberships for all hyperboxes
        bmax = mem.max()	                                          # get max membership value
        maxVind = np.nonzero(mem == bmax)[0]                         # get indexes of all hyperboxes with max membership
        mem_vals[i] = bmax

        cls_same_mem = np.unique(classId[maxVind])
        if len(cls_same_mem) > 1:
            cls_val = UNLABELED_CLASS
            
            is_find_prob_val = True
            if bmax == 1:
                id_box_with_one_sample = np.nonzero(numSamples[maxVind] == 1)[0]
                if len(id_box_with_one_sample) > 0:
                    is_find_prob_val = False
                    random.seed(0)
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
                    if sum_prod_denum != 0:
                        tmp = sum_pro_num / sum_prod_denum
                    else:
                        tmp = 0
                    
                    if tmp > max_prob or ((tmp == max_prob) and (pre_id_cls is not None) and (numSamples[maxVind[id_cls]].sum() > numSamples[maxVind[pre_id_cls]].sum())):
                        max_prob = tmp
                        cls_val = c
                        pre_id_cls = id_cls
                        id_winner[i] = maxVind[id_cls[0]]
          
            predicted_class[i] = cls_val
            if patClassIdTest is not None:
                if cls_val == patClassIdTest[i]:
                    misclass[i] = False
                else:
                    misclass[i] = True
        else:
            predicted_class[i] = classId[maxVind[0]]
            id_winner[i] = maxVind[0]
            if patClassIdTest is not None:
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

def predict_with_probability_extended(V, W, A, classId, numSamples, XlT, XuT, Xcat, gama = 1, alpha=0.5):
    """
    GFMM classifier (test routine): Using probability formular based on the number of samples in the case of many hyperboxes with different classes having the same maximum membership value
        This function can be applied to the data with both continous and numerical features
    
    INPUT
      V                 Tested model hyperbox lower bounds
      W                 Tested model hyperbox upper bounds
      A                 Tested model hyperbox bounds for the categorical features
      classId	        Input data (hyperbox) class labels (crisp)
      numSamples        Save number of samples of each corresponding hyperboxes contained in V, W, and A
      XlT               Test data lower bounds (rows = objects, columns = features)
      XuT               Test data upper bounds (rows = objects, columns = features)
      Xcat              Test data categorical features
      patClassIdTest    Test data class labels (crisp)
      gama              Membership function slope (default: 1)
      alpha             weighted factors trade-off between numerical and categorical features

   OUTPUT
      predicted_class   Predicted class
                          
    """
    if XlT is not None:
        if len(XlT.shape) == 1:
            XlT = XlT.reshape(1, -1)
            XuT = XuT.reshape(1, -1)
            
    if Xcat is not None:
        if len(Xcat.shape) == 1:
            if Xcat.shape[0] > 1:
                Xcat = Xcat.reshape(1, -1)
            else:
                X_tmp = np.zeros((1, 1), dtype=np.object)
                X_tmp.fill(deepcopy(Xcat))
                Xcat = X_tmp
        
    #initialization
    if XlT is not None:       
        yX = XlT.shape[0]
    else:
        yX = Xcat.shape[0]
    
    predicted_class = np.full(yX, None)
    random.seed(0)
    # classifications
    for i in range(yX):
        if (XlT is not None) and (Xcat is not None):
            mem = memberG_extended_iol(XlT[i, :], XuT[i, :], Xcat[i,:], V, W, A, gama, alpha) # calculate memberships for all hyperboxes
        else:
            if XlT is not None:
                mem = memberG_extended_iol(XlT[i, :], XuT[i, :], None, V, W, A, gama, alpha)
            else:
                mem = memberG_extended_iol(None, None, Xcat[i,:], V, W, A, gama, alpha)
            
        bmax = mem.max()	                                          # get max membership value
        maxVind = np.nonzero(mem == bmax)[0]                         # get indexes of all hyperboxes with max membership

        cls_same_mem = np.unique(classId[maxVind])
        if len(cls_same_mem) > 1:
            cls_val = UNLABELED_CLASS
            is_find_prob_val = True
            if bmax == 1:
                id_box_with_one_sample = np.nonzero(numSamples[maxVind] == 1)[0]
                if len(id_box_with_one_sample) > 0:
                    is_find_prob_val = False
                    cls_val = classId[random.choice(maxVind[id_box_with_one_sample])]
                    
            if is_find_prob_val == True:
                sum_prod_denum = (mem[maxVind] * numSamples[maxVind]).sum()
                max_prob = -1
                pre_id_cls = None
                for c in cls_same_mem:
                    id_cls = np.nonzero(classId[maxVind] == c)[0]
                    sum_pro_num = (mem[maxVind[id_cls]] * numSamples[maxVind[id_cls]]).sum()
                    if sum_prod_denum != 0:
                        tmp = sum_pro_num / sum_prod_denum
                    else:
                        tmp = 0
                    
                    if tmp > max_prob or ((tmp == max_prob) and (pre_id_cls is not None) and (numSamples[maxVind[id_cls]].sum() > numSamples[maxVind[pre_id_cls]].sum())):
                        max_prob = tmp
                        cls_val = c
                        pre_id_cls = id_cls
          
            predicted_class[i] = cls_val
        else:
            predicted_class[i] = classId[maxVind[0]]
    
    return predicted_class

def predict_with_voting_weighted(V, W, classId, weights, XlT, XuT, patClassIdTest, gama = 1, oper = 'min'):
    """
    GFMM classifier (test routine): Using voting in the case of many hyperboxes with different classes having the same maximum membership value
    If there are the same voting score, we continue to consider the next membership value for winning classes

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
        mem = memberG(XlT[i, :], XuT[i, :], V, W, gama, oper) # calculate memberships for all hyperboxes
        mem = mem * weights
        sorted_mem = np.sort(mem)[::-1]
        max_vote_cls = None
        is_selected_cls = False
        for m in range(len(sorted_mem)):
            bmax = sorted_mem[m]	                                          # get max membership value
            maxVind = np.nonzero(mem == bmax)[0]                         # get indexes of all hyperboxes with max membership
    
            considered_cls = classId[maxVind]
            if max_vote_cls is not None:
                id_considered_cls = np.in1d(considered_cls, max_vote_cls)
                considered_cls = considered_cls[id_considered_cls]
            
            unique_cls = np.unique(considered_cls)
            if len(unique_cls) > 1:
                # count the number of winining classes
                counter_dict = Counter(considered_cls)
                max_vote_cls = max(counter_dict.items(), key=operator.itemgetter(1))
                if len(max_vote_cls) == 1:
                    # only one class having the maximum voting
                    predicted_class[i] = max_vote_cls[0]
                    is_selected_cls = True 
                    break
            elif len(unique_cls) == 1:
                predicted_class[i] = considered_cls[0]
                is_selected_cls = True 
                break
        
        if is_selected_cls == False:
            predicted_class[i] = max_vote_cls[0]
        
        if patClassIdTest is not None:
            if predicted_class[i] == patClassIdTest[i]:
                misclass[i] = False
            else:
                misclass[i] = True
    # results
    summis = np.sum(misclass).astype(np.int64)

    result = Bunch(summis = summis, misclass = misclass, numSampleInBoundary = numPointInBoundary, predicted_class=predicted_class)
    
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
            
            if patClassIdTest is not None:
                if cls_val == patClassIdTest[i]:
                    misclass[i] = False
                else:
                    misclass[i] = True
        else:
            predicted_class[i] = classId[maxVind[0]]
            if patClassIdTest is not None:
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

def predict_with_probability_k_voting(V, W, classId, weights, XlT, XuT, patClassIdTest, K_threshold = 5, gama = 1, oper = 'min'):
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
        mem = memberG(XlT[i, :], XuT[i, :], V, W, gama, oper) # calculate memberships for all hyperboxes
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
                cur_prob = np.sum(weights[selected_id[id_cls]])
                cur_mem = np.sum(mem[selected_id[id_cls]])
                
                if max_prob < cur_prob:
                    max_prob = cur_prob
                    predicted_class[i] = c
                    max_mem_sum = cur_mem
                else:
                    if max_prob == cur_prob and max_mem_sum < cur_mem:
                        max_prob = cur_prob
                        predicted_class[i] = c
                        max_mem_sum = cur_mem
            
            if patClassIdTest is not None:
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
            
            if patClassIdTest is not None:
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

def predict_with_manhattan_test_sync_data(V, W, classId, XlT, XuT, patClassIdTest, gama = 1, oper = 'min'):
    """
    GFMM classifier (test routine): Using Manhattan distance in the case of many hyperboxes with different classes having the same maximum membership value
        This function is to test the encoding methods on the synthetic data
        
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
                          + predicted_class   Predicted class
                          + num_sample_boundary_lst    List of the number of samples on the boundaries for all testing samples
                          + num_sample_using_mem       Numbers of samples using membership function only
                          + num_sample_using_Mant      Numbers of samples using Manhattan distance to resolve ties
                          + mem_values                 Membership values for all testing samples
                          + num_sample_using_mem_correct    Numbers of samples using the membership function only and classify samples correctly
                          + num_sample_using_Mant_correct   Numbers of samples using the Manhattan distance to resolve the ties correctly
                          + mat_domimant_feature            A matrix to store all features contributing to the membership function of the hyperbox showing the maximum membership value for all testing samples
                                                                For each sample, we have a vector with the size of all features, if a feature is not used, then its value is zero, if a feature is used many times, then its value is 1 at that feature index

    """
    if len(XlT.shape) == 1:
        XlT = XlT.reshape(1, -1)
    if len(XuT.shape) == 1:
        XuT = XuT.reshape(1, -1)
        
    #initialization
    yX = XlT.shape[0]
    yY = XlT.shape[1]
    num_sample_boundary_lst = np.zeros(yX)
    num_sample_using_mem = 0
    num_sample_using_Mant = 0
    num_sample_using_mem_correct = 0
    num_sample_using_Mant_correct = 0
    mem_values = np.zeros(yX)
    mat_domimant_feature = np.zeros((yX, yY), dtype=int)
    predicted_class = np.full(yX, None)
    # classifications
    for i in range(yX):
        mem, selected_fets = memberG_with_selected_features(XlT[i, :], XuT[i, :], V, W, gama) # calculate memberships for all hyperboxes and selected features for each input pattern
        selected_fets = np.array(selected_fets)
        bmax = mem.max()	                                          # get max membership value
        maxVind = np.nonzero(mem == bmax)[0]                         # get indexes of all hyperboxes with max membership
        mem_values[i] = bmax
        
        for idx, list_fe in enumerate(selected_fets):
            for kk in list_fe:
                mat_domimant_feature[i, kk] = 1
        
        if len(np.unique(classId[maxVind])) > 1:
            num_sample_boundary_lst[i] = len(maxVind)
            num_sample_using_Mant = num_sample_using_Mant + 1
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
                num_sample_using_Mant_correct = num_sample_using_Mant_correct + 1          
        else:
            num_sample_using_mem = num_sample_using_mem + 1
            predicted_class[i] = classId[maxVind[0]]
            if classId[maxVind[0]] == patClassIdTest[i]:
                num_sample_using_mem_correct = num_sample_using_mem_correct + 1
                
    predicted_class = np.array(predicted_class, dtype=int)
    result = Bunch(predicted_class = predicted_class, num_sample_boundary_lst=num_sample_boundary_lst, num_sample_using_mem=num_sample_using_mem, num_sample_using_Mant=num_sample_using_Mant, mem_values=mem_values, num_sample_using_mem_correct=num_sample_using_mem_correct, num_sample_using_Mant_correct=num_sample_using_Mant_correct, mat_domimant_feature=mat_domimant_feature)
    
    return result

def predict_with_probability_test_sync_data(V, W, classId, numSamples, XlT, XuT, patClassIdTest, gama = 1, oper = 'min'):
    """
    GFMM classifier (test routine): Using probability formular based on the number of samples in the case of many hyperboxes with different classes having the same maximum membership value
        This function is to test the encoding methods on the synthetic data
        
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
                           + predicted_class   Predicted class
                          + num_sample_boundary_lst    List of the number of samples on the boundaries for all testing samples
                          + num_sample_using_mem       Numbers of samples using membership function only
                          + num_sample_using_Mant      Numbers of samples using Manhattan distance to resolve ties
                          + mem_values                 Membership values for all testing samples
                          + num_sample_using_mem_correct    Numbers of samples using the membership function only and classify samples correctly
                          + num_sample_using_Mant_correct   Numbers of samples using the Manhattan distance to resolve the ties correctly
                          + mat_domimant_feature            A matrix to store all features contributing to the membership function of the hyperbox showing the maximum membership value for all testing samples
                                                                For each sample, we have a vector with the size of all features, if a feature is not used, then its value is zero, if a feature is used many times, then its value is 1 at that feature index

    """
    if len(XlT.shape) == 1:
        XlT = XlT.reshape(1, -1)
    if len(XuT.shape) == 1:
        XuT = XuT.reshape(1, -1)
        
    #initialization
    yX = XlT.shape[0]
    yY = XlT.shape[1]
    num_sample_boundary_lst = np.zeros(yX)
    num_sample_using_mem = 0
    num_sample_using_prob = 0
    num_sample_using_mem_correct = 0
    num_sample_using_prob_correct = 0
    mem_values = np.zeros(yX)
    mat_domimant_feature = np.zeros((yX, yY), dtype=int)
    id_winner = np.zeros(yX, dtype=np.int64)
    predicted_class = np.full(yX, None)
    
    for i in range(yX):
        mem, selected_fets = memberG_with_selected_features(XlT[i, :], XuT[i, :], V, W, gama) # calculate memberships for all hyperboxes
        selected_fets = np.array(selected_fets)
        bmax = mem.max()	                                          # get max membership value
        maxVind = np.nonzero(mem == bmax)[0]                         # get indexes of all hyperboxes with max membership
        mem_values[i] = bmax
        
        for idx, list_fe in enumerate(selected_fets):
            for kk in list_fe:
                mat_domimant_feature[i, kk] = 1
                
        cls_same_mem = np.unique(classId[maxVind])
        if len(cls_same_mem) > 1:
            cls_val = UNLABELED_CLASS
            
            is_find_prob_val = True
            if bmax == 1:
                id_box_with_one_sample = np.nonzero(numSamples[maxVind] == 1)[0]
                if len(id_box_with_one_sample) > 0:
                    num_sample_using_mem = num_sample_using_mem + 1
                    is_find_prob_val = False
                    random.seed(0)
                    id_winner[i] = random.choice(maxVind[id_box_with_one_sample])
                    cls_val = classId[id_winner[i]]
                    
                    if cls_val == patClassIdTest[i]:
                        num_sample_using_mem_correct = num_sample_using_mem_correct + 1
                    
            if is_find_prob_val == True:
                #print('bmax=', bmax)
                #print("Using probability function")
                num_sample_boundary_lst[i] = len(maxVind)
                num_sample_using_prob = num_sample_using_prob + 1
                sum_prod_denum = (mem[maxVind] * numSamples[maxVind]).sum()
                max_prob = -1
                pre_id_cls = None
                for c in cls_same_mem:
                    id_cls = np.nonzero(classId[maxVind] == c)[0]
                    sum_pro_num = (mem[maxVind[id_cls]] * numSamples[maxVind[id_cls]]).sum()
                    if sum_prod_denum != 0:
                        tmp = sum_pro_num / sum_prod_denum
                    else:
                        tmp = 0
                    
                    if tmp > max_prob or ((tmp == max_prob) and (pre_id_cls is not None) and (numSamples[maxVind[id_cls]].sum() > numSamples[maxVind[pre_id_cls]].sum())):
                        max_prob = tmp
                        cls_val = c
                        pre_id_cls = id_cls
                        id_winner[i] = maxVind[id_cls[0]]
                        
                if cls_val == patClassIdTest[i]:
                    num_sample_using_prob_correct = num_sample_using_prob_correct + 1
          
            predicted_class[i] = cls_val
           
        else:
            num_sample_using_mem = num_sample_using_mem + 1
            predicted_class[i] = classId[maxVind[0]]
            id_winner[i] = maxVind[0]
            if predicted_class[i] == patClassIdTest[i]:
                num_sample_using_mem_correct = num_sample_using_mem_correct + 1
                
    predicted_class = np.array(predicted_class, dtype=int)
    result = Bunch(predicted_class = predicted_class, num_sample_boundary_lst=num_sample_boundary_lst, num_sample_using_mem=num_sample_using_mem, num_sample_using_prob=num_sample_using_prob, mem_values=mem_values, num_sample_using_mem_correct=num_sample_using_mem_correct, num_sample_using_prob_correct=num_sample_using_prob_correct, mat_domimant_feature=mat_domimant_feature)
    
    return result

def predict_with_probability_gaussian(V, W, classId, numSamples, XlT, XuT, patClassIdTest, gama = 1):
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
        mem = gaussian_memb(XlT[i, :], XuT[i, :], V, W, gama) # calculate memberships for all hyperboxes
        bmax = mem.max()	                                          # get max membership value
        maxVind = np.nonzero(mem == bmax)[0]                         # get indexes of all hyperboxes with max membership
        mem_vals[i] = bmax

        cls_same_mem = np.unique(classId[maxVind])
        if len(cls_same_mem) > 1:
            cls_val = UNLABELED_CLASS
            
            is_find_prob_val = True
            if bmax == 1:
                id_box_with_one_sample = np.nonzero(numSamples[maxVind] == 1)[0]
                if len(id_box_with_one_sample) > 0:
                    is_find_prob_val = False
                    random.seed(0)
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
                    if sum_prod_denum != 0:
                        tmp = sum_pro_num / sum_prod_denum
                    else:
                        tmp = 0
                    
                    if tmp > max_prob or ((tmp == max_prob) and (pre_id_cls is not None) and (numSamples[maxVind[id_cls]].sum() > numSamples[maxVind[pre_id_cls]].sum())):
                        max_prob = tmp
                        cls_val = c
                        pre_id_cls = id_cls
                        id_winner[i] = maxVind[id_cls[0]]
          
            predicted_class[i] = cls_val
            if patClassIdTest is not None:
                if cls_val == patClassIdTest[i]:
                    misclass[i] = False
                else:
                    misclass[i] = True
        else:
            predicted_class[i] = classId[maxVind[0]]
            id_winner[i] = maxVind[0]
            if patClassIdTest is not None:
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

def predict_with_manhattan_gaussian(V, W, classId, XlT, XuT, patClassIdTest, gama = 1):
    """
    GFMM classifier (test routine): Using Manhattan distance in the case of many hyperboxes with different classes having the same maximum membership value
    INPUT
      V                 Tested model hyperbox lower bounds
      W                 Tested model hyperbox upper bounds
      classId	          Input data (hyperbox) class labels (crisp)
      XlT               Test data lower bounds (rows = objects, columns = features)
      XuT               Test data upper bounds (rows = objects, columns = features)
      patClassIdTest    Test data class labels (crisp)
      gama              Membership function slope (default: 1)
      
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
        mem = gaussian_memb(XlT[i, :], XuT[i, :], V, W, gama) # calculate memberships for all hyperboxes
        bmax = mem.max()	                                          # get max membership value
        maxVind = np.nonzero(mem == bmax)[0]                         # get indexes of all hyperboxes with max membership
        mem_vals[i] = bmax
        
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
            if patClassIdTest is not None:
                if classId[maxVind[id_min_dist]] == patClassIdTest[i]:
                    misclass[i] = False
                else:
                    misclass[i] = True
        else:
            predicted_class[i] = classId[maxVind[0]]
            if patClassIdTest is not None:
                if classId[maxVind[0]] == patClassIdTest[i]:
                    misclass[i] = False
                else:
                    misclass[i] = True
                    #misclass[i] = ~(np.any(classId[maxVind] == patClassIdTest[i]) | (patClassIdTest[i] == 0))

    # results
    summis = np.sum(misclass).astype(np.int64)

    result = Bunch(summis = summis, misclass = misclass, numSampleInBoundary = numPointInBoundary, predicted_class=predicted_class, mem_vals=mem_vals)
    
    return result

def predict_with_weighted_manhattan_gaussian(V, W, classId, numSample, XlT, XuT, patClassIdTest, gama = 1):
    """
    GFMM classifier (test routine): Using Manhattan distance weighted by the number of samples included in each winning hyperbox 
    in the case of many hyperboxes with different classes having the same maximum membership value

    INPUT
      V                 Tested model hyperbox lower bounds
      W                 Tested model hyperbox upper bounds
      classId	        Input data (hyperbox) class labels (crisp)
      numSample         Numbers of samples contained in corresponding hyperboxes: an 1-D numpy array
      XlT               Test data lower bounds (rows = objects, columns = features)
      XuT               Test data upper bounds (rows = objects, columns = features)
      patClassIdTest    Test data class labels (crisp)
      gama              Membership function slope (default: 1)
      
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
        mem = gaussian_memb(XlT[i, :], XuT[i, :], V, W, gama) # calculate memberships for all hyperboxes
        bmax = mem.max()	                                          # get max membership value
        maxVind = np.nonzero(mem == bmax)[0]                         # get indexes of all hyperboxes with max membership
        mem_vals[i] = bmax
        
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
            # Combine with samples
            weighted_sample = numSample[maxVind] / numSample[maxVind].sum()
            weighted_maht_dist = maht_dist / weighted_sample
            
            id_min_dist = weighted_maht_dist.argmin()
            
            predicted_class[i] = classId[maxVind[id_min_dist]]
            if patClassIdTest is not None:
                if classId[maxVind[id_min_dist]] == patClassIdTest[i]:
                    misclass[i] = False
                else:
                    misclass[i] = True
        else:
            predicted_class[i] = classId[maxVind[0]]
            if patClassIdTest is not None:
                if classId[maxVind[0]] == patClassIdTest[i]:
                    misclass[i] = False
                else:
                    misclass[i] = True
                    
    # results
    summis = np.sum(misclass).astype(np.int64)

    result = Bunch(summis = summis, misclass = misclass, numSampleInBoundary = numPointInBoundary, predicted_class=predicted_class, mem_vals=mem_vals)
    
    return result

def predict_gaussian(V, W, classId, XlT, XuT, patClassIdTest, gama = 1):
    """
    GFMM classifier (test routine)

    INPUT
      V                 Tested model hyperbox lower bounds
      W                 Tested model hyperbox upper bounds
      classId	          Input data (hyperbox) class labels (crisp)
      XlT               Test data lower bounds (rows = objects, columns = features)
      XuT               Test data upper bounds (rows = objects, columns = features)
      patClassIdTest    Test data class labels (crisp)
      gama              Membership function slope (default: 1)
      
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
        mem = gaussian_memb(XlT[i, :], XuT[i, :], V, W, gama) # calculate memberships for all hyperboxes
        bmax = mem.max()	                                          # get max membership value
        maxVind = np.nonzero(mem == bmax)[0]                         # get indexes of all hyperboxes with max membership

        winner_cls = np.unique(classId[maxVind])
        if len(winner_cls) > 1:
            #print('Input is in the boundary')
            # make random selection
            predicted_class[i] = random.choice(winner_cls)
        else:
            predicted_class[i] = classId[maxVind[0]]
        
        if patClassIdTest is not None:
            if predicted_class[i] == patClassIdTest[i] or patClassIdTest[i] == UNLABELED_CLASS:
                misclass[i] = False
            else:
                misclass[i] = True
                
    # results
    summis = np.sum(misclass).astype(np.int64)

    result = Bunch(summis = summis, misclass = misclass, predicted_class=predicted_class)
    return result