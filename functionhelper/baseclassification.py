# -*- coding: utf-8 -*-
"""
Created on Thu Sep 20 22:40:41 2018

@author: Thanh Tung Khuat

Do classification using Simpson's method
"""

import numpy as np
from functionhelper.membershipcalc import simpsonMembership
from functionhelper.bunchdatatype import Bunch
from functionhelper.measurehelper import manhattan_distance, rfmm_distance
import random as rd

def predict(V, W, classId, XhT, patClassIdTest, gama = 1, is_using_manhattan=True):
    """
    FMNN classifier (test routine)
    
      result = predict(V,W,classId,XhT,patClassIdTest,gama)
  
    INPUT
      V                 Tested model hyperbox lower bounds
      W                 Tested model hyperbox upper bounds
      classId	           Input data (hyperbox) class labels (crisp)
      XhT               Test input data (rows = objects, columns = features)
      patClassIdTest    Test data class labels (crisp)
      gama              Membership function slope (default: 1)
  
   OUTPUT
      result           A object with Bunch datatype containing all results as follows:
                          + summis           Number of misclassified objects
                          + misclass         Binary error map
                          + sumamb           Number of objects with maximum membership in more than one class
                          + out              Soft class memberships
                          + mem              Hyperbox memberships

    """
    if len(XhT.shape) == 1:
        XhT = XhT.reshape(1, -1)

    #initialization
    yX = XhT.shape[0]
    predicted_class = np.full(yX, None)
    misclass = np.zeros(yX)
    mem = np.zeros((yX, V.shape[0]))

    # classifications
    for i in range(yX):
        mem[i, :] = simpsonMembership(XhT[i, :], V, W, gama) # calculate memberships for all hyperboxes
        bmax = mem[i,:].max()	                               # get max membership value
        maxVind = np.nonzero(mem[i,:] == bmax)[0]           # get indexes of all hyperboxes with max membership
        
        winner_cls = np.unique(classId[maxVind])
        
        if len(winner_cls) > 1:
            if is_using_manhattan == True:
                #print("Using Manhattan function")
                XgT_mat = np.ones((len(maxVind), 1)) * XhT[i]
                # Find all average points of all hyperboxes with the same membership value
                avg_point_mat = (V[maxVind] + W[maxVind]) / 2
                # compute the manhattan distance from XgT_mat to all average points of all hyperboxes with the same membership value
                maht_dist = manhattan_distance(avg_point_mat, XgT_mat)
                
                id_min_dist = maht_dist.argmin()
                
                predicted_class[i] = classId[maxVind[id_min_dist]]
            else:
                # select random class
                predicted_class[i] = rd.choice(winner_cls)
                
            if predicted_class[i] == patClassIdTest[i]:
                misclass[i] = False
            else:
                misclass[i] = True
        else:
            predicted_class[i] = classId[maxVind[0]]
            if predicted_class[i] == patClassIdTest[i]:
                misclass[i] = False
            else:
                misclass[i] = True
    
    # results
    summis = np.sum(misclass).astype(np.int64)
    
    result = Bunch(summis = summis, misclass = misclass, predicted_class=predicted_class)
    
    return result


def predict_rfmm_distance(V, W, classId, XhT, patClassIdTest, gama = 1):
    """
        prediction using the distance in the paper "A refined Fuzzy min-max neural network with new learning procedure for pattern classification"
    """
    if len(XhT.shape) == 1:
        XhT = XhT.reshape(1, -1)

    #initialization
    yX = XhT.shape[0]
    predicted_class = np.full(yX, None)
    misclass = np.zeros(yX)
    mem = np.zeros((yX, V.shape[0]))

    # classifications
    for i in range(yX):
        mem[i, :] = simpsonMembership(XhT[i, :], V, W, gama) # calculate memberships for all hyperboxes
        bmax = mem[i,:].max()	                               # get max membership value
        maxVind = np.nonzero(mem[i,:] == bmax)[0]           # get indexes of all hyperboxes with max membership
        
        if len(np.unique(classId[maxVind])) > 1:
            misclass[i] = True
        else:
            misclass[i] = ~(np.any(classId[maxVind] == patClassIdTest[i]))
            
        if len(np.unique(classId[maxVind])) > 1:
            #print("Using Manhattan function")
            XgT_mat = np.ones((len(maxVind), 1)) * XhT[i]
            # compute the distance from XgT_mat to all average points of all hyperboxes with the same membership value
            dist = rfmm_distance(XgT_mat, V[maxVind], W[maxVind])
            
            id_min_dist = dist.argmin()
            
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
    
    # results
    summis = np.sum(misclass).astype(np.int64)
    
    result = Bunch(summis = summis, misclass = misclass, predicted_class=predicted_class)
    
    return result