# -*- coding: utf-8 -*-
"""
Created on Sat May  4 13:14:44 2019

@author: Thanh Tung Khuat

This is a file to define all utility functions for measuring the performance 
"""

import numpy as np
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import roc_auc_score
from functionhelper import epsilon_missing_value

def multiclass_roc_auc_score(y_test, y_pred, average="macro"):
    """
        AUC ROC Curve Scoring Function for Multi-class Classification
    """
    lb = LabelBinarizer()
    lb.fit(y_test)
    y_test = lb.transform(y_test)
    y_pred = lb.transform(y_pred)
    
    return roc_auc_score(y_test, y_pred, average=average)

def manhattan_distance(X, Y):
    """
        Compute Manhattan distance of two matrices X and Y
        
        Input:
            X, Y: two numpy arrays (1D or 2D)
            
        Output:
            A numpy array containing manhattan distance. If X and Y are 1-D arrays, the output only contains one element
            The number of elements in the output array is the number of rows of X or Y
    """
    if X.ndim > 1:
        return (np.abs(X - Y)).sum(1)
    else:
        return (np.abs(X - Y)).sum()
    
def rfmm_distance(X, V, W):
    if V.ndim > 1:
        N = V.shape[1]
        return (np.abs(X - V) + np.abs(X - W)).sum(1) / (2 * N)
    else:
        N = len(V)
        return (np.abs(X - V) + np.abs(X - W)).sum() / (2 * N)

def manhattan_distance_with_missing_value(V, W, Y):
    """
        V, W contain missing value
        V, W: minimum and maximum points of hyperboxes
    """
    result = np.zeros(V.shape[0], dtype=np.float64)
    for i in range(V.shape[0]):
        id_non_missing_V = np.nonzero(V[i] != -epsilon_missing_value)[0]
        id_non_missing_W = np.nonzero(W[i] != 1 + epsilon_missing_value)[0]
        id_non_missing_val = np.intersect1d(id_non_missing_V, id_non_missing_W)
        Y_sel = Y[i, id_non_missing_val]
        X_sel = (V[i, id_non_missing_val] + W[i, id_non_missing_val]) / 2
        result[i] = np.abs(X_sel - Y_sel).sum()
        
    return result
    
def min_distance(X, Y):
    """
        Compute Manhattan distance of two matrices X and Y
        
        Input:
            X, Y: two numpy arrays (1D or 2D)
            
        Output:
            A numpy array containing manhattan distance. If X and Y are 1-D arrays, the output only contains one element
            The number of elements in the output array is the number of rows of X or Y
    """
    if X.ndim > 1:
        return (np.square(X - Y)).sum(1)
    else:
        return (np.square(X - Y)).sum()
    
    


