# -*- coding: utf-8 -*-
"""
Created on Fri Aug 31 18:23:03 2018

@author: Thanh Tung Khuat

Hyperbox adjustment handling: overlap testing, hyperbox contraction

"""
import numpy as np
from functionhelper import UNLABELED_CLASS, epsilon_missing_value

alpha = 0.000001

def isOverlapTest2Hyperbox(V1, W1, V2, W2, isCheckExtendedHyperbox = True):
    """
        Test overlap between two hyperboxes [V1, W1] and [V2, W2]
        
        isCheckExtendedHyperbox:
            + True: Check if V1 < W1 & V2 < W2 for all dimensions
            + False: No check if V1 < W1 & & V2 < W2 for all dimensions
    """
    if isCheckExtendedHyperbox:
        if ((V1 > W1).any() == True) or ((V2 > W2).any() == True):
            return False
    
    condWiWk = W1 - W2 > 0
    condViVk = V1 - V2 > 0
    condWkVi = W2 - V1 > 0
    condWiVk = W1 - V2 > 0

    c1 = ~condWiWk & ~condViVk & condWiVk
    c2 = condWiWk & condViVk & condWkVi
    c3 = condWiWk & ~condViVk
    c4 = ~condWiWk & condViVk
    c = c1 + c2 + c3 + c4

    ad = c.all()
    
    return ad

def overlapTestOneMany(V, W, ind, classId):
    """
    Checking overlap between hyperbox ind and remaning hyperboxes (1 vs many)
    Only do overlap testing with hyperboxes belonging to other classes
    
    INPUT
        V           Hyperbox lower bounds
        W           Hyperbox upper bounds
        ind         Index of the hyperbox to be checked for overlap
        classId     Class labels of hyperboxes
        
    OUTPUT
        False - no overlap,  True - overlap
    """
    if (V[ind] > W[ind]).any() == True:
        return False
    else:
        indcomp = np.nonzero((W >= V).all(axis = 1))[0] 	# examine only hyperboxes w/o missing dimensions, meaning that in each dimension upper bound is larger than lowerbound
        
        if len(indcomp) == 0:
            return False
        else:
            class_indcomp = classId[indcomp]
            newInd = indcomp[(class_indcomp != classId[ind]) | (classId[ind] == UNLABELED_CLASS)] # get index of hyperbox representing different classes
            numEl = len(newInd)
            if numEl > 0:
                iCur = 0
                isOver = False
                while iCur < numEl and isOver == False:
                    isOver = isOverlapTest2Hyperbox(V[ind], W[ind], V[newInd[iCur]], W[newInd[iCur]], False)
                    iCur = iCur + 1
                    
                return isOver
            else:
                return False
    
def hyperboxOverlapTest(V, W, ind, testInd):
    """
    Hyperbox overlap test

      dim = overlapTest(V, W, ind, testInd)
  
    INPUT
        V           Hyperbox lower bounds
        W           Hyperbox upper bounds
        ind         Index of extended hyperbox
        testInd     Index of hyperbox to test for overlap with the extended hyperbox

    OUTPUT
        dim         Result to be fed into contrG1, which is special numpy array

    """
    dim = np.array([])
    
    if (V[ind] > W[ind]).any() == True:
        return dim
    
    if (V[testInd] > W[testInd]).any() == True:
        return dim
    
    xW = W.shape[1]
    
    condWiWk = W[ind, :] - W[testInd, :] > 0
    condViVk = V[ind, :] - V[testInd, :] > 0
    condWkVi = W[testInd, :] - V[ind, :] > 0
    condWiVk = W[ind, :] - V[testInd, :] > 0

    c1 = ~condWiWk & ~condViVk & condWiVk
    c2 = condWiWk & condViVk & condWkVi
    c3 = condWiWk & ~condViVk
    c4 = ~condWiWk & condViVk
    c = c1 + c2 + c3 + c4

    ad = c.all()

    if ad == True:
        minimum = 1;
        for i in range(xW):
            if c1[i] == True:
                if minimum > W[ind, i] - V[testInd, i]:
                    minimum = W[ind, i] - V[testInd, i]
                    dim = np.array([1, i])
            
            elif c2[i] == True:
                if minimum > W[testInd, i] - V[ind, i]:
                    minimum = W[testInd, i] - V[ind, i]
                    dim = np.array([2, i])
            
            elif c3[i] == True:
                if minimum > (W[testInd, i] - V[ind,i]) and (W[testInd, i] - V[ind, i]) < (W[ind, i] - V[testInd, i]):
                    minimum = W[testInd, i] - V[ind, i]
                    dim = np.array([31, i])
                elif minimum > (W[ind, i] - V[testInd, i]):
                    minimum = W[ind, i] - V[testInd, i]
                    dim = np.array([32, i])
                    
            elif c4[i] == True:
                if minimum > (W[testInd, i] - V[ind, i]) and (W[testInd, i] - V[ind, i]) < (W[ind, i] - V[testInd, i]):
                    minimum = W[testInd, i] - V[ind, i]
                    dim = np.array([41, i])
                elif minimum > (W[ind, i] - V[testInd, i]):
                    minimum = W[ind, i] - V[testInd, i]
                    dim = np.array([42, i])
                
    return dim

def hyperboxOverlapTest_same_missing_features(V, W, ind, testInd):
    """
    Hyperbox overlap test
    This case handles overlap between hyperboxes with the same missing variables

      dim = overlapTest(V, W, ind, testInd)
  
    INPUT
        V           Hyperbox lower bounds
        W           Hyperbox upper bounds
        ind         Index of extended hyperbox
        testInd     Index of hyperbox to test for overlap with the extended hyperbox

    OUTPUT
        dim         Result to be fed into contrG1, which is special numpy array

    """
    dim = np.array([])
    
#    if (V[ind] > W[ind]).any() == True:
#        return dim
    id_missing_ind_V = V[ind] == 1 + epsilon_missing_value
    id_missing_testInd_V = V[testInd] == 1 + epsilon_missing_value
    
    if np.array_equal(id_missing_ind_V, id_missing_testInd_V):
        id_missing_ind_W = W[ind] == -epsilon_missing_value
        id_missing_testInd_W = W[testInd] == -epsilon_missing_value
        if np.array_equal(id_missing_ind_W, id_missing_testInd_W) == False:
            # position of missing values in W is different in two hyperboxes => no overlap
            return dim
    else:
        # position of missing values in V is different in two hyperboxes => no overlap
        return dim
    
    id_non_missing = np.logical_and(~id_missing_ind_V, ~id_missing_ind_W)
    
    condWiWk = W[ind, id_non_missing] - W[testInd, id_non_missing] > 0
    condViVk = V[ind, id_non_missing] - V[testInd, id_non_missing] > 0
    condWkVi = W[testInd, id_non_missing] - V[ind, id_non_missing] > 0
    condWiVk = W[ind, id_non_missing] - V[testInd, id_non_missing] > 0

    c1 = ~condWiWk & ~condViVk & condWiVk
    c2 = condWiWk & condViVk & condWkVi
    c3 = condWiWk & ~condViVk
    c4 = ~condWiWk & condViVk
    c = c1 + c2 + c3 + c4

    ad = c.all()

    if ad == True:
        minimum = 1
        id_original = np.nonzero(id_non_missing)[0]
        for j in range(len(c)):
            i = id_original[j]
            if c1[j] == True:
                if minimum > W[ind, i] - V[testInd, i]:
                    minimum = W[ind, i] - V[testInd, i]
                    dim = np.array([1, i])
            
            elif c2[j] == True:
                if minimum > W[testInd, i] - V[ind, i]:
                    minimum = W[testInd, i] - V[ind, i]
                    dim = np.array([2, i])
            
            elif c3[j] == True:
                if minimum > (W[testInd, i] - V[ind,i]) and (W[testInd, i] - V[ind, i]) < (W[ind, i] - V[testInd, i]):
                    minimum = W[testInd, i] - V[ind, i]
                    dim = np.array([31, i])
                elif minimum > (W[ind, i] - V[testInd, i]):
                    minimum = W[ind, i] - V[testInd, i]
                    dim = np.array([32, i])
                    
            elif c4[j] == True:
                if minimum > (W[testInd, i] - V[ind, i]) and (W[testInd, i] - V[ind, i]) < (W[ind, i] - V[testInd, i]):
                    minimum = W[testInd, i] - V[ind, i]
                    dim = np.array([41, i])
                elif minimum > (W[ind, i] - V[testInd, i]):
                    minimum = W[ind, i] - V[testInd, i]
                    dim = np.array([42, i])
                
    return dim

def hyperboxContraction(V1, W1, newCD, testedInd, ind):
    """
    Adjusting min-max points of overlaping clusters (with meet halfway)

      V, W = hyperboxContraction(V,W,newCD,testedInd,ind)
  
    INPUT
      V1            Lower bounds of existing hyperboxes
      W1            Upper bounds of existing hyperboxes
      newCD         Special parameters, output from hyperboxOverlapTest
      testedInd     Index of hyperbox to test for overlap with the extended hyperbox
      ind           Index of extended hyperbox	
   
    OUTPUT
      V             Lower bounds of adjusted hyperboxes
      W             Upper bounds of adjusted hyperboxes
    
    """
    V = V1.copy()
    W = W1.copy()
    if newCD[0] == 1:
        W[ind, newCD[1]] = (V[testedInd, newCD[1]] + W[ind, newCD[1]]) / 2
        V[testedInd, newCD[1]] = W[ind, newCD[1]] + alpha
    elif newCD[0] == 2:
        V[ind, newCD[1]] = (W[testedInd, newCD[1]] + V[ind, newCD[1]]) / 2
        W[testedInd, newCD[1]] = V[ind, newCD[1]] - alpha
    elif newCD[0] == 31:
        V[ind, newCD[1]] = W[testedInd, newCD[1]] + alpha
    elif newCD[0] == 32:
        W[ind, newCD[1]] = V[testedInd, newCD[1]] - alpha
    elif newCD[0] == 41:
        W[testedInd, newCD[1]] = V[ind, newCD[1]] - alpha
    elif newCD[0] == 42:
        V[testedInd, newCD[1]] = W[ind, newCD[1]] + alpha
    
    return (V, W)


def isOverlap(V, W, ind, classId):
    """
    Checking overlap between hyperbox ind and remaning hyperboxes (1 vs many)
    
    INPUT
        V           Hyperbox lower bounds
        W           Hyperbox upper bounds
        ind         Index of the hyperbox to be checked for overlap
        classId     Class labels of hyperboxes
        
    OUTPUT
        False - no overlap,  True - overlap
    """
    
    if (V[ind] > W[ind]).any() == True:
        return False
    else:
        indcomp = np.nonzero((W >= V).all(axis = 1))[0] 	# examine only hyperboxes w/o missing dimensions, meaning that in each dimension upper bound is larger than lowerbound
        
        if len(indcomp) == 0:
            return False
        else:
            # testedHyperIndex = np.where(indcomp == ind)[0][0]
            # newInd = np.append(indcomp[0:testedHyperIndex], indcomp[testedHyperIndex + 1:])
            newInd = indcomp[indcomp != ind]

            if len(newInd) > 0:
                onesTemp = np.ones((len(newInd), 1))
                condWiWk = (onesTemp * W[ind] - W[newInd]) > 0
                condViVk = (onesTemp * V[ind] - V[newInd]) > 0
                condWkVi = (W[newInd] - onesTemp * V[ind]) > 0
                condWiVk = (onesTemp * W[ind] - V[newInd]) > 0
                
                #print(condWiWk.shape)
                
                c1 = ~condWiWk & ~condViVk & condWiVk
                c2 = condWiWk & condViVk & condWkVi
                c3 = condWiWk & ~condViVk
                c4 = ~condWiWk & condViVk
                
                c = c1 + c2 + c3 + c4
                
                ad = c.all(axis = 1)
                #print("Ad = ", np.nonzero(ad)[0].size)
                ind2 = newInd[ad]
                
                ovresult = (classId[ind2] != classId[ind]).any()
                    
                return ovresult
            else:
                return False
            
def modifiedIsOverlap(V, W, ind, classId):
    """
    Checking overlap between hyperbox ind and remaning hyperboxes (1 vs many)
    Only do overlap testing with hyperboxes belonging to other classes
    
    INPUT
        V           Hyperbox lower bounds
        W           Hyperbox upper bounds
        ind         Index of the hyperbox to be checked for overlap
        classId     Class labels of hyperboxes
        
    OUTPUT
        False - no overlap,  True - overlap
    """
    if (V[ind] > W[ind]).any() == True:
        return False
    else:
        indcomp = np.nonzero((W >= V).all(axis = 1))[0] 	# examine only hyperboxes w/o missing dimensions, meaning that in each dimension upper bound is larger than lowerbound
        
        if len(indcomp) == 0:
            return False
        else:
            class_indcomp = classId[indcomp]
            newInd = indcomp[(class_indcomp != classId[ind]) | (class_indcomp == UNLABELED_CLASS)] # get index of hyperbox representing different classes
            
            if len(newInd) > 0:
                onesTemp = np.ones((len(newInd), 1))
                condWiWk = (onesTemp * W[ind] - W[newInd]) > 0
                condViVk = (onesTemp * V[ind] - V[newInd]) > 0
                condWkVi = (W[newInd] - onesTemp * V[ind]) > 0
                condWiVk = (onesTemp * W[ind] - V[newInd]) > 0
                
                #print(condWiWk.shape)
                
                c1 = ~condWiWk & ~condViVk & condWiVk
                c2 = condWiWk & condViVk & condWkVi
                c3 = condWiWk & ~condViVk
                c4 = ~condWiWk & condViVk
                
                c = c1 + c2 + c3 + c4
                
                ad = c.all(axis = 1)
                #print("Ad = ", np.nonzero(ad)[0].size)
                ind2 = newInd[ad]
                
                ovresult = len(ind2) > 0
                    
                return ovresult
            else:
                return False
            
def directedIsOverlap(V, W, V_cmp, W_cmp):
    """
    Checking overlap between hyperbox [V_cmp, W_cmp] and remaning hyperboxes (1 vs many) representing other classes
    
    INPUT
        V           Hyperbox lower bounds of hyperboxes representing other classes compared to V_cmp
        W           Hyperbox upper bounds of hyperboxes representing other classes compared to W_cmp
        V_cmp       Minimum point of the compared hyperbox
        W_cmp       Maximum point of the compared hyperbox
        
    OUTPUT
        False - no overlap,  True - overlap
    """
    if (V is None) or (len(V) == 0):
        return False
    
    if (V_cmp > W_cmp).any() == True:
        return False
    else:
        onesTemp = np.ones((V.shape[0], 1))
        condWiWk = (onesTemp * W_cmp - W) > 0
        condViVk = (onesTemp * V_cmp - V) > 0
        condWkVi = (W - onesTemp * V_cmp) > 0
        condWiVk = (onesTemp * W_cmp - V) > 0
        
        #print(condWiWk.shape)
        
        c1 = ~condWiWk & ~condViVk & condWiVk
        c2 = condWiWk & condViVk & condWkVi
        c3 = condWiWk & ~condViVk
        c4 = ~condWiWk & condViVk
        
        c = c1 + c2 + c3 + c4
        
        ad = c.all(axis = 1)
        
        ovresult = ad.any()
            
        return ovresult
    
def directedIsOverlapMissingValChecking(V, W, V_cmp, W_cmp):
    """
    Checking overlap between hyperbox [V_cmp, W_cmp] and remaning hyperboxes (1 vs many) representing other classes
    
    INPUT
        V           Hyperbox lower bounds of hyperboxes representing other classes compared to V_cmp
        W           Hyperbox upper bounds of hyperboxes representing other classes compared to W_cmp
        V_cmp       Minimum point of the compared hyperbox
        W_cmp       Maximum point of the compared hyperbox
        
    OUTPUT
        False - no overlap,  True - overlap
    """
    if (V is None) or (len(V) == 0):
        return False
    
    if (V_cmp > W_cmp).any() == True:
        return False
    else:
        # Check if existing at least one hyperbox in V and W containing missing values
        if (V > W).any() == True:
            return False
        
        onesTemp = np.ones((V.shape[0], 1))
        condWiWk = (onesTemp * W_cmp - W) > 0
        condViVk = (onesTemp * V_cmp - V) > 0
        condWkVi = (W - onesTemp * V_cmp) > 0
        condWiVk = (onesTemp * W_cmp - V) > 0
        
        #print(condWiWk.shape)
        
        c1 = ~condWiWk & ~condViVk & condWiVk
        c2 = condWiWk & condViVk & condWkVi
        c3 = condWiWk & ~condViVk
        c4 = ~condWiWk & condViVk
        
        c = c1 + c2 + c3 + c4
        
        ad = c.all(axis = 1)
        
        ovresult = ad.any()
            
        return ovresult
            

def is_overlap_general_formulas(V, W, V_cmp, W_cmp, find_min_overlap=False):
    """
    Checking overlap between hyperbox [V_cmp, W_cmp] and remaning hyperboxes (1 vs many) representing other classes
    
    INPUT
        V           Hyperbox lower bounds of hyperboxes representing other classes compared to V_cmp
        W           Hyperbox upper bounds of hyperboxes representing other classes compared to W_cmp
        V_cmp       Minimum point of the compared hyperbox
        W_cmp       Maximum point of the compared hyperbox
        
        find_min_overlap
                + True:   Find the dimension with minimum overlap of all hyperboxes causing overlapping with [V_cmp, W_cmp]
                + False:  Only test whether existing overlap or not?
        
    OUTPUT
        if find_min_overlap == False:
            return False - no overlap,  True - overlap
        else:
            return:
                + is_overlap: False - no overlap,  True - overlap
                + hyperbox_ids_overlap: indices of hyperboxes overlap with [V_cmp, W_cmp] - numpy array
                + min_overlap_dimension: dimension with minimum overlap value > 0 corresponding to hyperboxes with id located in hyperbox_id_overlap 
                
                if is_overlap == False:
                    hyperbox_id_overlap = min_overlap_dimension = None
    """
    if (V is None) or (len(V) == 0):
        return False
    
    if (V_cmp > W_cmp).any() == True:
        return False
    else:
        yX = V.shape[0]
        V_cmp_tile = np.repeat([V_cmp], yX, axis=0)
        W_cmp_tile = np.repeat([W_cmp], yX, axis=0)
        
        overlap_mat = np.minimum(W_cmp_tile, W) - np.maximum(V_cmp_tile, V)
        
        overlap_hyperbox_vec = (overlap_mat >= 0).all(axis=1)
        
        is_overlap = overlap_hyperbox_vec.any()
        
        if find_min_overlap == False:
            return is_overlap
        else:
            if is_overlap == False:
                return (is_overlap, None, None)
            else:
                # Find the dimension with min overlap values (>0) of hyperboxes overlap with [V_cmp, W_cmp]
                hyperbox_ids_overlap = np.nonzero(overlap_hyperbox_vec)[0]
                
                V_cmp_tile = np.repeat([V_cmp], len(hyperbox_ids_overlap), axis=0)
                W_cmp_tile = np.repeat([W_cmp], len(hyperbox_ids_overlap), axis=0)
                
                overlap_value_mat =  np.minimum((W[hyperbox_ids_overlap] - V_cmp_tile), (W_cmp_tile - V[hyperbox_ids_overlap]))
                
                overlap_value_mat = np.where(overlap_value_mat == 0, np.nan, overlap_value_mat)
                
                min_overlap_dimension = np.nanargmin(overlap_value_mat, axis=1)
                
                return (is_overlap, hyperbox_ids_overlap, min_overlap_dimension)

def hyperbox_contraction_rfmm(V1, W1, classId, id_parent_box, id_child_box, overlap_dim, scale=0.001):
    """
    Adjusting min-max points of overlaping clusters
    
    The content is on the paper "A refined Fuzzy Min-Max Neural Network with New Learning Procedures for Pattern Classification"

      V, W = hyperbox_contraction_rfmm(V1, W1, id_parent_box, id_child_box, overlap_dim)
  
    INPUT
      V1            Lower bounds of existing hyperboxes
      W1            Upper bounds of existing hyperboxes
      classId       Class of existing hyperboxes
      id_parent_box List of indices of parent hyperboxes containing the child hyperbox
      id_child_box  Index of child hyperbox
      overlap_dim   The dimensions to make contraction	
   
    OUTPUT
      V             Lower bounds of adjusted hyperboxes
      W             Upper bounds of adjusted hyperboxes
    
    """
    V = V1.copy()
    W = W1.copy()
    class_id = classId.copy()
    
    for index, it in enumerate(id_parent_box):
        if V[it, overlap_dim[index]] < V[id_child_box, overlap_dim[index]] and W[id_child_box, overlap_dim[index]] < W[it, overlap_dim[index]]:
            Wj2 = W[it].copy()
            W[it, overlap_dim[index]] = V[id_child_box, overlap_dim[index]] - scale
            Vj2 = W[id_child_box].copy()
            Vj2[overlap_dim[index]] = W[id_child_box, overlap_dim[index]] + scale
            
            V = np.concatenate((V, Vj2.reshape(1, -1)), axis = 0)
            W = np.concatenate((W, Wj2.reshape(1, -1)), axis = 0)
            class_id = np.concatenate((class_id, [classId[it]]))       
    
    return (V, W, class_id)           
          
def improvedHyperboxOverlapTest(V, W, ind, testInd, Xh):
    """
    Hyperbox overlap test - 9 cases

      dim = overlapTest(V, W, ind, testInd)
  
    INPUT
        V           Hyperbox lower bounds
        W           Hyperbox upper bounds
        ind         Index of extended hyperbox
        testInd     Index of hyperbox to test for overlap with the extended hyperbox
        Xh          Current input sample being considered (used for case 9)

    OUTPUT
        dim         Result to be fed into contrG1, which is special numpy array

    """
    dim = np.array([]);
    xW = W.shape[1]
    
    condWiWk = W[ind, :] - W[testInd, :] > 0
    condViVk = V[ind, :] - V[testInd, :] > 0
    condWkVi = W[testInd, :] - V[ind, :] > 0
    condWiVk = W[ind, :] - V[testInd, :] > 0
    
    condEqViVk = V[ind, :] - V[testInd, :] == 0
    condEqWiWk = W[ind, :] - W[testInd, :] == 0

    c1 = ~condWiWk & ~condViVk & condWiVk
    c2 = condWiWk & condViVk & condWkVi
    c3 = condEqViVk & condWiVk & ~condWiWk
    c4 = ~condViVk & condWiVk & condEqWiWk
    c5 = condEqViVk & condWkVi & condWiWk
    c6 = condViVk & condWkVi & condEqWiWk
    c7 = ~condViVk & condWiWk
    c8 = condViVk & ~condWiWk
    c9 = condEqViVk & ~condViVk & condEqWiWk
    
    c = c1 + c2 + c3 + c4 + c5 + c6 + c7 + c8 + c9

    ad = c.all()

    if ad == True:
        minimum = 1
        for i in range(xW):
            if c1[i] == True:
                if minimum > W[ind, i] - V[testInd, i]:
                    minimum = W[ind, i] - V[testInd, i]
                    dim = np.array([1, i])
            
            elif c2[i] == True:
                if minimum > W[testInd, i] - V[ind, i]:
                    minimum = W[testInd, i] - V[ind, i]
                    dim = np.array([2, i])
            
            elif c3[i] == True:
                if minimum > (W[testInd, i] - V[ind,i]) and (W[testInd, i] - V[ind, i]) < (W[ind, i] - V[testInd, i]):
                    minimum = W[testInd, i] - V[ind, i]
                elif minimum > (W[ind, i] - V[testInd, i]):
                    minimum = W[ind, i] - V[testInd, i]
                
                dim = np.array([3, i])
                    
            elif c4[i] == True:
                if minimum > (W[testInd, i] - V[ind, i]) and (W[testInd, i] - V[ind, i]) < (W[ind, i] - V[testInd, i]):
                    minimum = W[testInd, i] - V[ind, i]
                elif minimum > (W[ind, i] - V[testInd, i]):
                    minimum = W[ind, i] - V[testInd, i]
                
                dim = np.array([4, i])
                
            elif c5[i] == True:
                if minimum > (W[testInd, i] - V[ind,i]) and (W[testInd, i] - V[ind, i]) < (W[ind, i] - V[testInd, i]):
                    minimum = W[testInd, i] - V[ind, i]
                elif minimum > (W[ind, i] - V[testInd, i]):
                    minimum = W[ind, i] - V[testInd, i]
                    
                dim = np.array([5, i])
            
            elif c6[i] == True:
                if minimum > (W[testInd, i] - V[ind,i]) and (W[testInd, i] - V[ind, i]) < (W[ind, i] - V[testInd, i]):
                    minimum = W[testInd, i] - V[ind, i]
                elif minimum > (W[ind, i] - V[testInd, i]):
                    minimum = W[ind, i] - V[testInd, i]
                    
                dim = np.array([6, i])
                
            elif c7[i] == True:
                if minimum > (W[testInd, i] - V[ind,i]) and (W[testInd, i] - V[ind, i]) < (W[ind, i] - V[testInd, i]):
                    minimum = W[testInd, i] - V[ind, i]
                    dim = np.array([71, i])
                elif minimum > (W[ind, i] - V[testInd, i]):
                    minimum = W[ind, i] - V[testInd, i]
                    dim = np.array([72, i])
                    
            elif c8[i] == True:
                if minimum > (W[testInd, i] - V[ind,i]) and (W[testInd, i] - V[ind, i]) < (W[ind, i] - V[testInd, i]):
                    minimum = W[testInd, i] - V[ind, i]
                    dim = np.array([81, i])
                elif minimum > (W[ind, i] - V[testInd, i]):
                    minimum = W[ind, i] - V[testInd, i]
                    dim = np.array([82, i])
                    
            elif c9[i] == True:
                if minimum > (W[testInd, i] - V[ind,i]):
                    minimum = W[testInd, i] - V[ind,i]
                
                if W[ind, i] == Xh[i]: # maximum point is expanded
                    dim = np.array([91, i])
                else: # minimum point is expanded
                    dim = np.array([92, i])
                    
                
    return dim


def improvedHyperboxContraction(V1, W1, newCD, testedInd, ind):
    """
    Adjusting min-max points of overlaping regions (9 cases)
    
      V, W = hyperboxContraction(V,W,newCD,testedInd,ind)
  
    INPUT
      V1            Lower bounds of existing hyperboxes
      W1            Upper bounds of existing hyperboxes
      newCD         Special parameters, output from improvedHyperboxOverlapTest
      testedInd     Index of hyperbox to test for overlap with the extended hyperbox
      ind           Index of extended hyperbox	
   
    OUTPUT
      V             Lower bounds of adjusted hyperboxes
      W             Upper bounds of adjusted hyperboxes
    
    """
    V = V1.copy()
    W = W1.copy()
    if newCD[0] == 1 or newCD[0] == 91:
        W[ind, newCD[1]] = (V[testedInd, newCD[1]] + W[ind, newCD[1]]) / 2
        V[testedInd, newCD[1]] = W[ind, newCD[1]]
    elif newCD[0] == 2 or newCD[0] == 92:
        V[ind, newCD[1]] = (W[testedInd, newCD[1]] + V[ind, newCD[1]]) / 2
        W[testedInd, newCD[1]] = V[ind, newCD[1]]
    elif newCD[0] == 3 or newCD[0] == 82:
        V[testedInd, newCD[1]] = W[ind, newCD[1]]
    elif newCD[0] == 4 or newCD[0] == 72:
        W[ind, newCD[1]] = V[testedInd, newCD[1]]
    elif newCD[0] == 5 or newCD[0] == 71:
        V[ind, newCD[1]] = W[testedInd, newCD[1]]
    elif newCD[0] == 6 or newCD[0] == 81:
        W[testedInd, newCD[1]] = V[ind, newCD[1]]
        
    return (V, W)