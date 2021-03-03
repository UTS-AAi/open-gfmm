# -*- coding: utf-8 -*-
"""
Created on Fri Aug 31 18:23:03 2018

@author: Thanh Tung Khuat

Hyperbox adjustment handling: overlap testing, hyperbox contraction

"""
import numpy as np
from functionhelper import UNLABELED_CLASS, epsilon_missing_value
from functionhelper.utils import hashing
from copy import deepcopy
from functionhelper.membershipcalc import memberG

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
            
def extendedModifiedIsOverlap(V, W, ind, classId):
    """
    Checking overlap between hyperbox ind and remaning hyperboxes (1 vs many)
    Only do overlap testing with hyperboxes belonging to other classes
    
    This function using shorted distance similarity for overlap checking
    
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
                b = memberG(W[ind], V[ind], V[newInd], W[newInd], 1)
                id_check_cat = np.nonzero(b == 1)[0] # id of overlapping hyperboxes
                if len(id_check_cat) == 0:
                    # No overlap
                    return False
                else:
                    return True
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
    
def extendedDirectedIsOverlap(V, W, A, noSampleInBox, V_cmp, W_cmp, A_cmp, noSampleInBox_cmp):
    """
    Checking overlap between hyperbox [V_cmp, W_cmp, A_cmp] and remaning hyperboxes (1 vs many) representing other classes
    
    INPUT
        V              Hyperbox lower bounds of hyperboxes representing other classes compared to V_cmp
        W              Hyperbox upper bounds of hyperboxes representing other classes compared to W_cmp
        A              Categorical values of hyperboxes representing other classes compared to A_cmp
        noSampleInBox  A vector save the number of samples contained in each hyperbox in the lists of V, W, A
        V_cmp          Minimum point of the compared hyperbox
        W_cmp          Maximum point of the compared hyperbox
        A_cmp          Categorical values of the compared hyperbox
        noSampleInBox_cmp   a scalar storing the number of hyperboxes included in the hyperbox [V_cmp, W_cmp, A_cmp]
        
    OUTPUT
        False - no overlap,  True - overlap
    """
    # Note that: if the compared hyperbox does not overlap with any other hyperboxes on the numerical dims, we do not need to check the categorical dims
    # The checking for the categorical dims is performed for only overlapped hyperboxes on the numerical dims
    if V_cmp is not None:
        # there exists the numerical feature
        # NOTE: We can improve this operation by a general overlap test cases using shorted distance measure for membership function
        if (V is None) or (len(V) == 0):
            return False
    
        if (V_cmp > W_cmp).any() == True:
            return False
        else:
#            onesTemp = np.ones((V.shape[0], 1))
#            condWiWk = (onesTemp * W_cmp - W) > 0
#            condViVk = (onesTemp * V_cmp - V) > 0
#            condWkVi = (W - onesTemp * V_cmp) > 0
#            condWiVk = (onesTemp * W_cmp - V) > 0
#            
#            #print(condWiWk.shape)
#            
#            c1 = ~condWiWk & ~condViVk & condWiVk
#            c2 = condWiWk & condViVk & condWkVi
#            c3 = condWiWk & ~condViVk
#            c4 = ~condWiWk & condViVk
#            
#            c = c1 + c2 + c3 + c4
#            
#            ad = c.all(axis = 1)
#            
#            if A_cmp is None:
#                return ad.any()
#            
#            if ad.any() == False:
#                return False
#            
#            # there is at least one overlapping region
#            id_check_cat = np.nonzero(ad)[0]
            
            # Improved overlap test version for general case:
            b = memberG(W_cmp, V_cmp, V, W, 1)
            id_check_cat = np.nonzero(b == 1)[0] # id of overlapping hyperboxes
            if len(id_check_cat) == 0:
                # No overlap
                return False
            
            if A_cmp is None:
                if len(id_check_cat) == 0:
                    return False
                else:
                    return True      
    else:
        id_check_cat = list(range(A.shape[0]))
    
    A = A[id_check_cat, :]
    noSampleInBox = noSampleInBox[id_check_cat]
    ndim_cat = A.shape[1]
    for ind, hyperbox in enumerate(A):
        is_overlap = np.ones(ndim_cat, dtype=bool)
        for j in range(ndim_cat):
            inter_cat_val = np.intersect1d(list(hyperbox[j].keys()), list(A_cmp[j].keys()))
            if len(inter_cat_val) > 0:
                diff_all = True
                for it in inter_cat_val:
                    if hyperbox[j][it] / noSampleInBox[ind] == A_cmp[j][it] / noSampleInBox_cmp:
                        # overlap on dimension j
                        diff_all = False
                        break
                
                if diff_all == True:
                    # No categorical value between two cat features having the same probability => Two hyperbox does not overlap
                    is_overlap[j] = False
                    break                       
            else:
                # All categorical values on the dim j are different between two hyperboxes => Two hyperboxes does not overlap
                is_overlap[j] = False
                break
            
        if is_overlap.all() == True:
            # All categorical dims overlap between two hyperboxes
            return True
        
    # No hyperbox overlaps with the considier hyperbox on the categorical features
    return False   
    
    
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
                
                overlap_value_mat =  np.minimum((W[hyperbox_ids_overlap] - V_cmp_tile), (W_cmp_tile - V[hyperbox_ids_overlap])).astype(np.float64)
                
                # Find indices of rows in matrix with all values being 0
                all_zero_row_id = ~overlap_value_mat.any(axis=1)
                
                for index, val in enumerate(all_zero_row_id):
                    if val == False:
                        overlap_value_mat[index] = np.where(overlap_value_mat[index] == 0, np.nan, overlap_value_mat[index])
                
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

def isOverlapCatOneOne(E1, F1, E2, F2):
    """
        Check overlap among cat features of two hyperboxes
    """
    if len(E1.shape) == 1:
        num_dim = len(E1)
    else:
        num_dim = E1.shape[1]
        E1 = E1.flatten()
        E2 = E2.flatten()
        F1 = F1.flatten()
        F2 = F2.flatten()

    for i in range(num_dim):
        if (E1[i] != E2[i]) and (E1[i] != F2[i]) and (F1[i] != E2[i]) and (F1[i] != F2[i]):
            return False
        else:
            if (E1[i] != E2[i]) and (F1[i] == default_categorical_value) and (F2[i] == default_categorical_value):
                return False
    # two hyperboxes with cat features overlap when all dims overlap    
    return True

def isOverlapCatOneMany(E1, F1, E, F, id_cons=[]):
    """
        Check overlap among cat features of hyperbox [E1, F1] and all hyperboxes in list E and F with indices located in id_cons
    """
    if len(id_cons) > 0:
        for i in id_cons:
            if isOverlapCatOneOne(E1, F1, E[i], F[i]) == True:
                return True
            
    return False          

def hyperboxOverlapTestCatFeature(E, F, ind, testInd, X_e, X_f, sim_vec, pre_overlap_id):
    """
    Hyperbox overlap test

      dim = overlapTest(V, W, ind, testInd)
  
    INPUT
        V           Hyperbox lower bounds
        W           Hyperbox upper bounds
        ind         Index of extended hyperbox
        testInd     Index of hyperbox to test for overlap with the extended hyperbox
        X_e         Lower bounds for categorical features in input training samples
        X_f         Upper bounds for categorical features in input training samples
                   (X_e, X_f) are used to identify all unique values of each input feature
        sim_vec     A similarity vector with each element being a mapping table of similarity values among feature value pairs for each categorical feature
        pre_overlap_id   Id of hyperboxes overlapping with extended hyperbox but resolved by changing values of only cat features
                         => when replacing overlap area by other values, we are not allowed to create overlap regions with hyperboxes having id in pre_overlap_id

    OUTPUT
        dim         An array contains [dimension, [replaced values]]
                    dim = [] => No overlap
                    + dimension = -1 => Overlap but cannot find the suitable value to replace for any dimension      
                    + [replaced values] contains two values for lower bound and upper bound, if None => No change in that bound

    """
    dim = []
    isOverlap = isOverlapCatOneOne(E[ind], F[ind], E[testInd], F[testInd])
        
    if isOverlap == True:
        X_cat = np.vstack((X_e, X_f))
        # resolving overlap by contracting the one dimesion of the extended hyperbox
        min_change = 2
        dimension = -1
        replaced_val = [None, None]
        for i in range(E.shape[1]):
            unq_val_fe_i = np.unique(X_cat[:, i])
            min_change_i = 1
            newVal_lower = None
            newVal_upper = None
            is_changed_lower = False
            is_changed_upper = False
            is_overlap_lower = False
            is_overlap_upper = False
            cur_dist = sim_vec[i][hashing(E[ind, i], F[ind, i])]
            if (E[ind, i] == E[testInd, i]) or (E[ind, i] == F[testInd, i]):
                # replace the value at E[ind, i]
                # get the other values different from E[ind, i]
                is_overlap_lower = True
                replaceable_vals = unq_val_fe_i[unq_val_fe_i != E[ind, i]]
                if len(replaceable_vals) > 0:
                    # find the value such that similarity(replaceable_val, F[ind, i]) < similarity(E[ind, i], F[ind, i]) and the change is the minimum
                    # Note the replaced value is also different from E[testInd, i] and F[testInd, i]
                    for val in replaceable_vals:
                        if (val != E[testInd, i]) and (val != F[testInd, i]):
                            new_dist = sim_vec[i][hashing(val, F[ind, i])]
                            if (min_change_i > (cur_dist - new_dist)) and (cur_dist >= new_dist):
                                tmp_E_ind = deepcopy(E[ind])
                                tmp_E_ind[i] = val
                                if isOverlapCatOneMany(tmp_E_ind, F[ind], E, F, pre_overlap_id) == False:
                                    min_change_i = cur_dist - new_dist
                                    newVal_lower = val
                                    is_changed_lower = True
                                
            if (F[ind, i] != default_categorical_value) and ((F[ind, i] == E[testInd, i]) or (F[ind, i] == F[testInd, i]) ) :
                # replace the value at F[ind, i]
                is_overlap_upper = True
                if (is_overlap_lower == False) or ((is_overlap_lower == True) and (is_changed_lower == True)):
                    replaceable_vals = unq_val_fe_i[unq_val_fe_i != F[ind, i]]
                    if len(replaceable_vals) > 0:
                        # find the value such that similarity(E[ind, i], replaceable_val) < similarity(E[ind, i], F[ind, i]) and the change is the minimum
                        # Note the replaced value is also different from E[testInd, i] and F[testInd, i]
                        min_change_i = 1
                        for val in replaceable_vals:
                            if (val != E[testInd, i]) and (val != F[testInd, i]):
                                if is_changed_lower == True:
                                    new_dist = sim_vec[i][hashing(newVal_lower, val)]
                                else:
                                    new_dist = sim_vec[i][hashing(E[ind, i], val)]
                                if (min_change_i > (cur_dist - new_dist)) and (cur_dist >= new_dist):
                                    tmp_F_ind = deepcopy(F[ind])
                                    tmp_F_ind[i] = val
                                    tmp_E_ind = deepcopy(E[ind])
                                    if is_changed_lower == True:
                                        tmp_E_ind[i] = newVal_lower
                                    if isOverlapCatOneMany(tmp_E_ind, tmp_F_ind, E, F, pre_overlap_id) == False:
                                        min_change_i = cur_dist - new_dist
                                        newVal_upper = val
                                        is_changed_upper = True
                    
            is_changed = True  
            if (is_overlap_lower == True) and (is_changed_lower == False):
                is_changed = False
            if (is_overlap_upper == True) and (is_changed_upper == False):
                is_changed = False
            
            if (is_changed == True) and (min_change_i < min_change):
                min_change = min_change_i
                dimension = i
                replaced_val[0] = newVal_lower
                replaced_val[1] = newVal_upper
                        
        dim = [dimension, replaced_val]
        
    return dim

def hyperboxContractionCatFeature(E, F, newCD, ind, inplace=False):
    """
    Adjusting min-max points of overlaping clusters for categorical features

    INPUT
      E             Lower bounds of existing hyperboxes for categorical features
      F             Upper bounds of existing hyperboxes for categorical features
      newCD         Special parameters, output from hyperboxOverlapTestCatFeatures
      ind           Index of extended hyperbox
      inplace       Change directly in E and F or not?
                      + True: change directly
                      + No: create a copy before changing
      
    OUTPUT
      E             Lower bounds of adjusted hyperboxes for categorical features
      F             Upper bounds of adjusted hyperboxes for categorical features
    
    """
    if inplace == False:
        E = deepcopy(E)
        F = deepcopy(F)
    
    if len(newCD) > 0 and newCD[0] != -1:
        if newCD[1][0] is not None:
            E[ind, newCD[0]] = newCD[1][0]
        if newCD[1][1] is not None:
            F[ind, newCD[0]] = newCD[1][1]
    
    return (E, F)  

def overlapResolving(Vi, Wi, class_i, Vk, Wk, class_k):
    """
        This function is to resolve overlap between two hyperboxes Bi and Bk
    Parameters
    ----------
    Vi : numpy array
        Minimum point of hyperbox Bi.
    Wi : numpy array
        Maximum point of hyperbox Bi.
    class_i: integer
        Class of hyperbox Bi
    Vk : numpy array
        Minimum point of hyperbox Bk.
    Wk : numpy array
        Maximum point of hyperbox Bk.
    class_k: integer
        Class of hyperbox Bk

    Returns
    -------
    Min-max points of Bi and Bk resolved overlap regions
    """ 
    if class_i == class_k:
        return (Vi, Wi, Vk, Wk)

    delta_j = 2
    case = -1
    min_j = -1
    for j in range(len(Vi)):
        if (Vi[j] <= Vk[j]) and (Vk[j] < Wi[j]) and (Wi[j] <= Wk[j]):
            if delta_j > Wi[j] - Vk[j]:
                delta_j = Wi[j] - Vk[j]
                min_j = j
                case = 1
        elif (Vk[j] <= Vi[j]) and (Vi[j] < Wk[j]) and (Wk[j] <= Wi[j]):
            if delta_j > Wk[j] - Vi[j]:
                delta_j = Wk[j] - Vi[j]
                min_j = j
                case = 2
        elif (Vi[j] < Vk[j]) and (Wk[j] < Wi[j]):
            if (delta_j > (Wk[j] - Vi[j])) and ((Wk[j] - Vi[j]) <= (Wi[j] - Vk[j])):
                delta_j = Wk[j] - Vi[j]
                min_j = j
                case = 31
            elif delta_j > (Wi[j] - Vk[j]):
                delta_j = Wi[j] - Vk[j]
                min_j = j
                case = 32
        elif (Vk[j] < Vi[j]) and (Wi[j] < Wk[j]):
            if (delta_j > (Wk[j] - Vi[j])) and ((Wk[j] - Vi[j]) <= (Wi[j] - Vk[j])):
                delta_j = Wk[j] - Vi[j]
                min_j = j
                case = 41
            elif delta_j > (Wi[j] - Vk[j]):
                delta_j = Wi[j] - Vk[j]
                min_j = j
                case = 42
                
    #print("case = ", case, ", min = ", min_j)
                
    if case == 1:
        tmp = (Wi[min_j] + Vk[min_j]) / 2
        Wi[min_j] = tmp
        Vk[min_j] = tmp  + alpha
    elif case == 2:
        tmp = (Wk[min_j] + Vi[min_j]) / 2
        Vi[min_j] =  tmp + alpha
        Wk[min_j] = tmp
    elif case == 31:
        Vi[min_j] = Wk[min_j] + alpha
    elif case == 32:
        Wi[min_j] = Vk[min_j] - alpha
    elif case == 41:
        Wk[min_j] = Vi[min_j] - alpha
    elif case == 42:
        Vk[min_j] = Wi[min_j] + alpha
    else:
        print("WWWWWORING")
        
    return (Vi, Wi, Vk, Wk)