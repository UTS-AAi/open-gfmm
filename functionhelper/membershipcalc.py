# -*- coding: utf-8 -*-
"""
Created on Fri Aug 31 11:47:45 2018

@author: Thanh Tung Khuat

Fuzzy membership calculation

"""
import numpy as np
from copy import deepcopy
from functionhelper.utils import hashing, hashing_mat
from functionhelper import default_categorical_value

def extract_mem_for_classes(V, W, classId, XlT, XuT, gama = 1):
    """
    Extract membership values for all classes with respect to each testing sample
    This function works for only continous or numeric features
    
    Parameters
    ----------
    V : 2D numpy array
        Lower bounds of hyperboxes in the trained model.
    W : 2D numpy array
        Upper bounds of hyperboxes in the trained model.
    classId : 1D numpy array
        Class labels of hyperboxes in the trained model.
    XlT : 2D numpy array, each row is a sample
        Lower bounds of testing samples.
    XuT : 2D numpy array, each row is a sample
        Upper bounds of testing samples.
    gama : float, optional
        gamma value used in the membership function. The default is 1.
    
    Returns
    -------
    2D numpy array. Each row is a list of membership values. Each column is a class label sorted in an ascending order
    """
    if len(XlT.shape) == 1:
        XlT = XlT.reshape(1, -1)
    if len(XuT.shape) == 1:
        XuT = XuT.reshape(1, -1)
        
    #initialization
    yX = XlT.shape[0]
    mem_matrix = []
    class_vals = np.unique(classId)
    # classifications
    for i in range(yX):
        mem = memberG(XlT[i, :], XuT[i, :], V, W, gama) # calculate memberships for all hyperboxes
        class_c_mem = []
        for c in class_vals:
            id_c = np.nonzero(classId == c)[0]
            class_c_mem.append(mem[id_c].max())
            
        mem_matrix.append(class_c_mem)
        
    return np.array(mem_matrix)

def memberG(X_l, X_u, V, W, g, oper = 'min'):
    """
    Function for membership calculation
    
        b = memberG(X_l, X_u, V, W, g, oper)
 
   INPUT
     X_l        Input data lower bounds (a row vector with columns being features)
     X_u        Input data upper bounds (a row vector with columns being features)
     V          Hyperbox lower bounds
     W          Hyperbox upper bounds
     g          User defined sensitivity parameter 
     oper       Membership calculation operation: 'min' or 'prod' (default: 'min')
  
   OUTPUT
     b			Degrees of membership of the input pattern

   DESCRIPTION
    	Function provides the degree of membership b of an input pattern X (in form of upper bound Xu and lower bound Xl)
        in hyperboxes described by min points V and max points W. The sensitivity parameter g regulates how fast the 
        membership values decrease when an input pattern is separeted from hyperbox core.

    """
    
    yW = W.shape[0]
    onesMat = np.ones((yW, 1))
    violMax = 1 - fofmemb(onesMat * X_u - W, g)
    violMin = 1 - fofmemb(V - onesMat * X_l, g)

    if oper == 'prod':
        b = np.prod(np.minimum(violMax, violMin), axis = 1)
    else:
        b = np.minimum(violMax, violMin).min(axis = 1)
    
    return b

def memberG_extended_iol(X_l, X_u, X_cat, V, W, A, g, alpha = 0.5):
    """
        Membership function for the samples with both categorical and numerical features in the extended improved online learning algorithm
        
        INPUT:
            X_l        Input data lower bounds (a row vector with columns being features)
            X_u        Input data upper bounds (a row vector with columns being features)
            X_cat      Input data of categorical features
            V          Hyperbox lower bounds
            W          Hyperbox upper bounds
            A          Hyperbox bounds of categorical features
            g          User defined sensitivity parameter in the membership function of the numerical features
            alpha      The trade-off weighting factor between categorical features and numerical features (default: 0.5)
    """
    if (X_l is not None) and (X_cat is not None):
        b_numeric = np.array(memberG(X_l, X_u, V, W, g))
        b_cat = memberOnlyCat_extend_iol(X_cat, A)
        
        b = alpha * b_numeric + (1 - alpha) * b_cat
    else:
        if X_l is not None:
            b = np.array(memberG(X_l, X_u, V, W, g))
        else:
            b = memberOnlyCat_extend_iol(X_cat, A)
    
    return b

def memberOnlyCat_extend_iol(X_cat, A):
    """
    Compute membership function for only categorical features in the extened improved online learning algorithm
    
        X_cat      Input data of categorical features
        A          Hyperbox bounds of categorical features
    """
    ndim = len(X_cat)
    b = np.zeros(A.shape[0])
    
    for ind, hyperbox in enumerate(A):
        s = 0
        no_sample = sum(hyperbox[0].values())
        for j in range(ndim):
            if X_cat[j] in hyperbox[j]:
                s = s + hyperbox[j][X_cat[j]] / no_sample
        
        b[ind] = s / ndim
        
    return b
    

def memberG_with_cat_one_hot_features(X_l, X_u, X_d, V, W, D, g, oper = 'min'):
    """
    Function for membership value computation with categorical features
        
        INPUT:
            X_l        Input data lower bounds (a row vector with columns being features)
            X_u        Input data upper bounds (a row vector with columns being features)
            X_d        Input data of categorical features with each item being a vector of binary values
            V          Hyperbox lower bounds
            W          Hyperbox upper bounds
            D          Hyperbox bounds of categorical features
            g          User defined sensitivity parameter 
            oper       Membership calculation operation: 'min' or 'prod' (default: 'min')
          
        OUTPUT
             b			Degrees of membership of the input pattern
    """
    if (X_l is not None) and (X_d is not None):
        b_numeric = np.array(memberG(X_l, X_u, V, W, g, oper))
        b_cat = memberOnlyCat(X_d, D)
        
        b = (b_numeric + b_cat) / 2
    else:
        if X_l is not None:
            b = np.array(memberG(X_l, X_u, V, W, g, oper))
        else:
            b = memberOnlyCat(X_d, D)
    
    return b

def memberG_cat_freq(X_e, X_f, E, F, sim_vec):
    """
        Compute membership function for only categorical features based on frequency values
    """
    violMin = 1 - fcatmem(X_e, E, sim_vec)
    violMax = 1 - fcatmem(X_f, F, sim_vec)
    #print("violMin = ", violMin)
    #print("violMax = ", violMin)
    # Note: If one of two values with respect to E and F is 1 => return 1
    # Note: If F gets default values => its mem = 1, then the mem value depends on the E
    # If X_e = F => return 1, and in this case violMin = 1 to mem = 1
    violMax[(violMin == 1) & (E != default_categorical_value)] = 1
    violMin[(violMax == 1) & (F != default_categorical_value)] = 1
    
    b = np.minimum(violMax, violMin).min(axis = 1)
    
    return b

def memberG_with_frequency_based_cat_features(X_l, X_u, X_e, X_f, V, W, E, F, sim_vec, g, oper='min'):
    """
        This function is to compute the membership function of an input pattern with respect to all hyperboxes (including continous and categorical features)
            The membership value for categorical features is computed based on frequency values
            
        INPUT:
            X_l: Lower bounds of input pattern with continous features (a row vector with columns being features)
            X_u: Upper bounds of input pattern with continous features
            Xe: Lower bounds of input pattern with categorical features
            Xf: Upper bounds of input pattern with categorical features
            V:  A matrix contains all lower bounds of continous features
            W: A matrix contains all upper bounds of continous features
            E: A matrix contains all lower bounds of categorical features
            F: A matrix contains all upper bounds of categorical features
            sim_vec: a vector of mapping values of all features. Each element in this vector is a mapping table of similarity values for pairs of values of this feature
            g:  User defined sensitivity parameter 
            oper: Membership calculation operation: 'min' or 'prod' (default: 'min')
            
        OUTPUT
             b			Degrees of membership of the input pattern
    """
    if (X_l is not None) and (X_e is not None):
        b_con = np.array(memberG(X_l, X_u, V, W, g, oper))
        b_cat = np.array(memberG_cat_freq(X_e, X_f, E, F, sim_vec))

        b = np.minimum(b_con, b_cat)
    else:
        if (X_l is not None):
            b = np.array(memberG(X_l, X_u, V, W, g, oper))
        else:
            b = np.array(memberG_cat_freq(X_e, X_f, E, F, sim_vec))
    
    return b
  
def memberG_with_selected_features(X_l, X_u, V, W, g):
    """
        This function computes membership values of an input pattern and existing hyperboxes
        It also returns features with maximum membership values for corresponding hyperboxes
        
        INPUT
             X_l        Input data lower bounds (a row vector with columns being features)
             X_u        Input data upper bounds (a row vector with columns being features)
             V          Hyperbox lower bounds
             W          Hyperbox upper bounds
             g          User defined sensitivity parameter
             
        OUTPUT
             b			Degrees of membership of the input pattern
             f          Features with maximum membership values
    """
    yW = W.shape[0]
    onesMat = np.ones((yW, 1))
    violMax = 1 - fofmemb(onesMat * X_u - W, g)
    violMin = 1 - fofmemb(V - onesMat * X_l, g)
    
    tmp = np.minimum(violMax, violMin)
    b = tmp.min(axis = 1)
    #f = tmp.argmin(axis = 1) # only get first feature with the maximum membership value
    # find all indices with maximum membership values
    f = []
    for i in range(tmp.shape[0]):
        sk = [j for j, it in enumerate(tmp[i]) if it == b[i]]
        f.append(sk)
        
    return (b, f)   
    
def fofmemb(x, gama):

    """
    fofmemb - ramp threshold function for fuzzy membership calculation

        f = fofmemb(x,gama)
  
   INPUT
     x			Input data matrix (rows = objects, columns = attributes)
     gama		Steepness of membership function
  
   OUTPUT
     f			Fuzzy membership values

   DESCRIPTION
    	f = 1,     if x*gama > 1
    	x*gama,    if 0 =< x*gama <= 1
    	0,         if x*gama < 0
    """

    if np.size(gama) > 1: 
        p = x*(np.ones((x.shape[0], 1))*gama)
    else:
        p = x*gama

    f = ((p >= 0) * (p <= 1)).astype(np.float64) * p + (p > 1).astype(np.float64)
    
    return f

def simpsonMembership(Xh, V, W, g):
    """
    Function for membership calculation
    
        b = simpsonMembership(Xh, V, W, g)
 
   INPUT
     Xh         Input data (a row vector with columns being features)
     V          Hyperbox lower bounds
     W          Hyperbox upper bounds
     g          User defined sensitivity parameter 
  
   OUTPUT
     b			Degrees of membership of the input pattern

   DESCRIPTION
    	Function provides the degree of membership b of an input pattern X (in form of upper bound Xu and lower bound Xl)
        in hyperboxes described by min points V and max points W. This function uses the Simpson's method. The sensitivity parameter g regulates how fast the 
        membership values decrease when an input pattern is separeted from hyperbox core.

    """
    yW, xW = W.shape
    Xh_mat = np.ones((yW, 1)) * Xh
    zeros_mat = np.zeros((yW, xW))
    
    violMax1 = np.maximum(zeros_mat, 1 - np.maximum(zeros_mat, simpsonMin(Xh_mat - W, g)))
    violMax2 = np.maximum(zeros_mat, 1 - np.maximum(zeros_mat, simpsonMin(V - Xh_mat, g)))
    
    violMat = (violMax1 + violMax2)
    
    b = np.sum(violMat, axis=1) / (2 * xW)
    
    return b


def simpsonMin(x, gamma):
    """
    Min function for fuzzy membership calculation

        f = simpsonMin(x, gamma)
  
   INPUT
     x			Input data matrix (rows = objects, columns = attributes)
     gamma	Steepness of membership function
  
   OUTPUT
     f			function's values

   DESCRIPTION
    	f = gamma * min(1, x)
        
    """
    yX, xX = x.shape
    
    if np.size(gamma) > 1: 
        f = (np.ones((yX, 1)) * gamma) * np.minimum(np.ones((yX, xX)), x)
    else:
        f = gamma * np.minimum(np.ones((yX, xX)), x)

    return f
    

#def asym_similarity_one_many(Xl_k, Xu_k, V, W, g = 1, asym_oper = 'max', oper_mem = 'min'):
#    """
#    Calculate the asymetrical similarity value of the k-th hyperbox (lower bound - Xl_k, upper bound - Xu_k) and 
#    hyperboxes having lower and upper bounds stored in two matrix V and W respectively
#    
#    INPUT
#        Xl_k        Lower bound of the k-th hyperbox
#        Xu_k        Upper bound of the k-th hyperbox
#        V           Lower bounds of other hyperboxes
#        W           Upper bounds of other hyperboxes
#        g           User defined sensitivity parameter 
#        asym_oper   Use 'min' or 'max' (default) to compute the asymetrical similarity value
#        oper_mem    operator used to compute the membership value, 'min' or 'prod'
#        
#    OUTPUT
#        b           similarity values of hyperbox k with all hyperboxes having lower and upper bounds in V and W
#    
#    """
#    numHyperboxes = W.shape[0]
#    b = np.empty(numHyperboxes, dtype = float)
#    
#    for k in range(numHyperboxes):
#        violMax1 = 1 - fofmemb(np.ones((1, 1)) * (Xu_k - W[k]), g)
#        violMin1 = 1 - fofmemb((V[k] - Xl_k) * np.ones((1,1)), g)
#        
#        violMax2 = 1 - fofmemb(np.ones((1, 1)) * (W[k] - Xu_k), g)
#        violMin2 = 1 - fofmemb((Xl_k - V[k]) * np.ones((1,1)), g)
#        
#        if oper_mem == 'prod':
#            b1 = np.prod(np.minimum(violMax1, violMin1), axis = 1)[0]
#            b2 = np.prod(np.minimum(violMax2, violMin2), axis = 1)[0]
#        else:
#            b1 = np.minimum(violMax1, violMin1).min(axis = 1)[0]
#            b2 = np.minimum(violMax2, violMin2).min(axis = 1)[0]
#            
#        if asym_oper == 'max':
#            b[k] = np.maximum(b1, b2)
#        else:
#            b[k] = np.minimum(b1, b2)
#    
#    return b
    
def asym_similarity_one_many(Xl_k, Xu_k, V, W, g = 1, asym_oper = 'max', oper_mem = 'min'):
    """
    Calculate the asymetrical similarity value of the k-th hyperbox (lower bound - Xl_k, upper bound - Xu_k) and 
    hyperboxes having lower and upper bounds stored in two matrix V and W respectively
    
    INPUT
        Xl_k        Lower bound of the k-th hyperbox
        Xu_k        Upper bound of the k-th hyperbox
        V           Lower bounds of other hyperboxes
        W           Upper bounds of other hyperboxes
        g           User defined sensitivity parameter 
        asym_oper   Use 'min' or 'max' (default) to compute the asymetrical similarity value
        oper_mem    operator used to compute the membership value, 'min' or 'prod'
        
    OUTPUT
        b           similarity values of hyperbox k with all hyperboxes having lower and upper bounds in V and W
    
    """
    numHyperboxes = W.shape[0]
    
    Vk = np.tile(Xl_k, [numHyperboxes, 1])
    Wk = np.tile(Xu_k, [numHyperboxes, 1])
    
    violMax1 = 1 - fofmemb(Wk - W, g)
    violMin1 = 1 - fofmemb(V - Vk, g)
    
    violMax2 = 1 - fofmemb(W - Wk, g)
    violMin2 = 1 - fofmemb(Vk - V, g)
    
    if oper_mem == 'prod':
        b1 = np.prod(np.minimum(violMax1, violMin1), axis = 1)
        b2 = np.prod(np.minimum(violMax2, violMin2), axis = 1)
    else:
        b1 = np.minimum(violMax1, violMin1).min(axis = 1)
        b2 = np.minimum(violMax2, violMin2).min(axis = 1)
        
    if asym_oper == 'max':
        b = np.maximum(b1, b2)
    else:
        b = np.minimum(b1, b2)
    
    return b

def asym_similarity_one_many_with_missing_value(Xl_k, Xu_k, V, W, g = 1, asym_oper = 'max', oper_mem = 'min'):
    """
    Calculate the asymetrical similarity value of the k-th hyperbox (lower bound - Xl_k, upper bound - Xu_k) and 
    hyperboxes having lower and upper bounds stored in two matrix V and W respectively
    
    Input data of this method contain missing values. Do not using minimum and maximum when passing them to the method, keep them the same as input
    
    INPUT
        Xl_k        Lower bound of the k-th hyperbox
        Xu_k        Upper bound of the k-th hyperbox
        V           Lower bounds of other hyperboxes
        W           Upper bounds of other hyperboxes
        g           User defined sensitivity parameter 
        asym_oper   Use 'min' or 'max' (default) to compute the asymetrical similarity value
        oper_mem    operator used to compute the membership value, 'min' or 'prod'
        
    OUTPUT
        b           similarity values of hyperbox k with all hyperboxes having lower and upper bounds in V and W
    
    """
    numHyperboxes = W.shape[0]
    
    Vk = np.tile(Xl_k, [numHyperboxes, 1])
    Wk = np.tile(Xu_k, [numHyperboxes, 1])
    
    violMax1 = 1 - fofmemb(Wk - np.maximum(W, V), g)
    violMin1 = 1 - fofmemb(np.minimum(V, W) - Vk, g)
    
    violMax2 = 1 - fofmemb(W - np.maximum(Wk, Vk), g)
    violMin2 = 1 - fofmemb(np.minimum(Vk, Wk) - V, g)
    
    if oper_mem == 'prod':
        b1 = np.prod(np.minimum(violMax1, violMin1), axis = 1)
        b2 = np.prod(np.minimum(violMax2, violMin2), axis = 1)
    else:
        b1 = np.minimum(violMax1, violMin1).min(axis = 1)
        b2 = np.minimum(violMax2, violMin2).min(axis = 1)
        
    if asym_oper == 'max':
        b = np.maximum(b1, b2)
    else:
        b = np.minimum(b1, b2)
    
    return b

def count_num_bit_one_cat (v):
    """
        This function is to count number of bit 1 in all items in v
    """
    tmp = [np.any(i) for i in v]
    
    return np.sum(tmp)
    
def memberOnlyCat(Xcat, D):
    """
    This function is to compute the membership value of the categorical features and all categorical features of hyperboxes model stored in D
    
        INPUT:
            Xcat: Categorical features of the input pattern, each feature is save in an array of one-hot-encoding values
            D: Categorical features of all hyperboxes
            
        OUTPUT:
            An array storing degrees of membership of the input pattern
    """
#    if len(Xcat.shape) == 1:
#        if Xcat.shape[0] > 1:
#            Xcat = deepcopy(Xcat).reshape(1, -1)
#        else:
#            Xcat_tmp = np.zeros((1, 1), dtype=np.object)
#            Xcat_tmp.fill(deepcopy(Xcat))
#            Xcat = Xcat_tmp
#    if len(Xcat.shape) == 1:
#         Xcat = deepcopy(Xcat).reshape(1, -1)
    
    if len(D.shape) == 1:
        D = deepcopy(D).reshape(-1, 1)
    
    mem_bit_and = np.bitwise_and(D, Xcat)
    
    num_cat_feature = D.shape[1]
    
    # Count number of bit 1 in each cat feature. For each cat feature, if there is at least one True in the one-hot-vector => return 1
    mem_val = [count_num_bit_one_cat(i) / num_cat_feature for i in mem_bit_and]

    mem_val = np.array(mem_val)
    
    return mem_val

def fcatmem(Xe, E, sim_vec):
    """
        This function to identify the similarity value on each dim for each pair of E and Xe
    """
    Xe_mat = np.ones((E.shape[0], 1)) * Xe
    
    val_mat = hashing_mat(Xe_mat, E)
    
    val_mapping = np.array([sim_vec[idj].get(j, 0) for i in val_mat for idj, j in enumerate(i)]).reshape(val_mat.shape)
    
    return val_mapping

def exp_func(X, V, W, g):
    """
    Compute exponential of base e for membership function created by X, V, W, and g

    Parameters
    ----------
    X : 1D numpy array
        Input patterns, cols: features.
    V : 2D numpy array
        Lower bounds of all hyperboxes.
    W : 2D numpy array
        Upper bounds of all hyperboxes.
    g : 1D numpy array
        gamma value.

    Returns
    -------
    1D numpy array corresponding to membership values between X and each hyperbox.

    """
    X_mat = np.ones((V.shape[0], 1)) * X
    m = V.shape[0]
    n = V.shape[1]
    epower = np.ones((m, n))
    if np.size(g) == 1:
        g = [g] * n
    for i in range(m):
        for j in range(n):
            if X_mat[i, j] < V[i, j]:
                tmp = np.exp(-g[j] * (X_mat[i, j] - V[i, j]) * (X_mat[i, j] - V[i, j]))
            elif (V[i, j] <= X_mat[i, j]) and (X_mat[i, j] <= W[i, j]):
                tmp = 1
            else:
                tmp = np.exp(-g[j] * (X_mat[i, j] - W[i, j]) * (X_mat[i, j] - W[i, j]))
            
            epower[i, j] = tmp
            
    return epower

def gaussian_memb(X_l, X_u, V, W, g):
    """
    Gaussian membership function

    Parameters
    ----------
        X_l   1D numpy array   
            Input data lower bounds (a row vector with columns being features)
        X_u   1D numpy array
            Input data upper bounds (a row vector with columns being features)
        V     2D numpy array  
            Hyperbox lower bounds
        W     2D numpy array
            Hyperbox upper bounds
        g     float or 1D numpy array    
            User defined sensitivity parameter 

    Returns
    -------
        1D numpy array contains membership values from the input pattern to each hyperbox

    """
    violMax = exp_func(X_u, V, W, g)
    violMin = exp_func(X_l, V, W, g)

    b = np.minimum(violMax, violMin).min(axis = 1)
    
    return b