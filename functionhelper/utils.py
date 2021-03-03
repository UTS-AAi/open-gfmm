# -*- coding: utf-8 -*-
import numpy as np
import multiprocessing

def hashing(a, b):
    """
        This is a function to transform a pair of positive integer numbers into a unique number and commutation
    """
    c = max(a, b) * (max(a, b) + 1) / 2 + min(a, b)
    return c

def hashing_mat(A, B):
    """
     This is a function to transform each pairs of items in two matrices A and B to a unique number
     
     Return a matrix
    """
    C = np.maximum(A, B) * (np.maximum(A, B) + 1) / 2 + np.minimum(A, B)
    
    return C

def get_num_cpu_cores():
    num_cores = multiprocessing.cpu_count()
    return num_cores
