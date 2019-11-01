# -*- coding: utf-8 -*-
"""
Created on Sat Oct 12 15:51:26 2019

@author: Thanh Tung Khuat
"""
import numpy as np
import random as rd

class BaseIndividual(object):
    
    def __init__(self, dims):
        self.n_dims = dims
        self.vals = np.ones(self.n_dims, dtype=float)
        
    def set_val(self, ith, val):
        self.vals[ith] = val
        
    def set_all_val(self, val):
        self.vals = val.copy()
        
    def set_new_ind(self, newInd):
        self.vals = newInd.vals.copy()
        self.fitness = newInd.fitness
        
    def get_val_idim(self, ith):
        return self.vals[ith]
    
    def get_val_vec(self):
        return self.vals
    
    def init_random_val(self):
        for i in range(self.n_dims):
            self.vals[i] = rd.random()
        
    def init_uniform_val(self, a, b):
        for i in range(self.n_dims):
            self.vals[i] = rd.uniform(a, b)
            
    def __str__(self):
        return "%s\n" % (self.vals)
    
    def getFitness(self):
        return self.fitness
    
    def computeFiness(self, matMemb, matCorrect):
        """
        matMemb is a matrix storing all membership values. Each row is the membership value
            corresponding to a given input sample of all base learners
            
        matCorrect is a matrix storing corresponding prediction correct or not
            1: correct, 0: wrong
        """
        self.fitness = 0
        for i in range(matMemb.shape[0]):
            tmp_vec = self.vals * matMemb[i]
            correct_vec = tmp_vec[matCorrect[i] == 1]
            wrong_vec = tmp_vec[matCorrect[i] == 0]
            if len(correct_vec) > 0:
                max_mem_correct = max(correct_vec)
                if len(wrong_vec) == 0:
                    self.fitness = self.fitness + 1
                else:               
                    max_mem_wrong = max(wrong_vec)
                    if max_mem_correct > max_mem_wrong:
                        self.fitness = self.fitness + 1
                
        return self.fitness
            
        
