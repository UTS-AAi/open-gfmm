# -*- coding: utf-8 -*-
"""
Created on Sat Oct 12 16:02:52 2019

@author: Thanh Tung Khuat

This is the implementation of Differential Evolution algorithm
"""

from randombox.baseindividual import BaseIndividual

import numpy as np
import random as rd

min_range = -1
max_range = 2

class DifferentialEvolution(object):
    
    def __init__(self, dim, maxGen = 100, pSize = 100, crossRate=0.9, F_rate=0.5):
        self.pSize = pSize
        self.maxGen = maxGen
        self.n_dims = dim
        self.pop = np.empty(pSize, dtype=BaseIndividual)
        self.CR = crossRate
        self.F  = F_rate
        
    def printPop(self):
        for idx, ind in enumerate(self.pop):
            print("Individual %d" % idx)
            print(ind)
   
    def initPopulation(self):
        for i in range(self.pSize):
            self.pop[i] = BaseIndividual(self.n_dims)
            self.pop[i].init_random_val()
    
       
    def computeFiness(self, matMemb, matCorrect):
        for ind in self.pop:
            ind.computeFiness(matMemb, matCorrect)
            
    def run(self, matMemb, matCorrect):
        self.initPopulation()
        self.computeFiness(matMemb, matCorrect)
        
        for g in range(self.maxGen):
            new_pop = np.empty(self.pSize, dtype=BaseIndividual)
            for i in range(self.pSize):
                selected_idx = rd.sample(range(self.pSize), 4)
                
                for j in selected_idx:
                    if j == i:
                        j = selected_idx[-1]
                        break;
                
                j_rand = rd.randint(0, self.n_dims - 1)
                
                newInd = BaseIndividual(self.n_dims)
                for j in range(self.n_dims):
                    if (rd.random() < self.CR or j == j_rand):
                        val = self.pop[selected_idx[0]].get_val_idim(j) + self.F * (self.pop[selected_idx[1]].get_val_idim(j) - self.pop[selected_idx[2]].get_val_idim(j))
                        #if val >= min_range and val <= max_range:
                        newInd.set_val(j, val)
                        #else:
                            #newInd.set_val(j, rd.uniform(min_range, max_range))
                    else:
                        newInd.set_val(j, self.pop[i].get_val_idim(j))
                        
                newInd_fitness = newInd.computeFiness(matMemb, matCorrect)
                if newInd_fitness >= self.pop[i].getFitness():
                    new_pop[i] = newInd
                else:
                    new_pop[i] = self.pop[i]
            
            self.pop = new_pop
         
        # Get best individual
        best_fitness = -1
        best_ind = 0
        for i in range(self.pSize):
            if best_fitness < self.pop[i].getFitness():
                best_fitness = self.pop[i].getFitness()
                best_ind = i
                
        self.bestInd = BaseIndividual(self.n_dims)
        self.bestInd.set_new_ind(self.pop[best_ind])
        
    def getBestIndVal(self):
        return self.bestInd.get_val_vec()
                
        