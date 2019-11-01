# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 11:45:31 2019

@author: Thanh Tung Khuat
"""
import statistics as st

def find_max_mode(inputList):
    list_table = st._counts(inputList)
    len_table = len(list_table)

    if len_table == 1:
        max_mode = st.mode(inputList)
    else:
        new_list = []
        for i in range(len_table):
            new_list.append(list_table[i][0])
        max_mode = max(new_list) # use the max value here
    
    return max_mode