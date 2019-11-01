# -*- coding: utf-8 -*-
"""
Created on Sun Sep  1 11:39:57 2019

@author: Thanh Tung Khuat

Implentation of base class for BaseLearner
"""
import numpy as np

from functionhelper.membershipcalc import memberG_with_selected_features, memberG

class BaseLearner(object):
    def __init__(self, sub_selected_feature = np.array([])):
        self.sub_selected_feature = sub_selected_feature
        
    def set_value_sub_selected_feature(self, sub_selected_feature):
        self.sub_selected_feature = sub_selected_feature
        
    def important_feature_score(self, V, W, classId, used_features, Xl, Xu, patId=None, gamma=1):
        """
            Compute important score of each feature in input patterns
            patId = None: no using classes for computation => Scores are computed using all hyperboxes
                         + otherwise: score are computed using hyperboxes with the same class
                         
            Return:
                a numpy array containing important scores of features. Size equals to the number of features
        """
        Xl = Xl[:, used_features]
        Xu = Xu[:, used_features]
        ovr_score = np.zeros((Xl.shape[0], Xl.shape[1]))
        for i in range(Xl.shape[0]):           
            if patId is None:
                feature_score, feature_id = memberG_with_selected_features(Xl[i], Xu[i], V, W, gamma)
            else:
                same_class_id = classId == patId[i]
                V_same = V[same_class_id]
                W_same = W[same_class_id]
                feature_score, feature_id = memberG_with_selected_features(Xl[i], Xu[i], V_same, W_same, gamma)
            
            # compute score of each feature based on feature_score, feature_id
            sum_score = 0
            #ones_vct = np.ones(len(feature_score))
            used_features = np.unique(feature_id)
            for f in used_features:
                #sv = ones_vct[feature_id == f]
                sv = feature_score[feature_id == f]
                sc = sv.sum() / len(feature_score)
                ovr_score[i, f] = sc
                sum_score = sum_score + sc
            # Normalize score of all features on sample i by dividing by sum_score
            if sum_score > 0:
                ovr_score[i] = ovr_score[i] / sum_score                
        
        # compute average scores over all input samples
        return ovr_score.sum(axis = 0) / Xl.shape[0]
        