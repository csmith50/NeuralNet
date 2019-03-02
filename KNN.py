# -*- coding: utf-8 -*-
import numpy as np
from statistics import mode

class KNN ():
    def __init__(self):
        self.data = None
        self.target = None
    def fit(self, data, target_Train):
        self.data = data
        self.target = target_Train
    def predict(self, data_Test, k):
        targets_predicted = []
        i = 0
        j = 0
        while i < len(data_Test):
            distances = []
            j = 0
            while j < len(self.data):
                distances.append(self.findDistance(self.data[j], data_Test[i]))
                j += 1
            lowest = self.findMins(distances, k) #findMins returns an array of indexes into distances
            lowest_Targets = []
            L = 0
            while L < len(lowest):
                lowest_Targets.append(self.target[lowest[L]])
                L += 1
            targets_predicted.append(mode(lowest_Targets)) #we'll default to what statistics.mode does for a tie
            i += 1
        return targets_predicted
    def findDistance(self, data_array, target_array): 
        diff = target_array - data_array
        sqdiff = diff ** 2
        dist = sum(sqdiff)
        return dist
    def findMins(self, distances, k):
        distances = np.argsort(distances)
        return distances[:k]