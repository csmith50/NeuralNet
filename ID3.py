# -*- coding: utf-8 -*-
import numpy as np

class Tree(object):
    def __init__(self, data = 'root', children = None):
        self.data = data
        self.children = []
        if children is not None:
            for child in children:
                self.add_child(child)
    def __repr__(self):
        return self.data
    def add_child(self, node):
        self.children.append(node)

class ID3 ():
    def __init__(self):
        self.data = None
        self.target = None
        self.tree = None #this will be set to the root node
    def fit(self, data, target_Train, feature_Names):
        nData = len(data)
        nFeature = len(feature_Names)
        if nData == 0 or nFeature == 0: #we are at an empty branch; return most common element
            if not target_Train:
                return None
            else:
                return target_Train[np.argmax(target_Train)]
        elif (target_Train == target_Train[0]).sum() == len(data): #only 1 class remains; return a leaf
            return target_Train[0]
        else: 
            #choose the best feature to split on
            gain = np.zeros(nFeature)
            for i in range(nFeature):
                gain[i] = self.calc_info_gain(data, target_Train, i)
            bestFeature = np.argmin(gain)
            tree = {feature_Names[bestFeature]:{}}
            #find possible feataure values
            values = []
            for i in range(len(data)):
                if data[i][bestFeature] not in values:
                    values.append(data[i][bestFeature])
            for value in values:
                index = 0
                for datapoint in data:
                    newData = []
                    newClasses = []
                    if datapoint[bestFeature] == value:
                        if bestFeature == 0:
                            datapoint = datapoint[1:]
                            newNames = feature_Names[1:]
                        elif bestFeature == nFeature:
                            datapoint = datapoint[:-1]
                            newNames = feature_Names[:-1]
                        else:
                            datapoint = datapoint[:bestFeature]
                            datapoint = np.append(datapoint, datapoint[bestFeature+1:])
                            newNames = feature_Names[:bestFeature]
                            newNames = np.append(newNames, feature_Names[bestFeature+1:])
                        newData.append(datapoint)
                        newClasses.append(target_Train[index])
                    index += 1
                #recurse to the next level
                subtree = self.fit(newData, newClasses, newNames)
                tree[feature_Names[bestFeature]][value] = subtree
            return tree
        
    def predict(self, data_Test):
        targets_predicted = []
        i = 0
        while i < len(data_Test):
            targets_predicted.append(self.findPath(self.tree, data_Test[i][0]))
            i += 1
        return targets_predicted
            
    def calc_entropy(self, prob):
        if prob != 0:
            return -prob * np.log2(prob)
        else:
            return 0
        
    def calc_info_gain(self, data, target_Train, feature):
        gain = 0
        ndata = len(data)
        #list the values that feature can take
        values = []
        for datapoint in data:
            if datapoint[feature] not in values:
                values.append(datapoint[feature])
        
        featureCounts = np.zeros(len(values))
        entropy = np.zeros(len(values))
        valueIndex = 0
        for value in values:
            dataIndex = 0
            newClasses = []
            for datapoint in data:
                if datapoint[feature] == value:
                    featureCounts[valueIndex] += 1
                    newClasses.append(target_Train[dataIndex])
                dataIndex += 1
            
            #get the values in newClasses
            classValues = []
            for aclass in newClasses:
                if classValues.count(aclass) == 0:
                    classValues.append(aclass)
            classCounts = np.zeros(len(classValues))
            classIndex = 0
            for classValue in classValues:
                for aclass in newClasses:
                    if aclass == classValue:
                        classCounts[classIndex] += 1
                classIndex += 1
                
            for classIndex in range(len(classValues)):
                entropy[valueIndex] += self.calc_entropy(float(classCounts[classIndex]) / sum(classCounts))
            gain += float(featureCounts[valueIndex]) / ndata * entropy[valueIndex]
            valueIndex += 1
        return gain
    
    def findPath(self, tree, start, path = []):
        path = path + [start]
        if start not in tree:
            return None
        for node in tree[start]:
            if not node in path:
                path = self.findPath(self, self.tree, node, path)
        return path
                
        