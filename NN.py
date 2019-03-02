# -*- coding: utf-8 -*-
import numpy as np
from math import e

class Nueron ():
    def __init__(self, num_Inputs = 0, weights = [], output = False):
        self.num_inputs = num_Inputs
        self.weights = weights
        self.output = output
        self.activation = 0
        self.error = 0
    def compute(self, data, bias): #needs asserts that garuntee size of weights = num_of_inputs
        outputs = []
        bias_set = False
        for i in range(self.num_inputs - 1):
            if bias[0] and not bias_set:
                outputs.append(bias[1] * self.weights[i]) #i should start at 0 but keep an eye on this
                bias_set = True
                continue
            outputs.append(data[i] * self.weights[i])
        self.activation = 1 /(1 + e**-sum(outputs))
    def clear(self):
        self.activation = 0
        self.error = 0     
    def computeE(self, previous_layer = [], current_node_i = 0, target = 0):
        if self.output:
            self.error = self.activation*(1 - self.activation)*(self.activation - target)
        else:
            outputs = []
            for i in range(len(previous_layer)):
                outputs.append(previous_layer[i].weights[current_node_i] * previous_layer[i].error)
            self.error = self.activation * (1-self.activation) * sum(outputs)
    def updateWeights(self, learning_rate, bias, i_layer = []):
        #the number of weights should be equal to the number of nodes in the i layer
        #print(len(self.weights))
        for i in range(len(self.weights) + 1):
            if i == 0 and bias[0]:
                self.weights[i] = self.weights[i] - (learning_rate * self.error * bias[1])
            else:
                if i > len(i_layer):
                    continue
                i_activation = i_layer[i - 1].activation #python doesn't like doing this within calculations for some reason
                self.weights[i - 1] = self.weights[i - 1] - (learning_rate * self.error * i_activation)
class NN ():
    def __init__(self):
        self.nuerons = []
        self.bias = None
        self.layer_p = []
        self.learning_rate = 0
        
    def create_Network(self, data, target, layer_p):
        user_input = None
        bias = (False, 0)
        print("Would you like a bias node? (y or n)")
        user_input = input()
        if user_input == "y":
            print("What value would you like?")
            user_input = input()
            bias = (True, int(user_input))
        self.bias = bias
        self.layer_p = layer_p
        
        #the length of layer_p indicates the number of layers we have in our network
        #the values in each place are the number of nodes in each layer
        for i in range(len(layer_p)):
            nueron_layer = []
            for j in range(layer_p[i]):
                if i == 0:
                    if bias[0]:
                        nueron_layer.append(Nueron(len(data[0]) + 1, self.gen_random(len(data[0]) + 1)))
                    else:
                        nueron_layer.append(Nueron(len(data[0], self.gen_random(len(data[0])))))
                else:
                    if bias[0]:
                        num_inputs = layer_p[i - 1] + 1
                        weights = self.gen_random(num_inputs)
                        n = Nueron(num_inputs, weights)
                        nueron_layer.append(n)
                    else:
                        nueron_layer.append(Nueron(layer_p[i - 1], self.gen_random(layer_p[i - 1])))
            self.nuerons.append(nueron_layer)
        
        #next get our output nodes
        outputNodes = []
        num_in = layer_p[len(layer_p) - 1]
        for i in range(len(set(target))):
            outputNodes.append(Nueron(num_in, self.gen_random(num_in), True))
        self.nuerons.append(outputNodes)
        #print(self.nuerons)
        
    def fit(self, data_Train, target_Train):
        print("What learning rate would you like?")
        self.learning_rate = float(input()) #don't do this in professional code lol
        for i in range(len(data_Train)):
            predicted_value = self.predict(data_Train[i], self.layer_p)
            j = len(self.nuerons) - 1
            #calculate error in each node
            while j >= 0: #loop through layers
                num_n = len(self.nuerons[j])
                for k in range(num_n): #loop through nodes in layers
                    layer = self.nuerons[j]
                    if j == len(self.nuerons) - 1: #if output nodes
                        layer[k].computeE([], k, target_Train[i])
                    else: 
                        layer[k].computeE(self.nuerons[j + 1], k)
                j -= 1
                
            #next we update weights
            j = len(self.nuerons) - 1
            while j >= 0: #loop through layers; calculate error in each node
                num_n = len(self.nuerons[j])
                for k in range(num_n): #loop through nodes in layers
                    layer = self.nuerons[j]
                    if j == 0: #if last layer
                        layer[k].updateWeights(self.learning_rate, self.bias)
                    else: 
                        layer[k].updateWeights(self.learning_rate, self.bias, self.nuerons[j - 1])
                j -= 1
        
    def predict(self, data_Row, layer_p):
        for i in range(len(layer_p) + 1): #iterate between layers
            for j in range(len(self.nuerons[i])): #iterate between nodes in layers
                node = self.nuerons[i][j]
                if i == 0:
                    node.compute(data_Row, self.bias)
                else:
                    new_Row = []
                    for k in range(layer_p[i - 1]): #loop through and grab previous layer's outputs
                        nodeN = self.nuerons[i - 1][k]
                        new_Row.append(nodeN.activation)
                    node.compute(new_Row, self.bias)
        #at this piont we need to look at outermost layer and make a decesion
        #this code is calibrated to the iris data set
        greatest_value = 0
        n_index = len(self.nuerons) - 1
        for i in range(len(self.nuerons[n_index])):
            n = self.nuerons[n_index][i]
            nPrevious = n
            if i != 0:
                nPrevious = self.nuerons[n_index][i - 1]
                
            
            print(n.activation)
            #print(nPrevious.activation)
            if n.activation > nPrevious.activation: 
                greatest_value = i
        return greatest_value
    
    def gen_random(self, num_to_generate): 
        randomNums = 2 * np.random.random_sample(num_to_generate) - 1
        #print(randomNums)
        return randomNums.tolist()
        
    
