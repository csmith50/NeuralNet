# -*- coding: utf-8 -*-

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
import pandas as pd
import numpy as np
from HardCodedClassifier import HardCodedClassifier
from KNN import KNN
from ID3 import ID3
from NN import NN


def predict_custom_classifier(data_Train, data_Test, target_Train, target_Test, classifier, feature_Names = []):
    if type(classifier) is KNN:
        print("Would you like to specify a K value? A default of 3 will be used if not specified. (y or n)")
        user_input = input()
        if user_input == "y":
            print("New value?")
            user_input = input()
            k = int(user_input)
        else:
            k = 3
        classifier.fit(data_Train, target_Train)
        targets_predicted = classifier.predict(data_Test, k)
    elif type(classifier) is ID3:
        print("Building Tree...")
        classifier.tree = classifier.fit(data_Train, target_Train, feature_Names)
        print("Generating test results...")
        targets_predicted = classifier.predict(data_Test)
        print(targets_predicted)
    elif type(classifier) is NN:
        print("Building Net...")
        layer_info = []
        print("How many layers?")
        num = int(input())
        for i in range(num):
            print("How many nodes in layer " + str(i) + "?")
            u_input = int(input())
            layer_info.append(u_input)
        print("Building network...")
        classifier.create_Network(data_Train, target_Train, layer_info)

        print("Training Net...") 
        classifier.fit(data_Train, target_Train)
        
        print("Generating test results...")
        #TODO: put testing loop here i.e give predict one row at a time
        targets_predicted = []
        for i in range(len(data_Test)):
            targets_predicted.append(classifier.predict(data_Test[i], layer_info))
        print(targets_predicted)
          
    i = 0 
    score = 0
    while i < len(targets_predicted):
        if targets_predicted[i] == target_Test[i]:
            score += 1
        i += 1
    print("The custom classifier had an accuracy of: " + str((score / len(targets_predicted)) * 100) + "%")

def predict_default_Classifier(data_Train, data_Test, target_Train, target_Test): #TODO: Change this to built in NN?
    classifier = KNeighborsRegressor(n_neighbors=3)
    classifier.fit(data_Train, target_Train)
    targets_predicted = classifier.predict(data_Test)
    i = 0 
    score = 0
    while i < len(targets_predicted):
        if targets_predicted[i] == target_Test[i]:
            score += 1
        i += 1
    print("The built in KNN classifier with a k of 3 had an accuracy of: " + str((score / len(targets_predicted)) * 100) + "%")

def randomize_Data(dataset, targetset):
    return train_test_split(dataset, targetset, test_size = 0.3)

def preprocess (file_location): 
    names = ["buying", "maint", "doors", "persons", "lug_boot", "safety", "condition"]
    
    data = pd.read_csv(file_location, header = None, skipinitialspace=False, names=names,
                          na_values=[" "])
    #print(data)
    #print(data.columns)
    
    data.buying.value_counts()
    data.buying = data.buying.astype('category')
    data["buying_cat"] = data.buying.cat.codes
    
    data.maint.value_counts()
    data.maint = data.maint.astype('category')
    data["maint_cat"] = data.maint.cat.codes
    
    #we'll use find and replace for the next two columns
    cleanup = {"doors" : {"2" : 2, "3" : 3, "4" : 4, "5more" : 5},
               "persons" : {"2" : 2, "4" : 4, "more" : 5}}
    data.replace(cleanup, inplace = True)
    
    data.lug_boot.value_counts()
    data.lug_boot = data.lug_boot.astype('category')
    data["lug_boot_cat"] = data.lug_boot.cat.codes
    
    data.safety.value_counts()
    data.safety = data.safety.astype('category')
    data["safety_cat"] = data.safety.cat.codes
    
    data.condition.value_counts()
    data.condition = data.condition.astype('category')
    data["condition_cat"] = data.condition.cat.codes
    
    #print(data)
    #print(data.columns)
    
    dataset = data
    dataset = dataset.drop(columns=["condition_cat"])
    dataset = dataset.drop(columns = ["buying"])
    dataset = dataset.drop(columns = ["maint"])
    dataset = dataset.drop(columns = ["condition"])
    dataset = dataset.drop(columns = ["lug_boot"])
    dataset = dataset.drop(columns = ["safety"])
    dataset = dataset.as_matrix()
    target = data["condition_cat"].as_matrix()
    return dataset, target, names[:-1]

def main():
    print("Would you like to specify a file location? (y or n)")
    specifyFile = input()
    dataset = None
    user_Input = None
    if specifyFile == "y":
        print("Please type in file location")
        file_Location = input()
        dataset, target, feature_Names = preprocess(file_Location)
        print("Display Data? (y or n)")
        user_Input = input()
        if user_Input == "y":
            print(dataset)
            print(target)
    else:
        print ("Default data set will be used instead")
        data = datasets.load_iris()
        dataset = data.data
        target = data.target
        feature_Names = data.target_names
        print("Display data? (y or n)")
        user_Input = input()
        if user_Input == "y":
            print ("Printing the data \n ------------------")
            print(dataset)
            print(target)
    
    data_Train, data_Test, target_Train, target_Test = randomize_Data(dataset, target)
    print ("Data randomized. Display randomized data? (y or n)")
    user_Input = input()
    if user_Input == "y":
        print("Printing random data \n ------------------------")
        print(data_Train)
        print(target_Train)
        print(data_Test)
        print(target_Test)
    
    print("Using built in ID3 algorithim...\n")
    predict_default_Classifier(data_Train, data_Test, target_Train, target_Test)
    
    print("Use custom classifier? (y or n)")
    user_input = input()
    if user_input == "y":
        print("Please select custom classifier (K for KNN, H for HardCodedClassifier, I for ID3 decesion Tree, N for Nueral Net)")
        user_input = input()
        if user_input == "K" or user_input == "k":
            classifier = KNN()
        elif user_input == "I" or user_input == "i":
            classifier = ID3()
        elif user_input == "N" or user_input == "n":
            classifier = NN()
        else:
            classifier = HardCodedClassifier()
        predict_custom_classifier(data_Train, data_Test, target_Train, target_Test, classifier, feature_Names)
        
        
    
    


main()