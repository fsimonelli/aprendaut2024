import pandas as pd
import numpy as np

DATASET_FILE = "./formatted_weather_table.csv"

dataset = pd.read_csv(DATASET_FILE, sep=",")

# implementar el algoritmo ID3 con la extensión para atributos numéricos

class Node(object):
    def __init__(self, data):
        self.data = data
        self.children = []

    def add_child(self, obj):
        self.children.append(obj)
    
    def isLeaf(self):
        return len(self.children) == 0
        
# Entropy for boolean functions.
def entropy(dataset, target):
    values = dataset[target].value_counts()
    total = values.sum()
    if (len(values) > 1):
        return -(values.iloc[0]/total)*np.log2(values.iloc[0]/total) - (values.iloc[1]/total) * np.log2(values.iloc[1]/total)
    else: 
        return -(values.iloc[0]/total)*np.log2(values.iloc[0]/total)

def best_attribute(dataset, target, attributes):
    entropies = []
    for attribute in attributes:
        res = 0
        for value, count in dataset[attribute].value_counts().items():
            res += count*entropy(dataset.loc[dataset[attribute] == value], target)
        entropies.append(res / dataset.shape[0])
    return entropies.index(min(entropies))

def id3(dataset, target, attributes):
    print(dataset)
    if len(attributes) == 0:
        return Node(dataset[target].value_counts().index.tolist()[0])
    #print(dataset[target].value_counts().index.tolist())
    if len(dataset[target].value_counts().index.tolist()) == 1:
        return Node(dataset[target].value_counts().index.tolist()[0])
    else :
        best = best_attribute(dataset, target, attributes)
        print(best_attribute(dataset, target, attributes))
        tree = Node(attributes[best])
        for value in dataset[attributes[best]].value_counts().index.tolist():
            #print(dataset[atributos[best]].value_counts())
            new_attributes = attributes.copy()
            new_attributes.remove(attributes[best])
            #print(attributes)
            #print(new_attributes)
            #print(dataset.iloc[(dataset[atributos[best]] == value).values])
            tree.add_child(id3((dataset.iloc[(dataset[attributes[best]] == value).values]).drop(columns=attributes[best]), target, new_attributes))
        return tree
    
def printTree(node, i):
    for j in range(0,i):
        print("-", end = " ")
    if (not node.isLeaf()):
        print(node.data)
        for child in node.children:
            printTree(child, i+1)
    else:
        print(node.data)

#atributos = ['Dedicación','Dificultad','Horario','Humedad','Humor Doc']  
#print('El mejor atributo es: ' + atributos[best_attribute(dataset, "Salva", atributos)])
atributos = ['Outlook','Temp.','Humidity','Wind']
#print(dataset)
#resTree = id3(dataset, "Salva", atributos)
resTree = id3(dataset, "Decision", atributos)
printTree(resTree, 0)