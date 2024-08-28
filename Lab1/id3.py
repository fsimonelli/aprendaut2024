import pandas as pd
import numpy as np

DATASET_FILE = "./pedro.csv"

dataset = pd.read_csv(DATASET_FILE, sep=",")

# implementar el algoritmo ID3 con la extensión para atributos numéricos

class Node(object):
    def __init__(self, data):
        self.data = data
        self.children = []

    def add_child(self, obj):
        self.children.append(obj)
        
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
        #print(dataset.shape[0])
        entropies.append(res / dataset.shape[0])
    return entropies.index(min(entropies))

def id3(dataset, target, attributes):
    print("iter")
    if len(attributes) == 0:
        return dataset[target].value_counts().iloc[0]
    if len(dataset[target].value_counts()) == 1:
        return dataset[target].value_counts().iloc[0]
    else :
        best = best_attribute(dataset, target, attributes)
        tree = Node(best)
        for value, count in dataset[atributos[best]].value_counts().items():
            new_attributes = attributes.copy()
            new_attributes.remove(atributos[best])
            print(attributes)
            print(new_attributes)
            #Estamos pasando mal el dataset
            tree.add_child(id3(dataset[atributos[best]] == value, target, new_attributes))
        return tree
    
atributos = ['Dedicación','Dificultad','Horario','Humedad','Humor Doc']  
#print('El mejor atributo es: ' + atributos[best_attribute(dataset, "Salva", atributos)])

id3(dataset, "Salva", atributos)