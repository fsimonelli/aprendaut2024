import pandas as pd
import numpy as np
import pprint as pprint

TESTING_DATASET_FILE = "./formatted_weather_table.csv"
PEDRO_DATASET_FILE = "./pedro.csv"
DATASET_FILE = "./lab1_dataset.csv"

pedro_dataset = pd.read_csv(PEDRO_DATASET_FILE, sep=",")
testing_dataset = pd.read_csv(TESTING_DATASET_FILE, sep=",")

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
    total = dataset.shape[0]
    p0 = values.iloc[0]/total
    if (len(values) > 1):
        p1 = values.iloc[1]/total
        return -(p0)*np.log2(p0) - (p1) * np.log2(p1)
    else: 
        return -(p0)*np.log2(p0)

def best_attribute(dataset, target, attributes):
    entropies = []
    for attribute in attributes:
        res = 0
        for value, count in dataset[attribute].value_counts().items():
            res += count*entropy(dataset.loc[dataset[attribute] == value], target)
        entropies.append(res / dataset.shape[0])
    return attributes[entropies.index(min(entropies))]

def id3(dataset, target, attributes):
    if len(attributes) == 0 or len(dataset[target].value_counts().index) == 1:
        return dataset[target].iloc[0]
    else :
        best = best_attribute(dataset, target, attributes)
        tree = {best: {}}
        new_attributes = attributes.copy()
        new_attributes.remove(best)
        
        for value in dataset[best].value_counts().index:
            tree[best][value] = id3(dataset.loc[dataset[best] == value], target, new_attributes)
        return tree


pedro_attributes = ["Dedicación", "Dificultad", "Horario", "Humedad", "Humor Doc"]
pedro_target = "Salva"

testing_attributes = ["Outlook", "Temp.", "Humidity", "Wind"]
testing_target = "Decision"

pprint.pprint(id3(pedro_dataset, pedro_target, pedro_attributes))
pprint.pprint(id3(testing_dataset, testing_target, testing_attributes))

# Used Panda Functions

# read_csv
# value_counts => https://pandas.pydata.org/docs/reference/api/pandas.Series.value_counts.html
# items => https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.items.html
# loc => https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.loc.html
# iloc => https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.iloc.html
# drop => https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.drop.html
# shape => https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.shape.html
# index => https://pandas.pydata.org/docs/reference/api/pandas.Index.html