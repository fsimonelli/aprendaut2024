import pandas as pd
import numpy as np
import pprint as pprint

TESTING_DATASET_FILE = "./formatted_weather_table.csv"
PEDRO_DATASET_FILE = "./pedro.csv"
CONTINUOUS_DATASET_FILE = "./solvency_table.csv"
DATASET_FILE = "./lab1_dataset.csv"
# Continuous attributes: time, age, wtkg, karnof, preanti, cd40, cd420, cd80, cd820

pedro_dataset = pd.read_csv(PEDRO_DATASET_FILE, sep=",")
testing_dataset = pd.read_csv(TESTING_DATASET_FILE, sep=",")
continuous_dataset = pd.read_csv(CONTINUOUS_DATASET_FILE, sep=",")
        
def generate_every_pair_from_list(list):
    #A generalizar
    res = []
    for i in range(0, len(list)):
        for j in range(i+1, len(list)):
            res.append([list[i], list[j]])
    return res
            
#O(n^i)
def get_splits(dataset, attribute, target):
    min_entropy = 1
    dataset = dataset.sort_values(by=attribute)
    current_target = dataset[target].iloc[0]
    dataset_size = dataset.shape[0]
    candidate_splits = []
    
    for i in range(1,dataset_size):
        if current_target != dataset[target].iloc[i]:
            candidate_splits.append((dataset[attribute].iloc[i-1] + dataset[attribute].iloc[i])/2)
            current_target = dataset[target].iloc[i]
            
    #splits = generate_every_pair_from_list(candidate_splits)
    #Implementado para 2 rangos
    for split in candidate_splits:
        split_dataset = actually_split(dataset.copy(), attribute, split)
        aux_entropy = 0
        for value, count in split_dataset[attribute].value_counts().items():
            aux_entropy += count*entropy(split_dataset.loc[split_dataset[attribute] == value], target)
        aux_entropy = aux_entropy / split_dataset.shape[0]
            
        if (aux_entropy < min_entropy):
            min_entropy = aux_entropy
            best_values = split
            
    return (min_entropy,best_values)



def actually_split(dataset, attribute, splits):
    def discretize(x):
        if(x < splits):
            return '<' + str(splits)
        else:
            return '>' + str(splits)
    dataset[attribute] = dataset[attribute].apply(discretize)
    return dataset

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
    continuous = {}
    for attribute in attributes:
        # Continuous-Valued attribute
        if attribute in continuous_attributes:
            aux_entropy, best_split = get_splits(dataset, attribute, target)
            entropies.append(aux_entropy)
            continuous[attribute] = best_split
        else :
            res = 0
            for value, count in dataset[attribute].value_counts().items():
                res += count*entropy(dataset.loc[dataset[attribute] == value], target)
            entropies.append(res / dataset.shape[0])
            continuous[attribute] = None
            
    best_attribute = attributes[entropies.index(min(entropies))]
    
    if (continuous[best_attribute] is None):
        return best_attribute, dataset
    return best_attribute, actually_split(dataset.copy(), best_attribute, continuous[best_attribute])

def id3(dataset, target, attributes):
    if len(attributes) == 0 or len(dataset[target].value_counts().index) == 1:
        return dataset[target].value_counts().index[0]
    else :
        best, dataset = best_attribute(dataset, target, attributes)
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

continuous_attributes = ['EBIT_over_A','ln_A_over_L','RE_over_A','FCF_over_A']
continuous_target = 'Solvency'

pprint.pprint(id3(pedro_dataset, pedro_target, pedro_attributes))

pprint.pprint(id3(testing_dataset, testing_target, testing_attributes))

pprint.pprint(id3(continuous_dataset, continuous_target, continuous_attributes))

# Used Panda Functions

# read_csv
# value_counts => https://pandas.pydata.org/docs/reference/api/pandas.Series.value_counts.html
# items => https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.items.html
# loc => https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.loc.html
# iloc => https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.iloc.html
# drop => https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.drop.html
# shape => https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.shape.html
# index => https://pandas.pydata.org/docs/reference/api/pandas.Index.html
# apply => https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.apply.html