import pandas as pd
import numpy as np
import pprint as pprint
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn import tree
import matplotlib.pyplot as plt

TESTING_DATASET_FILE = "./formatted_weather_table.csv"
PEDRO_DATASET_FILE = "./pedro.csv"
CONTINUOUS_DATASET_FILE = "./solvency_table.csv"
DATASET_FILE = "./lab1_dataset.csv"
# Continuous features: time, age, wtkg, karnof, preanti, cd40, cd420, cd80, cd820

pedro_dataset = pd.read_csv(PEDRO_DATASET_FILE, sep=",")
testing_dataset = pd.read_csv(TESTING_DATASET_FILE, sep=",")
continuous_dataset = pd.read_csv(CONTINUOUS_DATASET_FILE, sep=",")
dataset = pd.read_csv(DATASET_FILE, sep=",")
        
def generate_every_pair_from_list(list, max_range_splits):
    #A generalizar
    res = []
    for i in range(0, len(list)):
        if (max_range_splits == 3):
            for j in range(i+1, len(list)):
                res.append([list[i], list[j]])
        res.append([list[i]])
    return res
            
#O(n^i)
def get_splits(dataset, feature, target, max_range_splits):
    min_entropy = 1
    dataset = dataset.sort_values(by=feature)
    print(dataset[target])
    current_target = dataset[target].iloc[0]
    dataset_size = dataset.shape[0]
    candidate_splits = []
    best_values = []
    
    # Finding splits
    for i in range(1,dataset_size):
        if current_target != dataset[target].iloc[i]:
            candidate_splits.append((dataset[feature].iloc[i-1] + dataset[feature].iloc[i])/2)
            current_target = dataset[target].iloc[i]
            
    splits = generate_every_pair_from_list(candidate_splits, max_range_splits)
    
    for split in splits:
        split_dataset = actually_split(dataset.copy(), feature, split)
        aux_entropy = 0
        for value, count in split_dataset[feature].value_counts().items():
            aux_entropy += count*entropy(split_dataset.loc[split_dataset[feature] == value], target)
        aux_entropy = aux_entropy / split_dataset.shape[0]
            
        if (aux_entropy < min_entropy):
            min_entropy = aux_entropy
            best_values = split
            
    return (min_entropy,best_values)

def actually_split(dataset, feature, split):
    print(split)
    def discretize(x):
        if (len(split) == 2):
            if (x < split[1] and x >= split[0]):
                return '[' + str(split[0]) + ',' + str(split[1]) + ']' 
        if(x < split[0]):
            return '<' + str(split[0])
        return '>' + str(split[-1])
    dataset[feature] = dataset[feature].apply(discretize)
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

def best_feature(dataset, target, features, max_range_splits):
    entropies = []
    continuous = {}
    for feature in features:
        # Continuous-Valued feature 
        if feature in continuous_features:
            aux_entropy, best_split = get_splits(dataset, feature, target, max_range_splits)
            entropies.append(aux_entropy)
            continuous[feature] = best_split
        else :
            res = 0
            for value, count in dataset[feature].value_counts().items():
                res += count*entropy(dataset.loc[dataset[feature] == value], target)
            entropies.append(res / dataset.shape[0])
            continuous[feature] = None
            
    best_feature = features[entropies.index(min(entropies))]
    
    if (continuous[best_feature] is None):
        return best_feature, dataset
    return best_feature, actually_split(dataset.copy(), best_feature, continuous[best_feature])

def id3(dataset, target, features, max_range_splits):
    if len(features) == 0 or len(dataset[target].value_counts().index) == 1:
        # value_counts[0] is either the only or the most common target value left in the current dataset.
        return dataset[target].value_counts().index[0] 
    
    best, dataset = best_feature(dataset, target, features, max_range_splits)
    decision_tree = {best: {}}
        
    new_features = features.copy()
    new_features.remove(best)
        
    for value in dataset[best].value_counts().index:
        examples = dataset.loc[dataset[best] == value]
        if (len(examples) == 0):
            decision_tree[best][value] = dataset.value_counts().index[0]
        else:
            decision_tree[best][value] = id3(examples, target, new_features, max_range_splits)
    return  decision_tree


pedro_features = ["Dedicación", "Dificultad", "Horario", "Humedad", "Humor Doc"]
pedro_target = "Salva"

testing_features = ["Outlook", "Temp.", "Humidity", "Wind"]
testing_target = "Decision"

solvency_continuous_features = ['EBIT_over_A','ln_A_over_L','RE_over_A','FCF_over_A']
solvency_continuous_target = 'Solvency'

continuous_features = ['time', 'age', 'wtkg', 'karnof', 'preanti', 'cd40', 'cd420', 'cd80', 'cd820']
target = 'cid'
features = ['time', 'trt', 'age', 'wtkg', 'hemo', 'homo', 'drugs', 'karnof',
       'oprior', 'z30', 'zprior', 'preanti', 'race', 'gender', 'str2', 'strat',
       'symptom', 'treat', 'offtrt', 'cd40', 'cd420', 'cd80', 'cd820']

print(features)

pprint.pprint(id3(dataset, target, features, 2))

# pprint.pprint(id3(pedro_dataset, pedro_target, pedro_features, 2))

# pprint.pprint(id3(testing_dataset, testing_target, testing_features, 2))

# pprint.pprint(id3(continuous_dataset, continuous_target, continuous_features, 3))

# X = continuous_dataset.drop('Solvency', axis=1) 
# y = continuous_dataset['Solvency']

# plt.figure()
# clf = tree.DecisionTreeClassifier(criterion='entropy')
# clf = clf.fit(X, y)

# tree.plot_tree(clf, class_names=['0', '1'])
# plt.show()


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