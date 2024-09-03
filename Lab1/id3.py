import pandas as pd
import numpy as np
import pprint as pprint
import random

DATASET_FILE = "./lab1_dataset.csv"

dataset = pd.read_csv(DATASET_FILE, sep=",")

continuous_features = ['time', 'age', 'wtkg', 'karnof', 'preanti', 'cd40', 'cd420', 'cd80', 'cd820']
target = 'cid'
features = ['time', 'trt', 'age', 'wtkg', 'hemo', 'homo', 'drugs', 'karnof',
       'oprior', 'z30', 'zprior', 'preanti', 'race', 'gender', 'str2', 'strat',
       'symptom', 'treat', 'offtrt', 'cd40', 'cd420', 'cd80', 'cd820']

def init():
    return dataset, features, continuous_features, target

def generate_every_pair_from_list(list, max_range_splits):
    res = []
    for i in range(0, len(list)):
        if (max_range_splits == 3):
            for j in range(i+1, len(list)):
                res.append([list[i], list[j]])
        res.append([list[i]])
    return res
            
#O(n^i)
def get_splits(dataset, feature, target, max_range_splits):
    min_entropy = 2
    dataset = dataset.sort_values(by=feature)
    current_target = dataset[target].iloc[0]
    dataset_size = dataset.shape[0]
    candidate_splits = []
    best_values = []
    
    # Finding splits
    for i in range(1, dataset_size):
        if current_target != dataset[target].iloc[i]:
            candidate_splits.append((dataset[feature].iloc[i-1] + dataset[feature].iloc[i])/2)
            current_target = dataset[target].iloc[i]
    sample = candidate_splits
    if len(candidate_splits) > 50:
        sample = random.sample(candidate_splits, 50)
    splits = generate_every_pair_from_list(sample, max_range_splits)
 
    for split in splits:
        splitted_dataset = split_dataset(dataset.copy(), feature, split)
        aux_entropy = 0
        for value, count in splitted_dataset[feature].value_counts().items():
            aux_entropy += count*entropy(splitted_dataset.loc[splitted_dataset[feature] == value], target)
        aux_entropy = aux_entropy / splitted_dataset.shape[0]
            
        if (aux_entropy < min_entropy):
            min_entropy = aux_entropy
            best_values = split
            
    return (min_entropy,best_values)

def split_dataset(dataset, feature, split):
    discretize = lambda x: ('(' + str(split[0]) + ',' + str(split[1]) + ']' if len(split) == 2 and split[0] < x <= split[1] else
                            '<=' + str(split[0]) if x <= split[0] else
                            '>' + str(split[-1])
                        )
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

def best_feature(dataset, target, features, continuous_features, max_range_splits):
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
    best_feature = features[entropies.index(min(entropies))]
    
    if not (best_feature in continuous):
        return best_feature, dataset
    return best_feature, split_dataset(dataset.copy(), best_feature, continuous[best_feature])

def id3(dataset, target, features, continuous_features, max_range_splits, intact_dataset):
    if len(features) == 0 or len(dataset[target].value_counts().index) == 1:
        # value_counts[0] is either the only or the most common target value left in the current dataset.
        return dataset[target].value_counts().index[0] 
    aux = dataset.copy()
    best, dataset = best_feature(dataset, target, features, continuous_features, max_range_splits)
    decision_tree = {best: {}}
    
    new_features = features.copy()
    new_features.remove(best)

    if best in continuous_features:
        aux_dataset = dataset.copy()
    else :
        aux_dataset = intact_dataset.copy()
    for value in aux_dataset[best].value_counts().index:
        if not (best in continuous_features):
            examples = dataset.copy().loc[dataset[best] == value]
            if (len(examples) == 0):
                decision_tree[best][value] = dataset[target].value_counts().index[0]
            else:
                decision_tree[best][value] = id3(examples, target, new_features, continuous_features, max_range_splits, intact_dataset)
        else:
            arr = []
            for i in range(0, aux.shape[0]):
                arr.append(isEqual(aux.iloc[i][best], value))
            examples = aux.copy().iloc[arr]
            if (len(examples) == 0):
                decision_tree[best][value] = dataset.value_counts().index[0]
            else:
                decision_tree[best][value] = id3(examples, target, new_features, continuous_features, max_range_splits, intact_dataset)
    return  decision_tree

def classify_instance(tree, instance):
    if isinstance(tree, dict):
        feature, branches = next(iter(tree.items()))
        feature_value = instance[feature]
        if isinstance(branches, dict):
            for condition, subtree in branches.items():
                if (isEqual(feature_value, condition)):
                    return classify_instance(subtree, instance)
        else:
            return branches
    else:
        return tree

def isEqual(instance_value, dataset_value):
    if isinstance(dataset_value, str):
        if '(' in dataset_value:
            lower_bound, upper_bound = dataset_value[1:-1].split(',')
            return float(lower_bound) <= instance_value < float(upper_bound)
        elif '<=' in dataset_value:
            return instance_value <= float(dataset_value[2:])
        elif '>' in dataset_value:
            return instance_value > float(dataset_value[1:])
    return instance_value == dataset_value

# train_size is a float between 0 and 1
def split_into_train_test(dataset, train_size):
    dataset_size = dataset.shape[0]
    train_dataset = dataset.iloc[0:int(dataset_size*train_size)]
    testing_dataset = dataset.iloc[int(dataset_size*train_size):]
    return train_dataset, testing_dataset
    

def test_instances(tree, dataset):
    res = 0
    for i in range(0,dataset.shape[0]):
        if classify_instance(tree, dataset.iloc[i]) == dataset.iloc[i][target]:
            res = res + 1 
    return (res/dataset.shape[0])*100


#random.seed(59)
#train_ds, test_ds = split_into_train_test(dataset, 0.8)
#tree = id3(train_ds, target, features, 2, train_ds)
#test_instances(tree, test_ds)




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