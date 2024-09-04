import pandas as pd
import numpy as np
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

def generate_combinations(list, max_range_splits):
    res = []
    for i in range(0, len(list)):
        if (max_range_splits == 3):
            for j in range(i+1, len(list)):
                res.append([list[i], list[j]])
        res.append([list[i]])
    return res
            
def get_splits(dataset, feature, target, max_range_splits):
    min_conditional_entropy = 2
    dataset = dataset.sort_values(by=feature)
    current_target = dataset[target].iloc[0]
    dataset_size = dataset.shape[0]
    candidate_splits = []
    best_splits = []
    
    # Finding splits
    for i in range(1, dataset_size):
        if current_target != dataset[target].iloc[i]:
            candidate_splits.append((dataset[feature].iloc[i-1] + dataset[feature].iloc[i])/2)
            current_target = dataset[target].iloc[i]
    
    sample = candidate_splits
    if len(candidate_splits) > 50:
        sample = random.sample(candidate_splits, 50)
    
    splits = generate_combinations(sample, max_range_splits)
 
    for split in splits:
        splitted_dataset = split_dataset(dataset, feature, split)
        aux_conditional_entropy = 0
        for value, count in splitted_dataset[feature].value_counts().items():
            aux_conditional_entropy += count*entropy(splitted_dataset.loc[splitted_dataset[feature] == value], target)
        aux_conditional_entropy = aux_conditional_entropy / splitted_dataset.shape[0]
            
        if (aux_conditional_entropy < min_conditional_entropy):
            min_conditional_entropy = aux_conditional_entropy
            best_splits = split
            
    return (min_conditional_entropy, best_splits)

def split_dataset(dataset, feature, split):
    dataset_copy = dataset.copy()
    discretize = lambda x: ('(' + str(split[0]) + ',' + str(split[1]) + ']' if len(split) == 2 and split[0] < x <= split[1] else
                            '<=' + str(split[0]) if x <= split[0] else
                            '>' + str(split[-1])
                        )
    dataset_copy[feature] = dataset_copy[feature].apply(discretize)
    return dataset_copy
    
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
    conditional_entropies = []
    continuous = {}
    for feature in features:
        # Continuous-Valued feature 
        if feature in continuous_features:
            aux_entropy, best_split = get_splits(dataset, feature, target, max_range_splits)
            conditional_entropies.append(aux_entropy)
            continuous[feature] = best_split
        else :
            res = 0
            for value, count in dataset[feature].value_counts().items():
                res += count*entropy(dataset.loc[dataset[feature] == value], target)
            conditional_entropies.append(res / dataset.shape[0])
    best_feature = features[conditional_entropies.index(min(conditional_entropies))]
    
    if not (best_feature in continuous):
        return best_feature, None
    return best_feature, continuous[best_feature]

def id3(dataset, target, features, continuous_features, max_range_splits, intact_dataset):
    if len(features) == 0 or len(dataset[target].value_counts().index) == 1:
        # value_counts[0] is either the only or the most common target value left in the current dataset.
        return dataset[target].value_counts().index[0] 
 
    best, best_splits = best_feature(dataset, target, features, continuous_features, max_range_splits)
    decision_tree = {best: {}}
    
    new_features = features.copy()
    new_features.remove(best)
    
    original_dataset = intact_dataset
    
    if best_splits:
        original_dataset = split_dataset(intact_dataset, best, best_splits)
        dataset = split_dataset(dataset, best, best_splits)
        
    for value in original_dataset[best].value_counts().index:
        examples = dataset.loc[dataset[best] == value]
        if (len(examples) == 0):
            decision_tree[best][value] = original_dataset.loc[original_dataset[best] == value][target].value_counts().index[0]
        else:
            decision_tree[best][value] = id3(examples, target, new_features, continuous_features, max_range_splits, intact_dataset)
    
    return decision_tree

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


train_dataset = dataset.iloc[0:1711]
test_dataset = dataset.iloc[1711:]
current_dataset = test_dataset.copy()

tree = id3(train_dataset, target, features, continuous_features, 2, dataset)
#pprint.pprint(tree)

res = 0
for i in range(0,current_dataset.shape[0]):
    #print(classify_instance(tree, continuous_dataset.iloc[i]), "vs", continuous_dataset.iloc[i][solvency_continuous_target])
    #print(i)
    if classify_instance(tree, current_dataset.iloc[i]) == current_dataset.iloc[i][target]:
        res = res + 1 
    # pprint.pprint(id3(weather_dataset, weather_target, weather_features, 2))
print((res/current_dataset.shape[0])*100)