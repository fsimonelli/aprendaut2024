import pandas as pd
import numpy as np
import random

DATASET_FILE = "./lab1_dataset.csv"
TEST_FILE = "./test_dataset.csv"

dataset = pd.read_csv(DATASET_FILE, sep=",")
test_dataset = pd.read_csv(TEST_FILE, sep=",")

continuous_features = ['time', 'age', 'wtkg', 'preanti', 'cd40', 'cd420', 'cd80', 'cd820']
target = 'cid'
features = ['time', 'trt', 'age', 'wtkg', 'hemo', 'homo', 'drugs', 'karnof',
       'oprior', 'z30', 'zprior', 'preanti', 'race', 'gender', 'str2', 'strat',
       'symptom', 'treat', 'offtrt', 'cd40', 'cd420', 'cd80', 'cd820']

# testing dataset and new instance
test_target = 'Juega'
test_features = ['Tiempo','Temperatura','Humedad','Viento']

data = {
    "Tiempo": ["Nuboso"],
    "Temperatura": ["Frío"],
    "Humedad": ["Alta"],
    "Viento": ["Fuerte"]
}

df = pd.DataFrame(data)

def init():
    return dataset, features, continuous_features, target

def entropy(dataset, target):
    values = dataset[target].value_counts()
    total = dataset.shape[0]
    p0 = values.iloc[0]/total
    if (len(values) > 1):
        p1 = values.iloc[1]/total
        return -(p0)*np.log2(p0) - (p1) * np.log2(p1)
    else: 
        return -(p0)*np.log2(p0)

def split_dataset(dataset, feature, split):
    dataset_copy = dataset.copy()
    discretize = lambda x: ('(' + str(split[0]) + ',' + str(split[1]) + ']' if len(split) == 2 and split[0] < x <= split[1] else
                            '<=' + str(split[0]) if x <= split[0] else
                            '>' + str(split[-1])
                        )
    dataset_copy[feature] = dataset_copy[feature].apply(discretize)
    return dataset_copy

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
    
def split_into_train_test(dataset, train_size):
    dataset_size = dataset.shape[0]
    train_dataset = dataset.iloc[0:int(dataset_size*train_size)]
    testing_dataset = dataset.iloc[int(dataset_size*train_size):]
    return train_dataset, testing_dataset

def preprocesser(dataset, target, continuous_features):
    for cont_feature in continuous_features:
        entropy, splits = get_splits(dataset,cont_feature,target,2)
        dataset = split_dataset(dataset,cont_feature,splits)
    return dataset

def test_instances(X_train, y_train, X_test, y_test, m):
    res = 0
    train_ds = X_train.copy()
    train_ds[target] = y_train
    for i in range(0, X_test.shape[0]):
        instance = X_test.iloc[i]
        if naive_bayes(train_ds, target, features, instance, m) == y_test.iloc[i]:
            res += 1
    return (res / X_test.shape[0]) * 100
        
def naive_bayes(dataset, target, features, instance, m):
    
    dataset_size = dataset.shape[0]
    prob_1 = dataset[target].value_counts()[1]/dataset_size
    prob_0 = dataset[target].value_counts()[0]/dataset_size
    
    sum_1 = np.log(prob_1)
    sum_0 = np.log(prob_0)
    
    for feature in features:
        examples = dataset.loc[dataset[feature] == instance[feature]][target].value_counts()
        
        # if no instances with a specific target value is found, the get method will return 0
        count_1 = examples.get(1, default=0)
        count_0 = examples.get(0, default=0)
        
        feature_range = len(dataset[feature].value_counts())
        
        numerator_1 = count_1 + (m / feature_range)
        numerator_0 = count_0 + (m / feature_range)

        # sum of sequence
        sum_1 += np.log( numerator_1 / (dataset[target].value_counts()[1] + m) )
        sum_0 += np.log( numerator_0 / (dataset[target].value_counts()[0] + m) )
        
    # argmax
    if ( sum_1 > sum_0):
        return 1
    else:
        return 0