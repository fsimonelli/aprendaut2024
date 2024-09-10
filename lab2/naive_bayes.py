import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import KBinsDiscretizer

DATASET_FILE = "./lab1_dataset.csv"
TEST_FILE = "./test_dataset.csv"

dataset = pd.read_csv(DATASET_FILE, sep=",")
test_dataset = pd.read_csv(TEST_FILE, sep=",")

continuous_features = ['time', 'age', 'wtkg', 'karnof', 'preanti', 'cd40', 'cd420', 'cd80', 'cd820']
target = 'cid'
features = ['time', 'trt', 'age', 'wtkg', 'hemo', 'homo', 'drugs', 'karnof',
       'oprior', 'z30', 'zprior', 'preanti', 'race', 'gender', 'str2', 'strat',
       'symptom', 'treat', 'offtrt', 'cd40', 'cd420', 'cd80', 'cd820']

test_target = 'Juega'
test_features = ['Tiempo','Temperatura','Humedad','Viento']
possible_test_features = {
    'Tiempo': 3,
    'Temperatura': 3,
    'Humedad': 2,
    'Viento': 2
}

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

def preprocesser(dataset, target, continuous_features):
    for cont_feature in continuous_features:
        entropy, splits = get_splits(dataset,cont_feature,target,2)
        dataset = split_dataset(dataset,cont_feature,splits)
    return dataset

def naive_bayes(dataset, target, features, feature_range, instance, m):
    prob_si = (dataset[target].value_counts()/dataset.shape[0])['Sí']
    prob_no = (dataset[target].value_counts()/dataset.shape[0])['No']
    
    product_si = product_no = 1
    for feature in features:
        examples = dataset.loc[dataset[feature] == instance[feature]][target].value_counts()
        
        count_si = examples.get('Sí', 0)
        count_no = examples.get('No', 0)
        
        numerador_si = count_si + m * 1/feature_range[feature]
        numerador_no = count_no + m * (1 / feature_range[feature])
            
        print(f'feature: {feature}, numerador_si: {numerador_si}, numerador_no: {numerador_no}')

        product_si = product_si * ( numerador_si / (dataset[target].value_counts()['Sí'] + m) )
        product_no = product_no * ( numerador_no / (dataset[target].value_counts()['No'] + m) )
    
    # print(product_si*prob_si, product_no*prob_no)
    if ( (product_si * prob_si) > (product_no * prob_no)):
        return 'Sí'
    else:
        return 'No'

data = {
    "Tiempo": ["Nuboso"],
    "Temperatura": ["Frío"],
    "Humedad": ["Alta"],
    "Viento": ["Fuerte"]
}

df = pd.DataFrame(data)

print(naive_bayes(test_dataset, test_target, test_features, possible_test_features, df.iloc[0], 2))


    