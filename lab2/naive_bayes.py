import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import accuracy_score, make_scorer
from sklearn.model_selection import cross_val_score
from sklearn.base import BaseEstimator, ClassifierMixin

DATASET_FILE = "./lab1_dataset.csv"

dataset = pd.read_csv(DATASET_FILE, sep=",")

target = 'cid'

continuous_features = [
    'time',
    'age', 
    'wtkg', 
    'preanti', 
    'cd40', 
    'cd420', 
    'cd80', 
    'cd820'
]

features = [
    'time', 
    'trt', 
    'age', 
    'wtkg', 
    'hemo', 
    'homo', 
    'drugs', 
    'karnof',
    'oprior', 
    'z30', 
    'zprior', 
    'preanti', 
    'race', 
    'gender', 
    'str2', 
    'strat',
    'symptom', 
    'treat', 
    'offtrt', 
    'cd40', 
    'cd420', 
    'cd80', 
    'cd820'
]

class CustomNaiveBayes(BaseEstimator, ClassifierMixin):
    def __init__(self, features, m):
        self.features = features
        self.m = m

    def fit(self, X_train, y_train):
        self.classes_ = np.unique(y_train)
        self.X_train = X_train.copy()
        self.y_train = y_train.copy()
        self.X_train[target] = y_train
        return self

    def predict(self, X_test):
        y_pred = []
        for i in range(0, X_test.shape[0]):
            instance = X_test.iloc[i]
            y_pred.append(naive_bayes(self.X_train, target, self.features, instance, self.m))
        return np.array(y_pred)
    
    def __sklearn_clone__(self):
        return self

def init():
    return dataset, features, continuous_features, target

# Discretize continuous features
kbins = KBinsDiscretizer(n_bins=2, encode='ordinal', strategy='quantile')
dataset[continuous_features] = kbins.fit_transform(dataset[continuous_features])

# Feature Selection
X = dataset.drop([target, 'pidnum'], axis=1)
y = dataset[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
        
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


def test_instances(X_train, y_train, X_test, y_test, features, m):
    y_pred = []
    train_ds = X_train.copy()
    train_ds[target] = y_train
    for i in range(0, X_test.shape[0]):
        instance = X_test.iloc[i]
        y_pred.append(naive_bayes(train_ds, target, features, instance, m))
    return accuracy_score(y_test, y_pred) * 100


# Without feature selection
accuracy_no_selection = test_instances(X_train, y_train, X_test, y_test, features, 1)

# With feature selection (all features)
# selector = SelectKBest(chi2, k=2)  # Select all features
# X_train_selected = selector.fit_transform(X_train, y_train)
# selected_features = X_train.columns[selector.get_support()]

# Convert the selected features back to DataFrames with feature names
""" X_train_selected = X_train[selected_features]
X_test_selected = X_test[selected_features]

accuracy_with_selection = test_instances(X_train_selected, y_train, X_test_selected, y_test, selected_features, 1)

print(f"Accuracy without feature selection: {accuracy_no_selection}%")
print(f"Accuracy with all features selected: {accuracy_with_selection}%") """

# Define accuracy as the scoring metric (a cambiar, no es lo mejor basarnos en solo accuracy)
scorer = make_scorer(accuracy_score)
acc = []
# misma idea para sin feature selection
for i in range(0, 23):
    selector = SelectKBest(chi2, k=i+1)
    X_train_selected = selector.fit_transform(X_train, y_train)
    selected_features = X_train.columns[selector.get_support()]
    model = CustomNaiveBayes(selected_features, m=1)
    X_selected = X[selected_features]  # Use the features selected earlier
    scores_with_selection = cross_val_score(model, X_selected, y, cv=5, scoring=scorer)
    acc.append(np.mean(scores_with_selection))

# Generate a sequence of x values corresponding to the number of features selected
x_values = range(1, 24)  # From 1 to 23 features

# Plot the accuracy values
plt.plot(x_values, acc, marker='o', linestyle='-', color='b')

# Add labels and title
plt.xlabel('Number of Selected Features')
plt.ylabel('Cross-Validated Accuracy')
plt.title('Accuracy vs. Number of Selected Features')

# Display the plot
plt.show()
    
    
""" # Initialize custom Naive Bayes classifier
model = CustomNaiveBayes(features, m=1)

# Perform 5-fold cross-validation on the dataset without feature selection
scores_no_selection = cross_val_score(model, X, y, cv=5, scoring=scorer)

print("Cross-validation scores without feature selection:", scores_no_selection)
print("Average accuracy without feature selection:", np.mean(scores_no_selection))


model = CustomNaiveBayes(selected_features, m=1)
# Perform 5-fold cross-validation on the dataset with feature selection
X_selected = X[selected_features]  # Use the features selected earlier
scores_with_selection = cross_val_score(model, X_selected, y, cv=5, scoring=scorer)

print("Cross-validation scores with feature selection:", scores_with_selection)
print("Average accuracy with feature selection:", np.mean(scores_with_selection)) """