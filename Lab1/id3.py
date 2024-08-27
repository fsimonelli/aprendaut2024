import pandas as pd
DATASET_FILE = "./pedro.csv"

dataset = pd.read_csv(DATASET_FILE, sep=",")

print(
    f"{dataset.shape[0]} records read from {DATASET_FILE}\n{dataset.shape[1]} attributes found"
)

print(dataset.Salva.value_counts())
# implementar el algoritmo ID3 con la extensión para atributos numéricos


# Entropy for boolean functions.
def entropy(dataset, column):
    values = dataset[column].value_counts()
    total = values.sum()
    return -(values[0]/total)*np.log2(values[0]/total) - (values[1]/total) * np.log2(values[1]/total)

def best_attribute(dataset, column, attributes):
    entropies = []
    for attribute in attributes:
        
        entropies.append(entropy(dataset, attribute))

def id3(dataset, column, attributes):
    if len(attributes) == 0:
        return dataset[column].value_counts()[0]
    if len(dataset[column].value_counts()) == 1:
        return dataset[column].value_counts()[0]
    
    