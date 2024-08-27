import pandas as pd
DATASET_FILE = "./pedro.csv"

dataset = pd.read_csv(DATASET_FILE, sep=",")
print(
    f"{dataset.shape[0]} records read from {DATASET_FILE}\n{dataset.shape[1]} attributes found"
)
print(dataset.head(10))

# implementar el algoritmo ID3 con la extensión para atributos numéricos
