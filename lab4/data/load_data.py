import pandas as pd
from sklearn.model_selection import train_test_split

def load_and_split_data(filepath):
    data = pd.read_csv(filepath)
    data = data.drop(columns=['pidnum','time'])
    train_full, test = train_test_split(data, test_size=0.2, random_state=0)
    train, val = train_test_split(train_full, test_size=0.1, random_state=0)
    return train, val, test
