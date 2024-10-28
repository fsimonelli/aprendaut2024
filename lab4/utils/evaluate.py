from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch
import pandas as pd

def evaluate_model(model, X_test, y_test):
    if isinstance(X_test, pd.DataFrame):
        X_test = torch.tensor(X_test.values, dtype=torch.float32)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    return accuracy, precision, recall, f1
