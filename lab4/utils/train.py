from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import torch.nn as nn
import torch

def train_model(model, X_train, y_train, X_val, y_val, epochs=100, lr=0.01, criterion='CrossEntropyLoss', modelNumber=2):
    if (criterion == 'CrossEntropyLoss'):
        loss_fn = torch.nn.CrossEntropyLoss()
    elif (criterion == 'BCELoss'): 
        loss_fn = torch.nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []
    
    for epoch in range(epochs):
        model.train()

        y_train_pred = model(X_train)
        loss = loss_fn(y_train_pred, y_train)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Calculate training metrics
        train_loss = loss.item()
        if (modelNumber == 2):
            predicted = torch.argmax(y_train_pred, dim=1)
        else:
            predicted = (y_train_pred >= 0.5).float()
        train_acc = (predicted == y_train).sum().item() / y_train.size(0)
        
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        
        # Validation metrics
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val)
            val_loss = loss_fn(val_outputs, y_val).item()
            if (modelNumber == 2):
                val_predicted = torch.argmax(val_outputs, dim=1)
            else:
                val_predicted = (val_outputs >= 0.5).float()
            val_acc = (val_predicted == y_val).sum().item() / len(y_val)
        
        # Store metrics
        
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
    
    return train_losses, val_losses, train_accuracies, val_accuracies
