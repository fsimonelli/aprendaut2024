from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import torch

def train_model(model, X_train, y_train, X_val, y_val, epochs=100, lr=0.01, criterion='CrossEntropyLoss'):
    if (criterion == 'CrossEntropyLoss'):
        criterion = torch.nn.CrossEntropyLoss()
    elif (criterion == 'BCELoss'): 
        criterion = torch.nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []
    
    best_val_accuracy = 0
    patience = 10  # Número de épocas para esperar antes de detener
    patience_counter = 0
    
    # Bucle de entrenamiento
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        # Calculate training metrics
        train_loss = loss.item()
        _, predicted = torch.max(outputs.data, 1)
        train_acc = (predicted == y_train).sum().item() / len(y_train)
        
        # Validation metrics
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val)
            val_loss = criterion(val_outputs, y_val).item()
            _, val_predicted = torch.max(val_outputs.data, 1)
            val_acc = (val_predicted == y_val).sum().item() / len(y_val)
        
        # Store metrics
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        
        """ # early stopping
        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc  # Actualizar la mejor precisión de validación
            patience_counter = 0         # Reiniciar el contador de paciencia
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            break """
    
    return train_losses, val_losses, train_accuracies, val_accuracies
