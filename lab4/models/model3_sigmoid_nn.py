import torch
import torch.nn as nn

class SigmoidNN(nn.Module):
    def __init__(self, input_size):
        super(SigmoidNN, self).__init__()
        self.linear = nn.Linear(input_size, 1)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))
    
    def predict(self, x, threshold=0.5):
        self.eval()  # Establecer el modelo en modo de evaluación
        with torch.no_grad():  # Desactivar gradientes para inferencia
            probabilities = self.forward(x)  # Obtener probabilidades
            predictions = (probabilities >= threshold).int()  # Convertir a etiquetas binarias
        return predictions
