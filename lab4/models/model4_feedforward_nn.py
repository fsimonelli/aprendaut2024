import torch
import torch.nn as nn

class FeedForwardNN(nn.Module):
    def __init__(self, input_size):
        super(FeedForwardNN, self).__init__()
        self.hidden = nn.Linear(input_size, 16)
        self.output = nn.Linear(16, 1)

    def forward(self, x):
        x = torch.sigmoid(self.hidden(x))
        return torch.sigmoid(self.output(x))
    
    def predict(self, x, threshold=0.5):
        self.eval()  # Configurar el modelo en modo de evaluación
        with torch.no_grad():  # Desactivar gradientes para inferencia
            probabilities = self.forward(x)  # Obtener probabilidades
            predictions = (probabilities >= threshold).int()  # Convertir a etiquetas binarias
        return predictions
