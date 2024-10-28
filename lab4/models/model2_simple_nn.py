import torch
import torch.nn as nn

class SimpleLinearNN(nn.Module):
    def __init__(self, input_size):
        super(SimpleLinearNN, self).__init__()
        self.linear = nn.Linear(input_size, 2)

    def forward(self, x):
        return self.linear(x)
    
    def predict(self, x):
        self.eval()
        
        with torch.no_grad():
            # Pasar los datos a través del modelo
            logits = self.forward(x)
            
            # Obtener la clase con la probabilidad más alta
            _, predicted_classes = torch.max(logits, 1)
        
        return predicted_classes
