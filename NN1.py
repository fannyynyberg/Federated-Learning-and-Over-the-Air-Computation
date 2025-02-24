import torch
import torch.nn as nn
import torch.nn.functional as F

class NeuralNetwork(nn.Module):
    """Simple feedforward neural network with one hidden layer."""
    def __init__(self, layer_1, layer_2=10):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(784, layer_1)
        nn.init.kaiming_normal_(self.fc1.weight, nonlinearity='relu')
        self.fc2 = nn.Linear(layer_1, layer_2)
        nn.init.kaiming_normal_(self.fc2.weight, nonlinearity='relu')

    def forward(self, x):
        x = x.view(-1, 784)  # Flatten the input
        x = F.relu(self.fc1(x))
        x = self.fc2(x)  # No activation before final output
        return x