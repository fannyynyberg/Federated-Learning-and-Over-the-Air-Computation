import torch
import torch.nn as nn
import torch.nn.functional as F

class NeuralNetwork(nn.Module):
    """Simple feedforward neural network with one hidden layer."""
    def __init__(self, layer_1, layer_2=10):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(784, layer_1)
        self.fc2 = nn.Linear(layer_1, layer_2)

    def forward(self, x):
        x = x.view(-1, 784)  # Flatten the input
        x = F.relu(self.fc1(x))
        x = self.fc2(x)  # No activation before final output
        return x

class Trainer:
    """Trainer class to handle training and weight management."""
    def __init__(self, model, lr=0.01):
        self.model = model
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)
    
    def train(self, train_loader, epochs=1):
        self.model.train()
        for epoch in range(epochs):
            for images, labels in train_loader:
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
    
    def get_weights(self):
        return [param.data.clone() for param in self.model.parameters()]
    
    def set_weights(self, new_weights):
        with torch.no_grad():
            for param, new_weight in zip(self.model.parameters(), new_weights):
                param.data.copy_(new_weight)
