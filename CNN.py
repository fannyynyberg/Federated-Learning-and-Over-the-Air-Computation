import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Define a neural network with three fully connected layers
class NeuralNetwork(nn.Module):
    def __init__(self, input_size=28*28, hidden_sizes=[128, 64], output_size=10):
        super(NeuralNetwork, self).__init__()
        
        # First fully connected layer
        self.fc1 = nn.Linear(input_size, hidden_sizes[0])
        # Second fully connected layer
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        # Output layer
        self.fc3 = nn.Linear(hidden_sizes[1], output_size)
        
    def forward(self, x):
        # Flatten the input tensor
        x = x.view(-1, 28*28)  # Flatten input
        # Apply ReLU activation after first layer
        x = torch.relu(self.fc1(x))
        # Apply ReLU activation after second layer
        x = torch.relu(self.fc2(x))
        # Compute final output without activation (for classification)
        x = self.fc3(x)
        return x

# Trainer class to handle training, weight extraction, and weight setting
class Trainer:
    def __init__(self, model, lr=0.01):
        self.model = model
        self.criterion = nn.CrossEntropyLoss() # Loss function for classification
        self.optimizer = optim.SGD(self.model.parameters(), lr=lr) # Stochastic Gradient Descent optimizer
    
    def train(self, train_loader, epochs=1):
        """Train the model for a given number of epochs using the provided data loader."""
        self.model.train() # Set model to training mode
        for epoch in range(epochs):
            for images, labels in train_loader:
                self.optimizer.zero_grad() # Reset gradients
                outputs = self.model(images) # Forward pass
                loss = self.criterion(outputs, labels) # Compute loss
                loss.backward() # Backpropagation
                self.optimizer.step() # Update model parameters

    def get_weights(self):
        """Return a copy of the model's current weights."""
        return [param.data.clone() for param in self.model.parameters()]
    
    def set_weights(self, new_weights):
        """Update the model's weights with new values."""
        with torch.no_grad(): # Disable gradient tracking
            for param, new_weight in zip(self.model.parameters(), new_weights):
                param.data.copy_(new_weight)
