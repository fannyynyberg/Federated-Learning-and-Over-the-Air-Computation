import torch
import torch.nn as nn
import torch.nn.functional as F

# Define a neural network
class MLP(nn.Module):
    # Define the layers
    def __init__(self):
        # Call the parent constructor nn.Module
        super(MLP, self).__init__()
        # Define the layers of the network
        self.fc1 = nn.Linear(28*28, 128)
        # Define the activation function
        self.relu = nn.ReLU() # Rectified Linear Unit
        # Define the output layer
        self.fc2 = nn.Linear(128, 10)
    
    # Define the forward pass
    def forward(self, x):
        # Flatten the input
        x = x.view(-1, 28*28)
        # Pass the input through the first layer
        x = self.fc1(x)
        # Apply the activation function
        x = self.relu(x)
        # Pass the output through the second layer
        x = self.fc2(x)
        # Return the output
        return x