# Description: MNIST, NN2
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend before importing pyplot
import matplotlib.pyplot as plt
from NN2 import NeuralNetwork

# Federated Learning setup
num_clients = 20
num_rounds = 50
epochs = 2
learning_rate = 0.01

# Transformation pipeline that converts images into PyTorch tensors and normalizes pixel values
transform = transforms.Compose([transforms.ToTensor()])
# Load MNIST dataset
mnist_data = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
# Splits the dataset between clients
client_data = torch.utils.data.random_split(mnist_data, [len(mnist_data)//num_clients]*num_clients)

# Define training model for local model
def train_local(client_model, data_loader, optimizer, criterion):
    # Set model to training mode
    client_model.train()
    # Iterate over the training data
    for images, labels in data_loader:
        # Clears the gradient before a new batch is processed
        optimizer.zero_grad()
        # Produces ten outputs
        outputs = client_model(images)
        # Compares model prediction with actual labels, where loss quantifies how far off the predictions are
        loss = criterion(outputs, labels)
        # Computes gradients of the loss with respect to model weights using backpropagation
        loss.backward()
        # Updates the model's weights using computed gradients
        optimizer.step()

# Define federated averaging function
def federated_avg(weights):
    # Initialize a dictionary to store the average weights
    avg_weights = {}
    # Iterate over the keys of the weights dictionary
    for key in weights[0].keys():
        # Compute the average of the weights for each key
        avg_weights[key] = sum(w[key] for w in weights) / len(weights)
        # Convert the average to a tensor
    return avg_weights

# Define test function 
def test_model(model, test_loader):
    # Set model to evaluation mode
    model.eval()
    # Initialize counters for correct predictions
    correct = 0
    # Initialize total number of predictions
    total = 0
    # Disable gradient computation
    with torch.no_grad():
        # Iterate over the test data
        for images, labels in test_loader:
            # Produces ten outputs
            outputs = model(images)
            # Choose the class with the maximum probability
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            # Update the counter for correct predictions
            correct += (predicted == labels).sum().item()
            # Return the accuracy
    return 100 * correct / total

# Initialize global model
global_model = NeuralNetwork()
# Initialize loss function
criterion = nn.CrossEntropyLoss()
# Initialize test loader
test_loader = data.DataLoader(datasets.MNIST(root="./data", train=False, download=True, transform=transform), batch_size=64, shuffle=False)

# Initialize list to store accuracies
accuracies = []

# Train the global model
import time
# Record the start time
start_time = time.time()

# Iterate over the rounds
for round in range(num_rounds):
    # Record the start time of the round
    round_start = time.time()
    # Initialize a list to store the weights of the clients
    client_weights = []
    
    # Train the local models
    for i in range(num_clients):
        # Initialize a local model
        local_model = NeuralNetwork()
        # Load the global model's weights
        local_model.load_state_dict(global_model.state_dict())
        # Initialize the optimizer
        optimizer = optim.SGD(local_model.parameters(), lr=learning_rate)
        # Initialize the data loader
        data_loader = data.DataLoader(client_data[i], batch_size=32, shuffle=True)
        # Train the local model
        train_local(local_model, data_loader, optimizer, criterion)
        # Append the local model's weights
        client_weights.append(local_model.state_dict())
    
    # Compute the global weights
    global_weights = federated_avg(client_weights)
    # Load the global weights to the global model
    global_model.load_state_dict(global_weights)
    
    # Test the global model
    accuracy = test_model(global_model, test_loader)
    # Append the accuracy to the list
    accuracies.append(accuracy)
    # Record the time it took to complete the round
    round_time = time.time() - round_start
    # Print the results of the round
    print(f"Round {round+1} completed - Accuracy: {accuracy:.2f}% - Time: {round_time:.2f} sec")

# Plot accuracy over rounds
plt.plot(range(1, num_rounds + 1), accuracies, marker='o')
plt.xlabel('Round')
plt.ylabel('Accuracy (%)')
plt.title('Federated Learning Convergence')
plt.grid()
total_time = time.time() - start_time
avg_time_per_round = total_time / num_rounds
print(f"Total training time: {total_time:.2f} seconds")
print(f"Average time per round: {avg_time_per_round:.2f} seconds")

#plt.show()

plt.savefig("federated_learning_convergence.png")  # Saves the plot as an image file
print("Plot saved as federated_learning_convergence.png")

