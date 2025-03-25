# Description: MNIST, Non-IID, Neural Network (no CNN)
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
from MLP import NeuralNetwork

# Federated Learning setup
num_clients = 10  # Number of clients
num_rounds = 100   # Communication rounds
epochs = 1        # Local training epochs
learning_rate = 0.01

# Transformation pipeline (convert images to PyTorch tensors)
transform = transforms.Compose([transforms.ToTensor()])

# Load MNIST dataset
mnist_data = datasets.MNIST(root="./data", train=True, download=True, transform=transform)

# Function to create Non-IID data (Label Skew)
def split_mnist_non_iid(mnist_dataset, num_clients, num_shards=40):
    """
    Creates a non-IID partition of MNIST data.
    Each client will receive only a subset of labels.
    """
    num_samples = len(mnist_dataset)
    data_indices = np.arange(num_samples)
    labels = np.array(mnist_dataset.targets)

    # Sort by label to make it easier to split by class
    sorted_indices = data_indices[np.argsort(labels)]

    # Split into shards
    shards = np.array_split(sorted_indices, num_shards)
    np.random.shuffle(shards)

    # Assign shards to clients (each client gets num_shards // num_clients)
    client_data = {i: np.concatenate(shards[i::num_clients]) for i in range(num_clients)}

    return client_data

# Create non-IID client datasets
client_indices = split_mnist_non_iid(mnist_data, num_clients)

# Create client-specific DataLoaders
client_data = {i: torch.utils.data.Subset(mnist_data, client_indices[i]) for i in range(num_clients)}

# Define local training function
def train_local(client_model, data_loader, optimizer, criterion):
    client_model.train()
    for images, labels in data_loader:
        optimizer.zero_grad()
        outputs = client_model(images.view(-1, 28*28))  # Flatten input
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

# Federated averaging function
def federated_avg(weights):
    avg_weights = {}
    for key in weights[0].keys():
        avg_weights[key] = sum(w[key] for w in weights) / len(weights)
    return avg_weights

# Define test function
def test_model(model, test_loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images.view(-1, 28*28))  # Flatten input
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

# Initialize global model
global_model = NeuralNetwork()
criterion = nn.CrossEntropyLoss()
test_loader = data.DataLoader(datasets.MNIST(root="./data", train=False, download=True, transform=transform), batch_size=64, shuffle=False)

# Training loop
accuracies = []
import time
start_time = time.time()

for round in range(num_rounds):
    round_start = time.time()
    client_weights = []

    for i in range(num_clients):
        local_model = NeuralNetwork()
        local_model.load_state_dict(global_model.state_dict())
        optimizer = optim.SGD(local_model.parameters(), lr=learning_rate)
        data_loader = data.DataLoader(client_data[i], batch_size=32, shuffle=True)
        train_local(local_model, data_loader, optimizer, criterion)
        client_weights.append(local_model.state_dict())

    # Aggregate updates
    global_weights = federated_avg(client_weights)
    global_model.load_state_dict(global_weights)

    # Test global model
    accuracy = test_model(global_model, test_loader)
    accuracies.append(accuracy)
    round_time = time.time() - round_start
    print(f"Round {round+1} - Accuracy: {accuracy:.2f}% - Time: {round_time:.2f} sec")

# Plot accuracy over rounds
plt.plot(range(1, num_rounds + 1), accuracies, marker='o')
plt.xlabel('Round')
plt.ylabel('Accuracy (%)')
plt.title('Federated Learning with Non-IID MNIST')
plt.grid()
plt.savefig("federated_learning_non_iid.png")
print("Plot saved as federated_learning_non_iid.png")

total_time = time.time() - start_time
print(f"Total training time: {total_time:.2f} seconds")
print(f"Average time per round: {total_time / num_rounds:.2f} seconds")
