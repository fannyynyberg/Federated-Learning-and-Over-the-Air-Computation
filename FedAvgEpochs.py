import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from MLP import NeuralNetwork
import time

# Transformation pipeline
transform = transforms.Compose([transforms.ToTensor()])

# Load MNIST once
mnist_data = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
test_loader = data.DataLoader(datasets.MNIST(root="./data", train=False, download=True, transform=transform),
                              batch_size=64, shuffle=False)

# Training function
def train_local(client_model, data_loader, optimizer, criterion, epochs):
    client_model.train()
    for _ in range(epochs):
        for images, labels in data_loader:
            optimizer.zero_grad()
            outputs = client_model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

# Federated averaging
def federated_avg(weights):
    avg_weights = {}
    for key in weights[0].keys():
        avg_weights[key] = sum(w[key] for w in weights) / len(weights)
    return avg_weights

# Test function
def test_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

# Configuration
num_rounds = 100
learning_rate = 0.01
num_clients = 20
epoch_counts = [1, 2, 3, 4]
colors = ['blue', 'green', 'orange', 'red']

plt.figure(figsize=(10, 6))

# Fix number of clients
print(f"\n--- Training with {num_clients} clients ---")
client_lengths = [len(mnist_data) // num_clients] * num_clients
client_lengths[-1] += len(mnist_data) - sum(client_lengths)
client_data = torch.utils.data.random_split(mnist_data, client_lengths)

# Loop over different number of epochs
for idx, epochs in enumerate(epoch_counts):
    print(f"\n--- Using {epochs} local epochs per client ---")
    global_model = NeuralNetwork()
    criterion = nn.CrossEntropyLoss()
    accuracies = []

    for round in range(num_rounds):
        client_weights = []
        for i in range(num_clients):
            local_model = NeuralNetwork()
            local_model.load_state_dict(global_model.state_dict())
            optimizer = optim.SGD(local_model.parameters(), lr=learning_rate)
            data_loader = data.DataLoader(client_data[i], batch_size=32, shuffle=True)
            train_local(local_model, data_loader, optimizer, criterion, epochs)
            client_weights.append(local_model.state_dict())

        global_weights = federated_avg(client_weights)
        global_model.load_state_dict(global_weights)
        accuracy = test_model(global_model, test_loader)
        accuracies.append(accuracy)
        print(f"Round {round+1} - Accuracy: {accuracy:.2f}%")

    # Plot accuracy for this number of epochs
    plt.plot(range(1, num_rounds + 1), accuracies, label=f'{epochs} epoch(s)', color=colors[idx])

# Finalize and save the plot
plt.xlabel('Rounds')
plt.ylabel('Accuracy (%)')
plt.title('FedAvg Convergence for Different Number of Local Epochs')
plt.legend()
plt.grid()
plt.savefig("fedavg_local_epochs_comparison.png")
print("\nPlot saved as fedavg_local_epochs_comparison.png")