import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
from scipy.special import expi
from MLP import NeuralNetwork
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time

# Configuration
num_rounds = 100
epochs = 1
learning_rate = 0.01
noise_variance = 0.001
threshold = 0.2
P0 = 0.2
rho = P0 / (-expi(-threshold))
client_counts = [20, 30, 40, 50]
colors = ['blue', 'green', 'orange', 'red']

# MNIST dataset setup
transform = transforms.Compose([transforms.ToTensor()])
mnist_data = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
test_loader = data.DataLoader(datasets.MNIST(root="./data", train=False, download=True, transform=transform), batch_size=64, shuffle=False)

# Train local model
def train_local(client_model, data_loader, optimizer, criterion):
    client_model.train()
    for images, labels in data_loader:
        optimizer.zero_grad()
        outputs = client_model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

# AirComp aggregation function
def aircomp_aggregate(weights):
    avg_weights = {}
    num_clients = len(weights)
    h = np.random.rayleigh(scale=1.0, size=num_clients)
    h_sq = h ** 2
    active_clients = [i for i in range(num_clients) if h_sq[i] >= threshold]
    A = len(active_clients)
    if A == 0:
        raise RuntimeError("No clients passed the threshold. Adjust the threshold value.")
    print(f"AirComp aggregation: {A}/{num_clients} clients active this round")
    for key in weights[0].keys():
        aggregated_value = 0.0
        for i in active_clients:
            hi = h[i]
            pk = np.sqrt(rho) / hi
            aggregated_value += weights[i][key] * pk
        noise = torch.normal(mean=0.0, std=noise_variance ** 0.5, size=aggregated_value.shape)
        aggregated_value += noise
        avg_weights[key] = aggregated_value / (A * np.sqrt(rho))
    return avg_weights

# Evaluate global model
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

# Main loop for multiple client counts
plt.figure(figsize=(10, 6))

for idx, num_clients in enumerate(client_counts):
    print(f"\n--- AirComp Training with {num_clients} clients ---")
    client_lengths = [len(mnist_data) // num_clients] * num_clients
    client_lengths[-1] += len(mnist_data) - sum(client_lengths)
    client_data = torch.utils.data.random_split(mnist_data, client_lengths)

    global_model = NeuralNetwork()
    criterion = nn.CrossEntropyLoss()
    accuracies = []

    for round in range(num_rounds):
        round_weights = []
        for i in range(num_clients):
            local_model = NeuralNetwork()
            local_model.load_state_dict(global_model.state_dict())
            optimizer = optim.SGD(local_model.parameters(), lr=learning_rate)
            data_loader = data.DataLoader(client_data[i], batch_size=32, shuffle=True)
            train_local(local_model, data_loader, optimizer, criterion)
            round_weights.append(local_model.state_dict())

        global_weights = aircomp_aggregate(round_weights)
        global_model.load_state_dict(global_weights)
        acc = test_model(global_model, test_loader)
        accuracies.append(acc)
        print(f"Round {round+1} - Accuracy: {acc:.2f}%")

    plt.plot(range(1, num_rounds + 1), accuracies, label=f'{num_clients} clients', color=colors[idx])

# Finalize and save the plot
plt.xlabel('Rounds')
plt.ylabel('Accuracy (%)')
plt.title('AirComp Convergence for Different Number of Clients')
plt.legend()
plt.grid()
plt.savefig("aircomp_learning_clients_comparison.png")
print("\nPlot saved as aircomp_learning_clients_comparison.png")