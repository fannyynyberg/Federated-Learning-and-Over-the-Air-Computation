import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
from scipy.special import expi
from MLP import MLP

num_clients = 20
num_rounds = 100
epochs = 2
learning_rate = 0.01
noise_variance = 0.0000001

threshold = 0.1
P0 = 0.5
rho = P0 / (-expi(-threshold))

transform = transforms.Compose([transforms.ToTensor()])
mnist_data = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
client_data = torch.utils.data.random_split(mnist_data, [len(mnist_data)//num_clients]*num_clients)

def train_local(client_model, data_loader, optimizer, criterion):
    client_model.train()
    for images, labels in data_loader:
        optimizer.zero_grad()
        outputs = client_model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

def aircomp_aggregate(weights):
    avg_weights = {}
    num_clients = len(weights)

    h = np.random.rayleigh(scale=1.0, size=num_clients)
    h_sq = h ** 2
    active_clients = [i for i in range(num_clients) if h_sq[i] >= threshold]
    A = len(active_clients)
    if A == 0:
        raise RuntimeError("No clients passed the threshold. Adjust threshold.")
    print(f"AirComp aggregation: {A}/{num_clients} clients active")

    for key in weights[0].keys():
        aggregated_value = 0.0
        for i in active_clients:
            hi = h[i]
            pk = np.sqrt(rho) / hi
            aggregated_value += weights[i][key] * pk

        noise = 0
        aggregated_value += noise

        avg_weights[key] = aggregated_value / (A * np.sqrt(rho))

    return avg_weights

def test_model(model, test_loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

global_model = MLP()
criterion = nn.CrossEntropyLoss()
test_loader = data.DataLoader(datasets.MNIST(root="./data", train=False, download=True, transform=transform), batch_size=64, shuffle=False)

accuracies = []

import time
start_time = time.time()

for round in range(num_rounds):
    round_start = time.time()
    client_weights = []

    for i in range(num_clients):
        local_model = MLP()
        local_model.load_state_dict(global_model.state_dict())
        optimizer = optim.SGD(local_model.parameters(), lr=learning_rate)
        data_loader = data.DataLoader(client_data[i], batch_size=32, shuffle=True)
        train_local(local_model, data_loader, optimizer, criterion)
        client_weights.append(local_model.state_dict())

    global_weights = aircomp_aggregate(client_weights)
    global_model.load_state_dict(global_weights)

    accuracy = test_model(global_model, test_loader)
    accuracies.append(accuracy)
    round_time = time.time() - round_start
    print(f"Round {round+1} completed - Accuracy: {accuracy:.2f}% - Time: {round_time:.2f} seconds")

import matplotlib.pyplot as plt
plt.plot(range(1, num_rounds + 1), accuracies, marker='o')
plt.xlabel('Round')
plt.ylabel('Accuracy (%)')
plt.title('AirComp IID MNIST MLP Convergence')
plt.grid()
plt.savefig("aircomp_iid_mnist_mlp_convergence.png")
print("Figure saved as aircomp_iid_mnist_mlp_convergence.png")
