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
import time
from scipy.special import expi

from MLP import NeuralNetwork
from CNN import CNN

# Shared config
num_clients = 20
num_rounds = 100
epochs = 1
learning_rate = 0.01

# AirComp parameters
threshold = 0.2
P0 = 0.2
noise_variance = 0.001
rho = P0 / (-expi(-threshold))

# ---------- MNIST SETUP ----------
print("\n--- Starting AirComp on MNIST ---")
transform_mnist = transforms.Compose([transforms.ToTensor()])
mnist_data = datasets.MNIST(root="./data", train=True, download=True, transform=transform_mnist)
client_data_mnist = torch.utils.data.random_split(mnist_data, [len(mnist_data) // num_clients] * num_clients)
test_loader_mnist = data.DataLoader(datasets.MNIST(root="./data", train=False, download=True, transform=transform_mnist), batch_size=64, shuffle=False)

# ---------- CIFAR-10 SETUP ----------
print("\n--- Starting AirComp on CIFAR-10 ---")
transform_cifar_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
transform_cifar_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
cifar10_train = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform_cifar_train)
client_data_cifar = torch.utils.data.random_split(cifar10_train, [len(cifar10_train) // num_clients] * num_clients)
cifar10_test = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform_cifar_test)
test_loader_cifar = data.DataLoader(cifar10_test, batch_size=64, shuffle=False)

# ---------- SHARED TRAINING UTILS ----------
def train_local(model, data_loader, optimizer, criterion):
    model.train()
    for images, labels in data_loader:
        optimizer.zero_grad()
        outputs = model(images)
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

# ---------- RUN MNIST AIRCOMP ----------
mnist_model = NeuralNetwork()
mnist_criterion = nn.CrossEntropyLoss()
mnist_accuracies = []

for rnd in range(num_rounds):
    client_weights = []
    for i in range(num_clients):
        local_model = NeuralNetwork()
        local_model.load_state_dict(mnist_model.state_dict())
        optimizer = optim.SGD(local_model.parameters(), lr=learning_rate)
        loader = data.DataLoader(client_data_mnist[i], batch_size=32, shuffle=True)
        train_local(local_model, loader, optimizer, mnist_criterion)
        client_weights.append(local_model.state_dict())
    global_weights = aircomp_aggregate(client_weights)
    mnist_model.load_state_dict(global_weights)
    acc = test_model(mnist_model, test_loader_mnist)
    mnist_accuracies.append(acc)
    print(f"[MNIST] Round {rnd+1} - Accuracy: {acc:.2f}%")

# ---------- RUN CIFAR AIRCOMP ----------
cifar_model = CNN()
cifar_criterion = nn.CrossEntropyLoss()
cifar_accuracies = []

for rnd in range(num_rounds):
    client_weights = []
    for i in range(num_clients):
        local_model = CNN()
        local_model.load_state_dict(cifar_model.state_dict())
        optimizer = optim.SGD(local_model.parameters(), lr=learning_rate)
        loader = data.DataLoader(client_data_cifar[i], batch_size=32, shuffle=True)
        train_local(local_model, loader, optimizer, cifar_criterion)
        client_weights.append(local_model.state_dict())
    global_weights = aircomp_aggregate(client_weights)
    cifar_model.load_state_dict(global_weights)
    acc = test_model(cifar_model, test_loader_cifar)
    cifar_accuracies.append(acc)
    print(f"[CIFAR-10] Round {rnd+1} - Accuracy: {acc:.2f}%")

# ---------- FINAL PLOTTING ----------
plt.figure(figsize=(10, 6))
plt.plot(range(1, num_rounds + 1), mnist_accuracies, label="MNIST", color="blue")
plt.plot(range(1, num_rounds + 1), cifar_accuracies, label="CIFAR-10", color="orange")
plt.xlabel("Rounds")
plt.ylabel("Accuracy (%)")
plt.title("AirComp Convergence on MNIST and CIFAR-10")
plt.legend()
plt.grid()
plt.savefig("aircomp_mnist_vs_cifar10.png")
print("\nPlot saved as aircomp_mnist_vs_cifar10.png")
