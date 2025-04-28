import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
from scipy.special import expi
import matplotlib.pyplot as plt
from CNN import CNN
from MLP import MLP  # MLP för MNIST

# Gemensamma parametrar
num_clients = 20
num_rounds = 100
epochs = 2
learning_rate = 0.01
noise_variance = 1e-7
threshold = 0.1
P0 = 0.5
rho = P0 / (-expi(-threshold))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def split_dataset(dataset):
    base_len = len(dataset) // num_clients
    lengths = [base_len] * num_clients
    for i in range(len(dataset) % num_clients):
        lengths[i] += 1
    return torch.utils.data.random_split(dataset, lengths)

def train_local(model, data_loader, optimizer, criterion):
    model.train()
    for images, labels in data_loader:
        images, labels = images.to(device), labels.to(device)
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
        raise RuntimeError("Inga klienter passerade tröskeln. Justera threshold-värdet.")

    for key in weights[0].keys():
        aggregated_value = torch.zeros_like(weights[0][key], dtype=torch.float32)
        for i in active_clients:
            hi = h[i]
            pk = np.sqrt(rho) / hi
            aggregated_value += weights[i][key].float() * pk
        noise = torch.normal(mean=0.0, std=noise_variance ** 0.5, size=aggregated_value.shape)
        aggregated_value += noise
        avg_weights[key] = aggregated_value / (A * np.sqrt(rho))
    return avg_weights

def test_model(model, test_loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

def federated_training(dataset_name):
    if dataset_name == "MNIST":
        transform = transforms.Compose([transforms.ToTensor()])
        train_set = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
        test_set = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
        ModelClass = MLP
    elif dataset_name == "CIFAR-10":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        train_set = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
        test_set = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
        ModelClass = CNN
    else:
        raise ValueError("Endast MNIST eller CIFAR-10 stöds")

    client_data = split_dataset(train_set)
    test_loader = data.DataLoader(test_set, batch_size=64, shuffle=False)

    global_model = ModelClass().to(device)
    criterion = nn.CrossEntropyLoss()
    accuracies = []

    for round in range(num_rounds):
        client_weights = []
        for i in range(num_clients):
            local_model = ModelClass().to(device)
            local_model.load_state_dict(global_model.state_dict())
            optimizer = optim.SGD(local_model.parameters(), lr=learning_rate)
            data_loader = data.DataLoader(client_data[i], batch_size=32, shuffle=True)
            train_local(local_model, data_loader, optimizer, criterion)
            client_weights.append(local_model.cpu().state_dict())

        global_weights = aircomp_aggregate(client_weights)
        global_model.load_state_dict(global_weights)
        global_model.to(device)

        acc = test_model(global_model, test_loader)
        accuracies.append(acc)
        print(f"[{dataset_name}] Runda {round+1}: {acc:.2f}%")

    return accuracies

# Kör båda experimenten
mnist_accuracies = federated_training("MNIST")
cifar_accuracies = federated_training("CIFAR-10")

# Plot
plt.figure(figsize=(10, 6))
plt.plot(range(1, num_rounds + 1), mnist_accuracies, label="MNIST", color="blue")
plt.plot(range(1, num_rounds + 1), cifar_accuracies, label="CIFAR-10", color="green")
plt.xlabel("Rounds")
plt.ylabel("Accuracy (%)")
plt.title("AirComp Convergence on MNIST and CIFAR-10")
plt.legend()
plt.grid()
plt.savefig("aircomp_mnist_vs_cifar10.png")
print("\nPlot saved as aircomp_mnist_vs_cifar10.png")