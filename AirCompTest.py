import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
from scipy.special import expi
from CNN import CNN
import matplotlib.pyplot as plt
import time
import os

# Federated Learning setup
num_clients = 20
num_rounds = 100
learning_rate = 0.01
noise_variance = 0.01

# CIFAR-10 Transformation pipeline
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load CIFAR-10 dataset
cifar10_train = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
cifar10_test = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)

client_data = torch.utils.data.random_split(cifar10_train, [len(cifar10_train) // num_clients] * num_clients)
test_loader = data.DataLoader(cifar10_test, batch_size=64, shuffle=False)

def train_local(client_model, data_loader, optimizer, criterion, epochs):
    client_model.train()
    for _ in range(epochs):
        for images, labels in data_loader:
            optimizer.zero_grad()
            outputs = client_model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

def aircomp_aggregate(weights, threshold, rho):
    avg_weights = {}
    num_clients = len(weights)
    h = np.random.rayleigh(scale=1.0, size=num_clients)

    for key in weights[0].keys():
        aggregated_value = 0.0
        active_count = 0
        for i in range(num_clients):
            hi = h[i]
            if h[i]**2 >= threshold:
                pk = np.sqrt(rho) / hi
                active_count += 1
            else:
                pk = 0.0
            aggregated_value += weights[i][key] * pk

        if active_count == 0:
            return None  # skip this round entirely

        noise = torch.normal(mean=0.0, std=noise_variance ** 0.5, size=aggregated_value.shape)
        aggregated_value += noise
        avg_weights[key] = aggregated_value / (active_count * np.sqrt(rho))

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

# Grid search
p0_values = [0.1, 0.2, 0.3, 0.4, 0.5]
threshold_values = [0.1, 0.2, 0.3, 0.4, 0.5]
epoch_values = [1, 2]

os.makedirs("plots", exist_ok=True)

for p0 in p0_values:
    for threshold in threshold_values:
        for epochs in epoch_values:
            print(f"\n=== Running P0={p0}, threshold={threshold}, epochs={epochs} ===")
            rho = p0 / (-expi(-threshold))
            global_model = CNN()
            criterion = nn.CrossEntropyLoss()
            accuracies = []

            skip = False
            for round in range(num_rounds):
                client_weights = []
                for i in range(num_clients):
                    local_model = CNN()
                    local_model.load_state_dict(global_model.state_dict())
                    optimizer = optim.SGD(local_model.parameters(), lr=learning_rate)
                    data_loader = data.DataLoader(client_data[i], batch_size=32, shuffle=True)
                    train_local(local_model, data_loader, optimizer, criterion, epochs)
                    client_weights.append(local_model.state_dict())

                global_weights = aircomp_aggregate(client_weights, threshold, rho)
                if global_weights is None:
                    print("No clients passed threshold. Skipping this round.")
                    continue

                global_model.load_state_dict(global_weights)
                accuracy = test_model(global_model, test_loader)
                print(f"Round {round+1}: Accuracy = {accuracy:.2f}%")
                accuracies.append(accuracy)

                if accuracy == 10.00:
                    print("Model collapsed. Skipping this combination.")
                    skip = True
                    break

            if not skip:
                plt.figure()
                plt.plot(range(1, len(accuracies)+1), accuracies, marker='o')
                plt.xlabel('Round')
                plt.ylabel('Accuracy (%)')
                plt.title(f'AirComp FL - P0={p0}, Threshold={threshold}, Epochs={epochs}')
                filename = f"plots/accuracy_plot_P0_{p0}_threshold_{threshold}_epochs_{epochs}.png"
                plt.grid()
                plt.savefig(filename)
                plt.close()
                print(f"Saved plot: {filename}")