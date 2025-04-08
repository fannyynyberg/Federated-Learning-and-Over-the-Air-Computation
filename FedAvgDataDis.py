import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
import matplotlib.pyplot as plt
from MLP import NeuralNetwork

# Inställningar
num_clients = 20
num_rounds = 100
epochs = 1
learning_rate = 0.01
transform = transforms.Compose([transforms.ToTensor()])
test_loader = data.DataLoader(
    datasets.MNIST(root="./data", train=False, download=True, transform=transform),
    batch_size=64,
    shuffle=False
)

# === Hjälpfunktioner ===
def train_local(model, data_loader, optimizer, criterion, flatten=False):
    model.train()
    for images, labels in data_loader:
        optimizer.zero_grad()
        if flatten:
            images = images.view(-1, 28*28)
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

def test_model(model, test_loader, flatten=False):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            if flatten:
                images = images.view(-1, 28*28)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    return 100 * correct / total

def federated_avg(weights):
    avg_weights = {}
    for key in weights[0].keys():
        avg_weights[key] = sum(w[key] for w in weights) / len(weights)
    return avg_weights

def split_non_iid(dataset, num_clients, num_shards=40):
    labels = np.array(dataset.targets)
    indices = np.arange(len(dataset))
    sorted_indices = indices[np.argsort(labels)]
    shards = np.array_split(sorted_indices, num_shards)
    np.random.shuffle(shards)
    client_indices = {i: np.concatenate(shards[i::num_clients]) for i in range(num_clients)}
    return {i: data.Subset(dataset, client_indices[i]) for i in range(num_clients)}

# === Körning för IID ===
def run_fedavg_iid():
    mnist_data = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    client_data = torch.utils.data.random_split(mnist_data, [len(mnist_data)//num_clients]*num_clients)
    global_model = NeuralNetwork()
    criterion = nn.CrossEntropyLoss()
    accuracies = []

    for _ in range(num_rounds):
        client_weights = []
        for i in range(num_clients):
            local_model = NeuralNetwork()
            local_model.load_state_dict(global_model.state_dict())
            optimizer = optim.SGD(local_model.parameters(), lr=learning_rate)
            loader = data.DataLoader(client_data[i], batch_size=32, shuffle=True)
            train_local(local_model, loader, optimizer, criterion)
            client_weights.append(local_model.state_dict())
        global_model.load_state_dict(federated_avg(client_weights))
        acc = test_model(global_model, test_loader)
        accuracies.append(acc)
    return accuracies

# === Körning för Non-IID ===
def run_fedavg_non_iid():
    mnist_data = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    client_data = split_non_iid(mnist_data, num_clients)
    global_model = NeuralNetwork()
    criterion = nn.CrossEntropyLoss()
    accuracies = []

    for _ in range(num_rounds):
        client_weights = []
        for i in range(num_clients):
            local_model = NeuralNetwork()
            local_model.load_state_dict(global_model.state_dict())
            optimizer = optim.SGD(local_model.parameters(), lr=learning_rate)
            loader = data.DataLoader(client_data[i], batch_size=32, shuffle=True)
            train_local(local_model, loader, optimizer, criterion, flatten=True)
            client_weights.append(local_model.state_dict())
        global_model.load_state_dict(federated_avg(client_weights))
        acc = test_model(global_model, test_loader, flatten=True)
        accuracies.append(acc)
    return accuracies

# === Huvudprogram ===
iid_accuracies = run_fedavg_iid()
non_iid_accuracies = run_fedavg_non_iid()

# === Plot ===
plt.figure()
plt.plot(range(1, num_rounds+1), iid_accuracies, label='IID', marker='o', markersize=3)
plt.plot(range(1, num_rounds+1), non_iid_accuracies, label='Non-IID', marker='x', markersize=3)
plt.xlabel('Round')
plt.ylabel('Accuracy (%)')
plt.title('FedAvg: IID vs Non-IID on MNIST')
plt.legend()
plt.grid()
plt.savefig("fedavg_iid_vs_non_iid.png")
print("Plot saved as fedavg_iid_vs_non_iid.png")
