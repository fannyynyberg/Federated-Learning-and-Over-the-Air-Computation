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

from CNN import CNN  # Your CNN model

# Config
num_clients = 10
num_rounds = 100
epochs = 2
learning_rate = 0.01
noise_variance = 0.0001

# Transforms
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load CIFAR-10 training set
cifar10_data = datasets.CIFAR10(root="./data", train=True, download=True, transform=train_transform)

# ----------- 🔀 Non-IID Partitioning Function ----------- #
def split_cifar10_non_iid(dataset, num_clients, num_shards=40):
    num_samples = len(dataset)
    data_indices = np.arange(num_samples)
    labels = np.array(dataset.targets)

    sorted_indices = data_indices[np.argsort(labels)]
    shards = np.array_split(sorted_indices, num_shards)
    np.random.shuffle(shards)

    client_data = {i: np.concatenate(shards[i::num_clients]) for i in range(num_clients)}
    return client_data

# Non-IID partitioning
client_indices = split_cifar10_non_iid(cifar10_data, num_clients)
client_data = {i: torch.utils.data.Subset(cifar10_data, client_indices[i]) for i in range(num_clients)}

# Load CIFAR-10 test set
cifar10_test = datasets.CIFAR10(root="./data", train=False, download=True, transform=test_transform)
test_loader = data.DataLoader(cifar10_test, batch_size=64, shuffle=False)

# ----------- 🔁 Training & Aggregation Functions ----------- #
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
    h = np.clip(h, a_min=0.5, a_max=None)

    for key in weights[0].keys():
        aggregated_value = sum((w[key] / torch.tensor(h[i], dtype=torch.float32)) for i, w in enumerate(weights))
        noise = torch.normal(mean=0.0, std=noise_variance ** 0.5, size=aggregated_value.shape)
        avg_weights[key] = aggregated_value / num_clients + noise
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

# ----------- 🚀 Federated AirComp Training Loop ----------- #
global_model = CNN()
criterion = nn.CrossEntropyLoss()
accuracies = []
start_time = time.time()

for round in range(num_rounds):
    round_start = time.time()
    client_weights = []

    for i in range(num_clients):
        local_model = CNN()
        local_model.load_state_dict(global_model.state_dict())
        optimizer = optim.SGD(local_model.parameters(), lr=learning_rate)
        data_loader = data.DataLoader(client_data[i], batch_size=64, shuffle=True)

        for _ in range(epochs):
            train_local(local_model, data_loader, optimizer, criterion)

        client_weights.append(local_model.state_dict())

    # AirComp Aggregation
    global_weights = aircomp_aggregate(client_weights)
    global_model.load_state_dict(global_weights)

    # Test and track
    accuracy = test_model(global_model, test_loader)
    accuracies.append(accuracy)
    round_time = time.time() - round_start
    print(f"Round {round+1} completed - Accuracy: {accuracy:.2f}% - Time: {round_time:.2f} sec")

# ----------- 📈 Plot Results ----------- #
plt.figure()
plt.plot(range(1, num_rounds + 1), accuracies, marker='o')
plt.xlabel('Round')
plt.ylabel('Accuracy (%)')
plt.title('AirComp on CIFAR-10 (Non-IID)')
plt.grid()
plt.savefig("aircomp_cifar10_non_iid.png")
print("Plot saved as aircomp_cifar10_non_iid.png")

total_time = time.time() - start_time
print(f"Total training time: {total_time:.2f} seconds")
