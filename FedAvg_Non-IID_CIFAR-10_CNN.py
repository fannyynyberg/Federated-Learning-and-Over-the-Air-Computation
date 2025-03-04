import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for plotting
import matplotlib.pyplot as plt
import time

from CNN import CNN

# Federated Learning setup
num_clients = 10
num_rounds = 10
epochs = 1  # Number of epochs per client per round
learning_rate = 0.01

# Transformation pipeline for CIFAR-10
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load CIFAR-10 dataset
cifar10_data = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)

def split_cifar10_non_iid(dataset, num_clients, num_shards=40):
    """
    Creates a non-IID partition of CIFAR-10 data.
    Each client will receive only a subset of labels.
    """
    num_samples = len(dataset)
    data_indices = np.arange(num_samples)
    labels = np.array(dataset.targets)
    
    # Sort by label to make it easier to split by class
    sorted_indices = data_indices[np.argsort(labels)]
    
    # Split into shards
    shards = np.array_split(sorted_indices, num_shards)
    np.random.shuffle(shards)
    
    # Assign shards to clients (each client gets num_shards // num_clients shards)
    client_data = {i: np.concatenate(shards[i::num_clients]) for i in range(num_clients)}
    
    return client_data

# Create non-IID client datasets
client_indices = split_cifar10_non_iid(cifar10_data, num_clients)
client_data = {i: torch.utils.data.Subset(cifar10_data, client_indices[i]) for i in range(num_clients)}

# Load CIFAR-10 test dataset
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
cifar10_test = datasets.CIFAR10(root="./data", train=False, download=True, transform=test_transform)
test_loader = data.DataLoader(cifar10_test, batch_size=64, shuffle=False)

# Function to train a local client model
def train_local(client_model, data_loader, optimizer, criterion):
    client_model.train()
    for images, labels in data_loader:
        optimizer.zero_grad()
        outputs = client_model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

# Federated averaging function
def federated_avg(weights):
    avg_weights = {}
    for key in weights[0].keys():
        avg_weights[key] = sum(w[key] for w in weights) / len(weights)
    return avg_weights

# Function to test the global model on the test dataset
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

# Initialize the global model
global_model = CNN()
criterion = nn.CrossEntropyLoss()
accuracies = []
start_time = time.time()

# Federated training loop
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
    
    # Perform federated averaging
    global_weights = federated_avg(client_weights)
    global_model.load_state_dict(global_weights)
    
    # Test the updated global model
    accuracy = test_model(global_model, test_loader)
    accuracies.append(accuracy)
    round_time = time.time() - round_start
    print(f"Round {round+1} completed - Accuracy: {accuracy:.2f}% - Time: {round_time:.2f} sec")

# Plot accuracy over rounds
plt.figure()
plt.plot(range(1, num_rounds + 1), accuracies, marker='o')
plt.xlabel('Round')
plt.ylabel('Accuracy (%)')
plt.title('Federated Learning on CIFAR-10 (Non-IID)')
plt.grid()
plt.savefig("federated_learning_cifar10_non_iid.png")
print("Plot saved as federated_learning_cifar10_non_iid.png")

total_time = time.time() - start_time
print(f"Total training time: {total_time:.2f} seconds")
print(f"Average time per round: {total_time / num_rounds:.2f} seconds")
