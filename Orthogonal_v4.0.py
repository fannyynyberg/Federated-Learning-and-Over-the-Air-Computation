import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for plotting
import matplotlib.pyplot as plt
import time

from CNN2 import CNN

# Federated Learning setup
num_clients = 10
num_rounds = 100
epochs = 1  # Number of epochs per client per round (not explicitly used below, but can be integrated)
learning_rate = 0.01

# Transformation pipeline for CIFAR-10
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load CIFAR-10 dataset (training)
#cifar10_train = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
# Split the training data among clients
#client_data = torch.utils.data.random_split(cifar10_train, [len(cifar10_train) // num_clients] * num_clients)

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

# Federated averaging function to average model weights
def federated_avg(weights):
    avg_weights = {}
    # Assume all clients have the same keys in state_dict
    for key in weights[0].keys():
        avg_weights[key] = sum(w[key] for w in weights) / len(weights)
    return avg_weights

# Function to test the global model on the test dataset
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

# Initialize the global model
global_model = CNN()
criterion = nn.CrossEntropyLoss()

# List to store test accuracies over rounds
accuracies = []

# Record the start time
start_time = time.time()

# Federated training loop
for round in range(num_rounds):
    round_start = time.time()
    client_weights = []
    
    # Train each client locally
    for i in range(num_clients):
        # Initialize local model and load global weights
        local_model = CNN()
        local_model.load_state_dict(global_model.state_dict())
        optimizer = optim.SGD(local_model.parameters(), lr=learning_rate)
        data_loader = data.DataLoader(client_data[i], batch_size=64, shuffle=True)
        
        # Train for the defined number of epochs (if using epochs > 1)
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
plt.title('Federated Learning Convergence on CIFAR-10')
plt.grid()
plt.savefig("federated_learning_convergence.png")
print("Plot saved as federated_learning_convergence.png")

total_time = time.time() - start_time
avg_time_per_round = total_time / num_rounds
print(f"Total training time: {total_time:.2f} seconds")
print(f"Average time per round: {avg_time_per_round:.2f} seconds")
