"""Orthogonal simulation with calculated validation accuracy."""
import numpy as np
import torch
import matplotlib.pyplot as plt
# from CNN import NeuralNetwork, Trainer
from NN import NeuralNetwork, Trainer 
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

# Initialize dataset with normalization
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# Create DataLoader for test dataset
test_loader = DataLoader(test_dataset, batch_size=10, shuffle=False)

# Split dataset among multiple clients
num_clients = 3  # Number of clients in the simulation
dataset_size = len(train_dataset)
indices = np.array_split(np.arange(dataset_size), num_clients)
client_datasets = [Subset(train_dataset, idx) for idx in indices]

# Initialize models and trainers for each client
clients = []
for i in range(num_clients):
    model = NeuralNetwork(layer_1=128)
    trainer = Trainer(model, lr=0.01)
    train_loader = DataLoader(client_datasets[i], batch_size=10, shuffle=True)
    clients.append((trainer, train_loader))

# Initialize global model for Federated Learning simulation
num_rounds = 10 # Increased rounds
global_model = NeuralNetwork(layer_1=128)
global_trainer = Trainer(global_model, lr=0.01)

def aggregate_weights(client_trainers):
    aggregated_weights = []
    trainers = [trainer for trainer, _ in client_trainers]
    for i in range(len(trainers[0].get_weights())):
        layer_weights = torch.stack([trainer.get_weights()[i] for trainer in trainers])
        aggregated_weights.append(torch.mean(layer_weights, dim=0))
    return aggregated_weights

def evaluate(model, test_loader):
    """Evaluates the model on the test dataset."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    return 100 * correct / total

# Perform Federated Learning rounds
accuracies = []
for round in range(num_rounds):
    print(f'Round {round + 1}')
    
    # Train each client independently
    for trainer, train_loader in clients:
        trainer.train(train_loader, epochs=5)  # Increased epochs
    
    # Aggregate weights from clients and update global model
    global_weights = aggregate_weights(clients)
    global_trainer.set_weights(global_weights)
    
    # Evaluate global model
    acc = evaluate(global_model, test_loader)
    accuracies.append(acc)
    print(f'Validation Accuracy: {acc:.2f}%')

# Visualize training progress
plt.plot(range(1, num_rounds + 1), accuracies, marker='o')
plt.xlabel('Federated Learning Rounds')
plt.ylabel('Validation Accuracy (%)')
plt.title('FedAvg Convergence')
plt.grid()
plt.show()

print('Federated learning simulation with FedAvg completed.')
