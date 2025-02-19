"""Functioning simple implementation of the orthogonal case with FedAvg."""
import numpy as np
import torch
from CNN import NeuralNetwork, Trainer
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

# Initialize dataset with normalization
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

# Split dataset among multiple clients using indices
num_clients = 3
dataset_size = len(train_dataset)
indices = np.array_split(np.arange(dataset_size), num_clients)

client_datasets = [Subset(train_dataset, idx) for idx in indices]  # Correct way to split

# Initialize models and trainers for each client
clients = []
for i in range(num_clients):
    model = NeuralNetwork()
    trainer = Trainer(model, lr=0.01)
    train_loader = DataLoader(client_datasets[i], batch_size=32, shuffle=True)
    clients.append((trainer, train_loader))

# Initialize global model for Federated Learning simulation
num_rounds = 10
global_model = NeuralNetwork()
global_trainer = Trainer(global_model, lr=0.01)

# Function to aggregate weights from clients
# Uses simple averaging to combine model parameters across clients
def aggregate_weights(client_trainers):
    aggregated_weights = []
    
    # Extract trainers from tuples
    trainers = [trainer for trainer, _ in client_trainers]
    
    for i in range(len(trainers[0].get_weights())):
        layer_weights = torch.stack([trainer.get_weights()[i] for trainer in trainers])
        aggregated_weights.append(torch.mean(layer_weights, dim=0))
    
    return aggregated_weights

# Perform Federated Learning rounds
for round in range(num_rounds):
    print(f'Round {round + 1}')
    
    # Train each client independently
    for trainer, train_loader in clients:
        trainer.train(train_loader, epochs=1)
    
    # Aggregate weights from clients and update global model
    global_weights = aggregate_weights(clients)
    global_trainer.set_weights(global_weights)

print('Federated learning simulation with orthogonal case completed.')
