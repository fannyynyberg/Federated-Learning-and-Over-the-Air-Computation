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

# Split dataset among multiple clients
num_clients = 3  # Number of clients in the simulation
client_datasets = np.array_split(train_dataset, num_clients)

# Initialize models and trainers for each client
clients = []
for i in range(num_clients):
    model = NeuralNetwork()
    trainer = Trainer(model, lr=0.01)
    train_loader = DataLoader(client_datasets[i], batch_size=32, shuffle=True)
    clients.append((trainer, train_loader))

# Initialize global model
num_rounds = 10
global_model = NeuralNetwork()
global_trainer = Trainer(global_model, lr=0.01)

# TDMA-based Federated Learning simulation
for round in range(num_rounds):
    print(f'Round {round + 1}')
    
    # Clients send updates sequentially in time slots (TDMA mechanism)
    for i, (trainer, train_loader) in enumerate(clients):
        print(f'Client {i+1} sending updates in time slot {i+1}')
        trainer.train(train_loader, epochs=1)
        client_weights = trainer.get_weights()
        global_trainer.set_weights(client_weights)  # Update global model after each client's transmission
    
print('TDMA-based Federated Learning simulation completed.')
