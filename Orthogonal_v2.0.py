"""Optimized Orthogonal simulation with calculated validation accuracy."""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from CNN import NeuralNetwork
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

# Enhet (GPU om tillgänglig)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Dataset & Transformering
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Laddar MNIST-datasetet
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# Skapa DataLoader för testning
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Split dataset among multiple clients (Minskat till 10 klienter)
num_clients = 10  
dataset_size = len(train_dataset)
indices = np.array_split(np.arange(dataset_size), num_clients)
client_datasets = [Subset(train_dataset, idx) for idx in indices]

# Neural Network Training Class
class Trainer:
    def __init__(self, model, lr=0.01):
        self.model = model.to(device)
        self.optimizer = optim.SGD(self.model.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss()

    def train(self, train_loader, epochs=2):  # Minskade epochs till 2
        self.model.train()
        for _ in range(epochs):
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

    def get_weights(self):
        return [param.data.clone() for param in self.model.parameters()]

    def set_weights(self, new_weights):
        for param, new_weight in zip(self.model.parameters(), new_weights):
            param.data = new_weight.clone()

# Initiera klienter med tränare
clients = []
for i in range(num_clients):
    model = NeuralNetwork()
    trainer = Trainer(model, lr=0.01)
    train_loader = DataLoader(client_datasets[i], batch_size=32, shuffle=True)  # Ökad batch size
    clients.append((trainer, train_loader))

# Initiera global modell
num_rounds = 50  # Minskat till 50 rounds
global_model = NeuralNetwork().to(device)
global_trainer = Trainer(global_model, lr=0.01)

# FedAvg viktaggregering
def aggregate_weights(client_trainers):
    aggregated_weights = []
    trainers = [trainer for trainer, _ in client_trainers]
    for i in range(len(trainers[0].get_weights())):
        layer_weights = torch.stack([trainer.get_weights()[i] for trainer in trainers]).to(device)
        aggregated_weights.append(torch.mean(layer_weights, dim=0))
    return aggregated_weights

# Evaluera modellen
def evaluate(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    return 100 * correct / total

# Kör Federated Learning
accuracies = []
for round in range(num_rounds):
    print(f'Round {round + 1}')

    # Train each client independently
    for trainer, train_loader in clients:
        trainer.train(train_loader, epochs=2)

    # Aggregate weights and update global model
    global_weights = aggregate_weights(clients)
    global_trainer.set_weights(global_weights)

    # Evaluate global model
    acc = evaluate(global_model, test_loader)
    accuracies.append(acc)
    print(f'Validation Accuracy: {acc:.2f}%')

# Visa träningsprogress
plt.plot(range(1, num_rounds + 1), accuracies, marker='o')
plt.xlabel('Federated Learning Rounds')
plt.ylabel('Validation Accuracy (%)')
plt.title('FedAvg Convergence')
plt.grid()
plt.show()

print('Optimized federated learning simulation with FedAvg completed.')
