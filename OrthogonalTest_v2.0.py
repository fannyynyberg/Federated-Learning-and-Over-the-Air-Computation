import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

# 1. Ladda MNIST-dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # Normalisera med medelvärde och standardavvikelse för MNIST
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# 2. Dela upp datasetet mellan klienter
num_clients = 10  # Antal klienter
client_data_size = len(train_dataset) // num_clients  # Data per klient

client_datasets = []
for i in range(num_clients):
    start_idx = i * client_data_size
    end_idx = (i + 1) * client_data_size
    client_datasets.append(Subset(train_dataset, range(start_idx, end_idx)))

# 3. Definiera en enkel neural nätverksmodell
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)  # Input layer
        self.fc2 = nn.Linear(128, 64)       # Hidden layer
        self.fc3 = nn.Linear(64, 10)        # Output layer (10 klasser)

    def forward(self, x):
        x = x.view(-1, 28 * 28)  # Flatten input
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 4. Funktion för att träna en klient och returnera dess vikter
def train_client(model, dataset, epochs=1, lr=0.01):
    criterion = nn.CrossEntropyLoss()  # Förlustfunktion
    optimizer = optim.SGD(model.parameters(), lr=lr)  # Optimizer
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)  # DataLoader

    model.train()  # Sätt modellen i träningsläge
    for epoch in range(epochs):
        for images, labels in dataloader:
            optimizer.zero_grad()  # Nollställ gradienter
            outputs = model(images)  # Förutsägelser
            loss = criterion(outputs, labels)  # Beräkna förlust
            loss.backward()  # Backpropagation
            optimizer.step()  # Uppdatera vikter

    return [param.data.numpy() for param in model.parameters()]  # Returnera vikterna som numpy-arrayer

# 5. Funktion för att aggregera vikterna
def aggregate_weights(client_weights):
    aggregated_weights = []
    for i in range(len(client_weights[0])):  # Iterera över varje lager
        layer_weights = np.stack([client_weight[i] for client_weight in client_weights])
        aggregated_weights.append(np.mean(layer_weights, axis=0))  # Beräkna medelvärdet för varje lager
    return aggregated_weights

# 6. Simulera federated learning
num_rounds = 10
global_model = SimpleNN()  # Skapa en global modell
global_weights_history = []  # För att spara historiken för globala vikter

for round in range(num_rounds):
    print(f"Runda {round + 1}")
    client_weights = []

    # Träna varje klient och spara dess vikter
    for i in range(num_clients):
        model = SimpleNN()  # Skapa en ny modell för varje klient
        model.load_state_dict(global_model.state_dict())  # Initiera med globala vikter
        weights = train_client(model, client_datasets[i], epochs=1)
        client_weights.append(weights)

    # Aggregera vikterna
    global_weights = aggregate_weights(client_weights)

    # Uppdatera den globala modellen med de aggregerade vikterna
    for param, new_weight in zip(global_model.parameters(), global_weights):
        param.data = torch.tensor(new_weight)

    # Spara den globala vikten för visualisering
    global_weights_history.append(global_weights)

# 7. Visualisera konvergensen av den globala vikten
plt.figure(figsize=(10, 6))
for i in range(len(global_weights_history[0])):
    layer_weights = [global_weights[i].mean() for global_weights in global_weights_history]
    plt.plot(range(1, num_rounds + 1), layer_weights, label=f'Layer {i+1}')
plt.title('Konvergens av den globala medelvikten')
plt.xlabel('Runda')
plt.ylabel('Global medelvikt')
plt.legend()
plt.grid()
plt.show()