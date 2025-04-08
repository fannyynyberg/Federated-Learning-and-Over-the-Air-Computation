import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expi
from MLP import NeuralNetwork

# === Allmänna parametrar ===
num_clients = 20
num_rounds = 100
epochs = 1
learning_rate = 0.01
threshold = 0.1
P0 = 0.2
rho = P0 / (-expi(-threshold))
noise_variance = 0.001
transform = transforms.Compose([transforms.ToTensor()])
test_loader = data.DataLoader(
    datasets.MNIST(root="./data", train=False, download=True, transform=transform),
    batch_size=64,
    shuffle=False
)

# === Hjälpfunktioner ===
def train_local(model, loader, optimizer, criterion):
    model.train()
    for images, labels in loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

def test_model(model):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

def aircomp_aggregate(weights):
    avg_weights = {}
    h = np.random.rayleigh(scale=1.0, size=len(weights))
    h_sq = h ** 2
    active = [i for i in range(len(weights)) if h_sq[i] >= threshold]
    A = len(active)
    if A == 0:
        raise RuntimeError("No clients active. Lower the threshold.")
    print(f"AirComp aggregation: {A}/{len(weights)} clients active")
    for key in weights[0].keys():
        agg = sum(weights[i][key] * (np.sqrt(rho) / h[i]) for i in active)
        noise = torch.normal(mean=0.0, std=noise_variance ** 0.5, size=agg.shape)
        agg += noise
        avg_weights[key] = agg / (A * np.sqrt(rho))
    return avg_weights

def split_non_iid(dataset, num_clients, num_shards=60):
    indices = np.arange(len(dataset))
    labels = np.array(dataset.targets)
    sorted_indices = indices[np.argsort(labels)]
    shards = np.array_split(sorted_indices, num_shards)
    np.random.shuffle(shards)
    client_indices = {i: np.concatenate(shards[i::num_clients]) for i in range(num_clients)}
    return {i: data.Subset(dataset, client_indices[i]) for i in range(num_clients)}

# === Körning för AirComp med IID ===
def run_aircomp_iid():
    print("\n=== Running AirComp IID ===")
    mnist_data = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    client_data = torch.utils.data.random_split(mnist_data, [len(mnist_data)//num_clients]*num_clients)
    global_model = NeuralNetwork()
    criterion = nn.CrossEntropyLoss()
    acc = []
    for rnd in range(num_rounds):
        client_weights = []
        for i in range(num_clients):
            model = NeuralNetwork()
            model.load_state_dict(global_model.state_dict())
            optimizer = optim.SGD(model.parameters(), lr=learning_rate)
            loader = data.DataLoader(client_data[i], batch_size=32, shuffle=True)
            train_local(model, loader, optimizer, criterion)
            client_weights.append(model.state_dict())
        global_model.load_state_dict(aircomp_aggregate(client_weights))
        round_acc = test_model(global_model)
        acc.append(round_acc)
        print(f"Round {rnd+1} completed – Accuracy: {round_acc:.2f}%")
    return acc

# === Körning för AirComp med Non-IID ===
def run_aircomp_non_iid():
    print("\n=== Running AirComp Non-IID ===")
    mnist_data = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    client_data = split_non_iid(mnist_data, num_clients)
    global_model = NeuralNetwork()
    criterion = nn.CrossEntropyLoss()
    acc = []
    for rnd in range(num_rounds):
        client_weights = []
        for i in range(num_clients):
            model = NeuralNetwork()
            model.load_state_dict(global_model.state_dict())
            optimizer = optim.SGD(model.parameters(), lr=learning_rate)
            loader = data.DataLoader(client_data[i], batch_size=32, shuffle=True)
            train_local(model, loader, optimizer, criterion)
            client_weights.append(model.state_dict())
        global_model.load_state_dict(aircomp_aggregate(client_weights))
        round_acc = test_model(global_model)
        acc.append(round_acc)
        print(f"Round {rnd+1} completed – Accuracy: {round_acc:.2f}%")
    return acc

# === Huvudprogram ===
iid_accuracies = run_aircomp_iid()
non_iid_accuracies = run_aircomp_non_iid()

# === Plot ===
plt.figure()
plt.plot(range(1, num_rounds+1), iid_accuracies, label='AirComp IID', marker='o', markersize=3)
plt.plot(range(1, num_rounds+1), non_iid_accuracies, label='AirComp Non-IID', marker='x', markersize=3)
plt.xlabel('Rounds')
plt.ylabel('Accuracy (%)')
plt.title('AirComp convergence on IID and Non-IID')
plt.legend()
plt.grid()
plt.savefig("aircomp_iid_vs_non_iid.png")
print("Plot saved as aircomp_iid_vs_non_iid.png")
