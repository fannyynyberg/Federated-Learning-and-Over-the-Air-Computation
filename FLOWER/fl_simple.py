import torch
import torch.nn as nn
import torch.optim as optim
import flwr as fl
import flwr.client
from collections import OrderedDict

# ------------------ 1. Skapa en enkel PyTorch-modell ------------------
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(2, 5)  # 2 input features -> 5 hidden
        self.fc2 = nn.Linear(5, 1)  # 5 hidden -> 1 output

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x

# ------------------ 2. Skapa träningsdata ------------------
def generate_dummy_data():
    X = torch.rand(100, 2)  # 100 datapunkter med 2 features
    y = (X[:, 0] + X[:, 1] > 1).float().unsqueeze(1)  # Enkel XOR-liknande regel
    return X, y

X_train, y_train = generate_dummy_data()

# ------------------ 3. Träningsfunktion ------------------
def train(model, X, y, epochs=3):
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    criterion = nn.BCELoss()  # Binary Cross Entropy (för 0/1-klasser)
    
    for _ in range(epochs):
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

# ------------------ 4. Skapa en FL-klient ------------------
#class FlowerClient(fl.client.NumPyClient):
class FlowerClient(flwr.client.NumPyClient):
    def __init__(self, model, X, y):
        self.model = model
        self.X = X
        self.y = y

    def get_parameters(self):
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        train(self.model, self.X, self.y, epochs=3)
        return self.get_parameters(), len(self.X), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss = nn.BCELoss()(self.model(self.X), self.y).item()
        return loss, len(self.X), {"loss": loss}

# ------------------ 5. Starta servern och klienterna ------------------
def main():
    # Skapa klienter
    client1 = FlowerClient(Net(), X_train[:50], y_train[:50])  # Första halvan av datan
    client2 = FlowerClient(Net(), X_train[50:], y_train[50:])  # Andra halvan

    # Starta FL-simulering
    fl.simulation.start_simulation(
        client_fn=lambda cid: client1 if cid == "0" else client2,
        num_clients=2,
        config=fl.server.ServerConfig(num_rounds=3),
    )

if __name__ == "__main__":
    main()
