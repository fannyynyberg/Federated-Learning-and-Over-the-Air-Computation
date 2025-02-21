from flwr.simulation import start_simulation
from dataset import load_datasets
from client import create_client_fn
import flwr as fl

if __name__ == "__main__":
    trainloaders, valloaders, testloader = load_datasets()
    client_fn = create_client_fn(trainloaders, valloaders)

    start_simulation(
        client_fn=client_fn,
        num_clients=10,
        config=fl.server.ServerConfig(num_rounds=3),
    )
