from collections import OrderedDict
import torch
import flwr as fl
from model import Net
from train import train, test

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, model, trainloader, valloader):
        self.model = model
        self.trainloader = trainloader
        self.valloader = valloader

    def get_parameters(self):
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        train(self.model, self.trainloader, epochs=1)
        return self.get_parameters(), len(self.trainloader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        accuracy = test(self.model, self.valloader)
        return 0.0, len(self.valloader.dataset), {"accuracy": accuracy}

def create_client_fn(trainloaders, valloaders):
    def client_fn(cid):
        return FlowerClient(Net(), trainloaders[int(cid)], valloaders[int(cid)])
    return client_fn
