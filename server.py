import flwr as fl

def start_server():
    fl.server.start_server(
        config=fl.server.ServerConfig(num_rounds=3),
    )

if __name__ == "__main__":
    start_server()
