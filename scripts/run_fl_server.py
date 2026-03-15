import flwr as fl

def main():
    print("Starting Flower server...")

    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        min_fit_clients=3,
        min_available_clients=3,
    )

    fl.server.start_server(
        server_address="0.0.0.0:8081",
        strategy=strategy,
        config=fl.server.ServerConfig(num_rounds=3),
    )

if __name__ == "__main__":
    main()