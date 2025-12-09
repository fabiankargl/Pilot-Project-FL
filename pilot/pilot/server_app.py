import torch
from flwr.app import ArrayRecord, ConfigRecord, Context
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg

from pilot.task import BankNet, load_data

app = ServerApp()

@app.main()
def main(grid: Grid, context: Context) -> None:
    # Read run config
    fraction_train: float = context.run_config["fraction-train"]
    fraction_evaluate: float = context.run_config["fraction-evaluate"]
    num_rounds: int = context.run_config["num-server-rounds"]
    lr: float = context.run_config["lr"]
    min_clients: int = context.run_config["min-available-clients"]
    
    trainloader, _ = load_data(partition_id=0,
                               num_partitions=0)
    sample_batch = next(iter(trainloader))
    input_dim = sample_batch[0].shape[1]

    # Load global model
    global_model = BankNet(input_dim=input_dim)
    arrays = ArrayRecord(global_model.state_dict())

    # Initialize FedAvg strategy
    strategy = FedAvg(fraction_train=fraction_train,
                      fraction_evaluate=fraction_evaluate,
                      min_train_nodes=min_clients,
                      min_evaluate_nodes=min_clients,
                      min_available_nodes=min_clients,
                      )

    # Start strategy, run FedAvg for `num_rounds`
    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=ConfigRecord({"lr": lr}),
        num_rounds=num_rounds,
    )

    # Save final model to disk
    print("\nSaving final model to disk...")
    state_dict = result.arrays.to_torch_state_dict()
    torch.save(state_dict, "final_model.pt")
