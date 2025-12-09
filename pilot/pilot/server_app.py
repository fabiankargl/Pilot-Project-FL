import torch
import json
import os
from flwr.app import ArrayRecord, ConfigRecord, Context
from flwr.serverapp import Grid, ServerApp

from pilot.task import BankNet, load_data
from pilot.strategy import FedAvgWithHistory

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

    # Initialize Custom FedAvg strategy with history tracking
    strategy = FedAvgWithHistory(
        fraction_train=fraction_train,
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
    
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    
    with open(os.path.join(results_dir, "global_history.json"), "w") as f:
        json.dump(strategy.global_history, f, indent=2)
    print(f"Global history saved with {len(strategy.global_history['round'])} rounds")
    
    with open(os.path.join(results_dir, "local_history.json"), "w") as f:
        json.dump(strategy.local_history, f, indent=2)
    print(f"Local history saved with {len(strategy.local_history['round'])} rounds")

    # Save final model to disk
    print("\nSaving final model to disk...")
    state_dict = result.arrays.to_torch_state_dict()
    model_path = os.path.join(results_dir, "final_model.pt")
    torch.save(state_dict, model_path)
    print(f"Final model saved to {model_path}")