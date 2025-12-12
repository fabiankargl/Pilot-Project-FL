import torch
import json
import os
from flwr.app import ArrayRecord, ConfigRecord, Context
from flwr.serverapp import Grid, ServerApp

from pilot.task import IncomeNet, LogisticRegression, load_data
from pilot.strategy import FedAvgWithHistory, FedProxWithHistory

app = ServerApp()

@app.main()
def main(grid: Grid, context: Context) -> None:
    # Read run config
    fraction_train: float = context.run_config["fraction-train"]
    fraction_evaluate: float = context.run_config["fraction-evaluate"]
    num_rounds: int = context.run_config["num-server-rounds"]
    lr: float = context.run_config["lr"]
    min_clients: int = context.run_config["min-available-clients"]
    local_epochs = context.run_config.get("local-epochs", 1)
    model_type = context.run_config.get("model-type", "nn")
    strategy_name = context.run_config.get("strategy", "fedavg").lower()
    proximal_mu = context.run_config.get("proximal-mu", 0.1)
    
    run_name = f"{strategy_name}_r{num_rounds}_e{local_epochs}_{model_type}_lr{lr}"
    
    if strategy_name == "fedprox":
        run_name += f"_mu{proximal_mu}"
    
    trainloader, _ = load_data(partition_id=0,
                               num_partitions=0)
    sample_batch = next(iter(trainloader))
    input_dim = sample_batch[0].shape[1]

    if model_type == "logreg":
        global_model = LogisticRegression(input_dim=input_dim)
    else:
        global_model = IncomeNet(input_dim=input_dim)
        
    arrays = ArrayRecord(global_model.state_dict())

    if strategy_name == "fedavg":
        strategy = FedAvgWithHistory(
            fraction_train=fraction_train,
            fraction_evaluate=fraction_evaluate,
            min_train_nodes=min_clients,
            min_evaluate_nodes=min_clients,
            min_available_nodes=min_clients,
        )
    elif strategy_name == "fedprox":
        strategy = FedProxWithHistory(
            fraction_train=fraction_train,
            fraction_evaluate=fraction_evaluate,
            min_train_nodes=min_clients,
            min_evaluate_nodes=min_clients,
            min_available_nodes=min_clients,
            proximal_mu=proximal_mu
        )
    else:
        raise ValueError(f"Unknown strategy: {strategy_name}")

    # Start strategy, run FedAvg for `num_rounds`
    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=ConfigRecord({"lr": lr}),
        num_rounds=num_rounds,
    )
    
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    
    global_hist_path = os.path.join(results_dir, f"global_history_{run_name}.json")
    local_hist_path = os.path.join(results_dir, f"local_history_{run_name}.json")

    with open(global_hist_path, "w") as f:
        json.dump(strategy.global_history, f, indent=2)
    
    with open(local_hist_path, "w") as f:
        json.dump(strategy.local_history, f, indent=2)

    # Save final model to disk
    print("\nSaving final model to disk...")
    state_dict = result.arrays.to_torch_state_dict()
    model_path = os.path.join(results_dir, f"final_model_{run_name}.pt")
    torch.save(state_dict, model_path)
    print(f"Final model saved to {model_path}")
    
    config_path = os.path.join(results_dir, f"config_{run_name}.json")
    with open(config_path, "w") as f:
        json.dump(context.run_config, f, indent=2)