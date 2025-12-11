import toml
import subprocess
import time

experiments = [
    {
        "strategy": "fedavg", 
        "model-type": "logreg", 
        "local-epochs": 1, 
        "num-server-rounds": 10, 
        "lr": 0.01, 
        "proximal-mu": 0.0,
        "description": "LogReg Benchmark (10R, 1E, LR 0.01)"
    },
    {
        "strategy": "fedavg", 
        "model-type": "nn", 
        "local-epochs": 1, 
        "num-server-rounds": 10, 
        "lr": 0.001, 
        "proximal-mu": 0.0,
        "description": "NN Baseline (10R, 1E, LR 0.001)"
    },
    {
        "strategy": "fedavg", 
        "model-type": "nn", 
        "local-epochs": 3, 
        "num-server-rounds": 10, 
        "lr": 0.001, 
        "proximal-mu": 0.0,
        "description": "NN Drift Test (10R, 3E, LR 0.001)"
    },
    {
        "strategy": "fedprox", 
        "model-type": "nn", 
        "local-epochs": 3, 
        "num-server-rounds": 10, 
        "lr": 0.001, 
        "proximal-mu": 0.1, 
        "description": "NN FedProx Stable (10R, 3E, LR 0.001)"
    },
    {
        "strategy": "fedavg", 
        "model-type": "nn", 
        "local-epochs": 3, 
        "num-server-rounds": 10, 
        "lr": 0.005, 
        "proximal-mu": 0.0,
        "description": "NN Drift Extreme (10R, 3E, LR 0.005)"
    },
    {
        "strategy": "fedprox", 
        "model-type": "nn", 
        "local-epochs": 3, 
        "num-server-rounds": 10, 
        "lr": 0.005, 
        "proximal-mu": 0.1, 
        "description": "NN FedProx Speed (10R, 3E, LR 0.005)"
    }
]

TOML_FILE = "pyproject.toml"

def update_toml_config(settings):
    try:
        with open(TOML_FILE, "r") as f:
            data = toml.load(f)

        if "tool" not in data or "flwr" not in data["tool"] or "app" not in data["tool"]["flwr"] or "config" not in data["tool"]["flwr"]["app"]:
            print("Error: pyproject.toml structure is not correct.")
            return False

        config_section = data["tool"]["flwr"]["app"]["config"]

        config_section["strategy"] = settings["strategy"]
        config_section["local-epochs"] = settings["local-epochs"]
        config_section["num-server-rounds"] = settings["num-server-rounds"]
        config_section["proximal-mu"] = settings["proximal-mu"]
        config_section["model-type"] = settings["model-type"]
        config_section["lr"] = settings["lr"]

        with open(TOML_FILE, "w") as f:
            toml.dump(data, f)
            
        return True
    except Exception as e:
        print(f"Error updating TOML file: {e}")
        return False

def run_simulation():
    print(">> Starting simulation...")
    try:
        subprocess.run(["flwr", "run", "."], check=True)
        print(">> Simulation successful.\n")
    except subprocess.CalledProcessError as e:
        print(f"!! Simulation failed (Code {e.returncode}) !!\n")
    except FileNotFoundError:
        print("!! Command 'flwr' not found. Is the virtual environment activated? !!\n")

def main():
    print("="*60)
    print(f"STARTING 10-ROUND EXPERIMENTS ({len(experiments)} runs)")
    print("="*60)
    
    start_time = time.time()

    for i, exp in enumerate(experiments):
        print(f"Run {i + 1}/{len(experiments)}: {exp['description']}")
        
        success = update_toml_config(exp)
        if not success:
            break
        
        run_simulation()
        time.sleep(2)

    duration = (time.time() - start_time) / 60
    print("="*60)
    print(f"ALL EXPERIMENTS FINISHED IN {duration:.1f} MINUTES.")
    print("="*60)

if __name__ == "__main__":
    try:
        import toml
        main()
    except ImportError:
        print("Error: 'toml' module is missing. -> pip install toml")