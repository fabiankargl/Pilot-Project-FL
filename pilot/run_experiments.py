import toml
import subprocess
import time

experiments = [
    {
        "strategy": "fedavg",
        "local-epochs": 1,
        "num-server-rounds": 5,
        "proximal-mu": 0.0,
        "description": "Baseline FedAvg (1 Epoch)"
    },
    {
        "strategy": "fedavg",
        "local-epochs": 5, 
        "num-server-rounds": 10, 
        "proximal-mu": 0.0,
        "description": "Stress Test FedAvg (5 Epochs)"
    },
    {
        "strategy": "fedprox",
        "local-epochs": 5,
        "num-server-rounds": 10,
        "proximal-mu": 0.1,
        "description": "Solution FedProx (mu=0.1)"
    },
    {
        "strategy": "fedprox",
        "local-epochs": 5,
        "num-server-rounds": 10,
        "proximal-mu": 1.0,
        "description": "Tuning FedProx (mu=1.0)"
    }
]

TOML_FILE = "pyproject.toml"

def update_toml_config(settings):
    """Reads the TOML, changes values, and saves it again."""
    try:
        with open(TOML_FILE, "r") as f:
            data = toml.load(f)
        
        if "tool" not in data or "flwr" not in data["tool"] or "app" not in data["tool"]["flwr"] or "config" not in data["tool"]["flwr"]["app"]:
            print("Error: pyproject.toml does not have the expected structure [tool.flwr.app.config]")
            return False

        config_section = data["tool"]["flwr"]["app"]["config"]

        config_section["strategy"] = settings["strategy"]
        config_section["local-epochs"] = settings["local-epochs"]
        config_section["num-server-rounds"] = settings["num-server-rounds"]
        config_section["proximal-mu"] = settings["proximal-mu"]

        with open(TOML_FILE, "w") as f:
            toml.dump(data, f)
            
        return True
    except Exception as e:
        print(f"Error updating the TOML: {e}")
        return False

def run_simulation():
    """Executes the Flower command."""
    print(">> Starting simulation...")
    try:
        subprocess.run(["flwr", "run", "."], check=True)
        print(">> Simulation finished successfully.\n")
    except subprocess.CalledProcessError as e:
        print(f"!! Simulation failed with error code {e.returncode} !!\n")
    except FileNotFoundError:
        print("!! Command 'flwr' not found. Is the virtual environment activated? !!\n")

def main():
    print(f"Starting {len(experiments)} experiments...\n")
    
    start_time = time.time()

    for i, exp in enumerate(experiments):
        print("="*60)
        print(f"EXPERIMENT {i + 1}/{len(experiments)}: {exp['description']}")
        print(f"Settings: {exp}")
        print("="*60)

        success = update_toml_config(exp)
        if not success:
            break
        
        run_simulation()
        
        time.sleep(2)

    duration = (time.time() - start_time) / 60
    print(f"All experiments completed in {duration:.1f} minutes.")

if __name__ == "__main__":
    try:
        import toml
        main()
    except ImportError:
        print("Error: The 'toml' module is missing.")
        print("Please install it with: pip install toml")