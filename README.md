# Pilot-Project-FL

This repository contains the code for a federated learning pilot project using the Flower framework.

## Project Structure

The project is organized into several key directories, each with a specific purpose.

### `/data/`

This directory at the root of the `/pilot/` folder contains the raw `.csv` datasets for each participating bank:

- `BankA.csv`
- `BankB.csv`
- `BankC.csv`

### Data Analysis (`/bank_A_analysis/`, `/bank_B_analysis/`, `/bank_C_analysis/`)

These folders contain preliminary, centralized data analysis for individual bank datasets to understand their characteristics and establish baselines.

### `/pilot/` (Federated Learning Core)

This is the main directory for the Flower-based federated learning application.

- **`/pilot/pilot/`**: The core application package.
  - `client_app.py`: Defines the Flower `ClientApp`, which connects a client's data and model training logic from `task.py` to the simulation.
  - `server_app.py`: Defines the Flower `ServerApp`, which configures the server-side strategy and orchestrates the federated learning process.
  - `task.py`: Defines the PyTorch models, data loading logic for each bank, and the client-side `train`/`test` functions.
  - `strategy.py`: Implements custom Flower strategies (`FedAvgWithHistory`, `FedProxWithHistory`) that record detailed global and local metrics during training.
- **`run_experiments.py`**: A utility script to automate running a series of predefined federated learning experiments. It dynamically updates the `pyproject.toml` configuration for each run.
- **`plot_results.py`**: A script to parse the JSON output from the experiment runs and generate detailed and comparative plots of all results.
- **`/results/`**: The default output directory where metrics and history from the federated learning runs are saved as `.json` files.

### Root Directory & Final Plotting

- **`/final_results/`**: Contains a curated selection of the most important experiment results (`.json` files), copied from `/pilot/results/`. This folder holds the data used to generate the final, presentation-ready plots.
- **`plot_results.py`**: A top-level script that reads from `/final_results/` to generate the final comparison plots for the project summary and presentation.
