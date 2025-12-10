import json
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import os
import glob

plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.dpi'] = 100
RESULTS_DIR = "results"

def load_strategy_data(strategy_name):
    """Loads global and local history for a specific strategy."""
    global_file = os.path.join(RESULTS_DIR, f"global_history_{strategy_name}.json")
    local_file = os.path.join(RESULTS_DIR, f"local_history_{strategy_name}.json")
    
    if not os.path.exists(global_file) or not os.path.exists(local_file):
        print(f"Warning: Files for strategy '{strategy_name}' not found.")
        return None, None
        
    with open(global_file, "r") as f:
        global_hist = json.load(f)
    with open(local_file, "r") as f:
        local_hist = json.load(f)
        
    return global_hist, local_hist

def extract_client_data(local_history, metric_key):
    """Helper function to structure client data cleanly."""
    client_data = {}
    for round_idx, round_data in enumerate(local_history["client_results"]):
        round_num = local_history["round"][round_idx]
        for client in round_data["eval"]:
            client_id = client.get("client_id", -1)
            if "partition_id" in client:
                client_id = client["partition_id"]
                
            metric_value = client.get(metric_key, 0)
            
            if client_id not in client_data:
                client_data[client_id] = {"rounds": [], "values": []}
            
            client_data[client_id]["rounds"].append(round_num)
            client_data[client_id]["values"].append(metric_value)
    return client_data

def plot_single_strategy(strategy_name, global_history, local_history):
    """Creates the 4-panel plot for ONE strategy (Global vs Clients)."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.patch.set_facecolor('white')
    
    rounds = global_history["round"]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c'] 
    client_names = ['Bank A', 'Bank B', 'Bank C']
    
    metrics = [
        ('eval_loss', 'Evaluation Loss', axes[0, 0], 'black'),
        ('eval_acc', 'Accuracy', axes[0, 1], 'darkgreen'),
        ('eval_f1', 'F1 Score (Macro)', axes[1, 0], 'darkgreen'),
        ('eval_auc', 'AUC-ROC', axes[1, 1], 'darkgreen')
    ]
    
    for metric_key, metric_name, ax, global_color in metrics:
        ax.set_facecolor('white')
        ax.grid(True, alpha=0.4, linestyle='-', linewidth=0.8, color='gray', zorder=0)
        
        client_data = extract_client_data(local_history, metric_key)
        for client_id in sorted(client_data.keys()):
            data = client_data[client_id]
            c_idx = int(client_id) if int(client_id) < len(colors) else 0
            label = client_names[c_idx] if int(client_id) < len(client_names) else f"Client {client_id}"
            
            ax.plot(data["rounds"], data["values"], 
                    marker='o', linestyle='--', linewidth=1.5, markersize=6,
                    alpha=0.6, color=colors[c_idx], label=label, zorder=5)
        
        ax.plot(rounds, global_history[metric_key], 
                marker='D', linestyle='-', linewidth=3.0, markersize=8,
                label='Global Model', color=global_color, zorder=10)
        
        ax.set_xlabel('Round', fontweight='bold')
        ax.set_ylabel(metric_name, fontweight='bold')
        ax.set_title(f'{metric_name}', fontweight='bold', fontsize=12)
        ax.legend(loc='best', fontsize=9, framealpha=0.9)
        ax.set_xticks(rounds)
    
    plt.suptitle(f'Federated Learning Analysis: {strategy_name.upper()}', fontsize=16, fontweight='bold', y=0.99)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f'plot_{strategy_name}.png'), dpi=150)
    plt.close()
    print(f"Plot saved: plot_{strategy_name}.png")

def plot_strategy_comparison(strategies_data):
    """Compares global models and shows the BEST value in the legend."""
    if len(strategies_data) < 2:
        print("Not enough strategies for a comparison plot.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.patch.set_facecolor('#f5f5f5') 
    
    metrics_config = [
        ('eval_loss', 'Evaluation Loss', axes[0, 0], 'min'), 
        ('eval_acc', 'Accuracy', axes[0, 1], 'max'),        
        ('eval_f1', 'F1 Score (Macro)', axes[1, 0], 'max'),
        ('eval_auc', 'AUC-ROC', axes[1, 1], 'max')
    ]
    
    styles = {
        'fedavg': {'color': '#1f77b4', 'marker': 'o'},
        'fedprox': {'color': '#d62728', 'marker': 'D'} 
    }
    
    for metric_key, title, ax, mode in metrics_config:
        # Plot Style
        ax.set_facecolor('white')
        ax.grid(True, which='major', linestyle='--', linewidth=0.7, color='gray', alpha=0.5, zorder=0)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        
        for name, (glob_hist, _) in strategies_data.items():
            rounds = glob_hist["round"]
            values = glob_hist[metric_key]
            
            style = styles.get(name, {'color': 'gray', 'marker': 'x'})
            
            if mode == 'min':
                best_val = min(values)
            else:
                best_val = max(values)
            
            label_text = f"{name.upper()} (Best: {best_val:.4f})"
            
            ax.plot(rounds, values, 
                    marker=style['marker'], 
                    linewidth=2.5, 
                    markersize=8,
                    label=label_text,
                    color=style['color'],
                    alpha=0.8,
                    zorder=10)

        ax.set_title(title, fontweight='bold', fontsize=12)
        ax.set_xlabel("Round", fontweight='bold')

        ax.legend(loc='best', frameon=True, shadow=True, fancybox=True, fontsize=10, title="Strategy Performance")
        
    plt.suptitle('Battle of Algorithms: Standard FedAvg vs. FedProx', fontsize=18, fontweight='bold', y=0.99)
    plt.tight_layout()
    
    save_path = os.path.join("results", 'plot_comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Final comparison plot saved: {save_path}")

def main():
    files = glob.glob(os.path.join(RESULTS_DIR, "global_history_*.json"))
    strategy_names = [os.path.basename(f).replace("global_history_", "").replace(".json", "") for f in files]
    
    if not strategy_names:
        print("No result files found!")
        return

    print(f"Found strategies: {strategy_names}")
    
    all_data = {}

    for name in strategy_names:
        g_hist, l_hist = load_strategy_data(name)
        if g_hist and l_hist:
            all_data[name] = (g_hist, l_hist)
            plot_single_strategy(name, g_hist, l_hist)
    
    plot_strategy_comparison(all_data)

if __name__ == "__main__":
    main()