import json
import matplotlib.pyplot as plt
import os

plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.dpi'] = 100

def load_histories():
    results_dir = "results"
    
    with open(os.path.join(results_dir, "global_history.json"), "r") as f:
        global_history = json.load(f)
    
    with open(os.path.join(results_dir, "local_history.json"), "r") as f:
        local_history = json.load(f)
    
    return global_history, local_history

def extract_client_data(local_history, metric_key):
    client_data = {}
    
    for round_idx, round_data in enumerate(local_history["client_results"]):
        round_num = local_history["round"][round_idx]
        
        for client in round_data["eval"]:
            client_id = client["client_id"]
            metric_value = client[metric_key]
            
            if client_id not in client_data:
                client_data[client_id] = {"rounds": [], "values": []}
            
            client_data[client_id]["rounds"].append(round_num)
            client_data[client_id]["values"].append(metric_value)
    
    return client_data

def plot_all_metrics(global_history, local_history):
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
            ax.plot(data["rounds"], data["values"], 
                    marker='o', linestyle='--', linewidth=2.5, markersize=7,
                    alpha=0.75, color=colors[client_id],
                    label=client_names[client_id], zorder=5)
        
        ax.plot(rounds, global_history[metric_key], 
                marker='D', linestyle='-', linewidth=3.5, markersize=9,
                label='Global Model', color=global_color, zorder=10)
        
        ax.set_xlabel('Round', fontweight='bold')
        ax.set_ylabel(metric_name, fontweight='bold')
        ax.set_title(f'{metric_name}', fontweight='bold', fontsize=13)
        ax.legend(loc='best', frameon=True, shadow=True, fancybox=True, fontsize=10)
        ax.set_xticks(rounds)
        
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)
            spine.set_color('black')
    
    plt.suptitle('Federated Learning: Performance Metrics (Global Model vs Local Clients)', 
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    results_dir = "results"
    plt.savefig(os.path.join(results_dir, 'federated_metrics.png'), dpi=300, bbox_inches='tight', facecolor='white')
    
    return fig


def main():
    print("Loading history files...")
    global_history, local_history = load_histories()
    print("Data loaded successfully")
    print("\nCreating plot...")
    plot_all_metrics(global_history, local_history)
    print("\nPlot created successfully!")

if __name__ == "__main__":
    main()