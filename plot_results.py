import json
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import os
import glob
import itertools
from typing import Dict, Any, Optional, Tuple, Iterator

plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.dpi'] = 150
RESULTS_DIR = "presentation_results"

def load_experiment_data(exp_name: str) -> Tuple[Optional[Dict], Optional[Dict]]:
    """
    Loads global and local history JSON files for a given experiment name.

    Args:
        exp_name: The unique identifier for the experiment run.

    Returns:
        A tuple containing the global and local history dictionaries.
        Returns (None, None) if files are not found.
    """
    global_file = os.path.join(RESULTS_DIR, f"global_history_{exp_name}.json")
    local_file = os.path.join(RESULTS_DIR, f"local_history_{exp_name}.json")
    
    if not os.path.exists(global_file) or not os.path.exists(local_file):
        return None, None
        
    with open(global_file, "r") as f:
        global_hist = json.load(f)
    with open(local_file, "r") as f:
        local_hist = json.load(f)
        
    return global_hist, local_hist

def extract_client_data(local_history: Dict[str, Any], 
                        metric_key: str) -> Dict[int, Dict[str, list]]:
    """
    Parses local history to extract a specific metric for each client over all rounds.

    Args:
        local_history: The local history dictionary containing results from all
            clients across all rounds.
        metric_key: The key for the metric to extract.

    Returns:
        A dictionary where keys are client IDs and values are dictionaries
        containing lists of rounds and corresponding metric values.
    """
    client_data = {}
    if "client_results" not in local_history:
        return {}

    for round_idx, round_data in enumerate(local_history["client_results"]):
        if round_idx >= len(local_history["round"]):
            break
            
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

def format_legend_label(exp_name: str) -> str:
    """
    Parses an experiment name to create a clean legend label.

    Args:
        exp_name: The experiment name string.

    Returns:
        A formatted string suitable for plot legends.
    """
    model = "IncomeNet-66k"
    if "logreg" in exp_name.lower():
        model = "Logistic Regression"
    
    strategy = "FedAvg"
    if "fedprox" in exp_name.lower():
        strategy = "FedProx"
  
    return f"{model} {strategy}"

def get_plot_style(exp_name: str, 
                   color_cycle: Iterator, 
                   marker_cycle: Iterator) -> Dict[str, Any]:
    """
    Determines the plot style (color, marker, linestyle) for an experiment.

    Args:
        exp_name: The name of the experiment.
        color_cycle: An iterator for plot colors.
        marker_cycle: An iterator for plot markers.

    Returns:
        A dictionary containing style properties for plotting.
    """
    exp_lower = exp_name.lower()
    
    style = {
        'color': next(color_cycle),
        'marker': next(marker_cycle),
        'linestyle': '-'
    }

    if "logreg" in exp_lower:
        style['linestyle'] = '--' 
        
        if "fedprox" in exp_lower:
            style['color'] = '#9467bd' 
            style['marker'] = 'x'
        elif "fedavg" in exp_lower:
            style['color'] = '#ff7f0e' 
            style['marker'] = 'v'      
            
    else:
        style['linestyle'] = '-'  
        
        if "fedprox" in exp_lower:
            style['color'] = '#2ca02c' 
            style['marker'] = 'D'     
        elif "fedavg" in exp_lower:
            style['color'] = '#1f77b4'  
            style['marker'] = 'o'      

    return style

def plot_single_experiment(exp_name: str, 
                           global_history: Dict[str, Any],
                           local_history: Dict[str, Any]) -> None:
    """
    Generates and saves a detailed 2x2 plot for a single experiment.

    The plot shows global and per-client metrics for evaluation loss, accuracy,
    F1 score, and AUC over the federated learning rounds.

    Args:
        exp_name: The unique name of the experiment.
        global_history: The aggregated global metrics history.
        local_history: The per-client local metrics history.
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.patch.set_facecolor('white')
    
    rounds = global_history["round"]
    client_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'] 
    
    metrics = [
        ('eval_loss', 'Evaluation Loss', axes[0, 0], 'black'),
        ('eval_acc', 'Accuracy', axes[0, 1], 'darkgreen'),
        ('eval_f1', 'F1 Score (Macro)', axes[1, 0], 'darkgreen'),
        ('eval_auc', 'AUC-ROC', axes[1, 1], 'darkgreen')
    ]
    
    formatted_title = format_legend_label(exp_name)

    for metric_key, metric_name, ax, global_color in metrics:
        ax.set_facecolor('white')
        ax.grid(True, alpha=0.4, linestyle='-', linewidth=0.8, color='gray', zorder=0)
        
        client_data = extract_client_data(local_history, metric_key)
        for client_id in sorted(client_data.keys()):
            data = client_data[client_id]
            c_idx = int(client_id) % len(client_colors)
            label = f"Bank {chr(65+int(client_id))}" if int(client_id) < 3 else f"Client {client_id}"
            
            ax.plot(data["rounds"], data["values"], 
                    marker='o', linestyle='--', linewidth=1.5, markersize=6,
                    alpha=0.6, color=client_colors[c_idx], label=label, zorder=5)
        
        ax.plot(rounds, global_history[metric_key], 
                marker='D', linestyle='-', linewidth=3.0, markersize=8,
                label='Global Model', color=global_color, zorder=10)
        
        ax.set_xlabel('Round', fontweight='bold')
        ax.set_ylabel(metric_name, fontweight='bold')
        ax.set_title(f'{metric_name}', fontweight='bold', fontsize=12)
        
        ax.legend(loc='best', fontsize=9, framealpha=0.9)
        ax.set_xticks(rounds)
    
    plt.suptitle(f'Experiment Detail: {formatted_title}', fontsize=16, fontweight='bold', y=0.99)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f'plot_{exp_name}.png'), dpi=150)
    plt.close()
    print(f"  -> Detail plot saved: plot_{exp_name}.png")

def plot_comparison(all_experiments: Dict[str, Tuple[Dict, Dict]]) -> None:
    """
    Generates and saves a comprehensive 2x2 comparison plot for multiple experiments.

    This function visualizes the global performance metrics (loss, accuracy, F1-score,
    and AUC) across different experimental runs. It plots each experiment's global
    history on four separate subplots.

    Args:
        all_experiments: A dictionary where keys are experiment names and values are
            tuples containing the global and local history dictionaries for that run.
    """
    if len(all_experiments) < 2:
        print("Not enough experiments for a comparison plot.")
        return

    print(f"\nCreating comparison plot for {len(all_experiments)} experiments...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.patch.set_facecolor('#f5f5f5') 
    
    metrics_config = [
        ('eval_loss', 'Evaluation Loss', axes[0, 0]), 
        ('eval_acc', 'Accuracy', axes[0, 1]),        
        ('eval_f1', 'F1 Score (Macro)', axes[1, 0]),
        ('eval_auc', 'AUC-ROC', axes[1, 1])
    ]
    
    color_cycle = itertools.cycle(plt.cm.tab10.colors)
    marker_cycle = itertools.cycle(['o', 's', 'D', '^', 'v', 'X'])

    exp_styles = {}
    for name in sorted(all_experiments.keys()):
        exp_styles[name] = get_plot_style(name, color_cycle, marker_cycle)

    legend_handles = []
    legend_labels = []
    
    for i, (metric_key, title, ax) in enumerate(metrics_config):
        ax.set_facecolor('white')
        ax.grid(True, which='major', linestyle='--', linewidth=0.7, color='gray', alpha=0.5, zorder=0)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        
        max_round = 0
        
        current_metric_final_values = {}

        for exp_name in sorted(all_experiments.keys()):
            glob_hist, _ = all_experiments[exp_name]
            rounds = glob_hist["round"]
            values = glob_hist[metric_key]
            
            if rounds[-1] > max_round:
                max_round = rounds[-1]

            if "logreg" in exp_name.lower():
                current_metric_final_values['LogReg'] = (rounds[-1], values[-1])
            else:
                current_val = values[-1]
                if 'IncomeNet' not in current_metric_final_values or current_val > current_metric_final_values['IncomeNet'][1]:
                    current_metric_final_values['IncomeNet'] = (rounds[-1], current_val)

            style = exp_styles[exp_name]
            label_text = format_legend_label(exp_name)
            
            line, = ax.plot(rounds, values, 
                    marker=style['marker'], 
                    linewidth=2.5,
                    linestyle=style['linestyle'],
                    markersize=8,
                    color=style['color'],
                    alpha=0.8,
                    zorder=10)
            
            if i == 0:
                if label_text not in legend_labels:
                    legend_labels.append(label_text)
                    legend_handles.append(line)

        target_metrics = ['eval_acc', 'eval_f1', 'eval_auc']
        
        if metric_key in target_metrics and 'IncomeNet' in current_metric_final_values and 'LogReg' in current_metric_final_values:
            x_pos, y_incomenet = current_metric_final_values['IncomeNet']
            _, y_logreg = current_metric_final_values['LogReg']
            
            gap = y_incomenet - y_logreg
            mid_point = (y_incomenet + y_logreg) / 2
            
            ax.annotate(
                '', xy=(x_pos, y_incomenet), xytext=(x_pos, y_logreg),
                arrowprops=dict(arrowstyle='<->', color='red', lw=2)
            )
            
            ax.text(
                x_pos - (max_round * 0.05), mid_point, 
                f'+{gap*100:.1f}%\nLift', 
                color='red', 
                fontweight='bold', 
                fontsize=11,
                ha='right', 
                va='center',
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="red", alpha=0.8)
            )

        ax.set_title(title, fontweight='bold', fontsize=12)
        ax.set_xlabel("Round", fontweight='bold')

    plt.suptitle('Benchmark: Model Type & Strategy Comparison', fontsize=18, fontweight='bold', y=0.96)
    
    plt.subplots_adjust(bottom=0.18, hspace=0.3, wspace=0.2)
    
    fig.legend(
        handles=legend_handles, 
        labels=legend_labels, 
        loc='lower center', 
        ncol=4,
        bbox_to_anchor=(0.5, 0.02), 
        frameon=True, 
        shadow=True, 
        fancybox=True, 
        fontsize=10, 
        title="Experiment Settings"
    )
    
    save_path = os.path.join(RESULTS_DIR, 'final_benchmark_comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Final benchmark plot saved: {save_path}")
    
def main() -> None:
    search_pattern = os.path.join(RESULTS_DIR, "global_history_*.json")
    files = glob.glob(search_pattern)
    
    if not files:
        print(f"No files found in {RESULTS_DIR}!")
        return

    exp_names = []
    for f in files:
        filename = os.path.basename(f)
        name = filename.replace("global_history_", "").replace(".json", "")
        exp_names.append(name)

    print(f"Found experiments: {exp_names}")
    
    all_data = {}

    for name in exp_names:
        g_hist, l_hist = load_experiment_data(name)
        if g_hist and l_hist:
            all_data[name] = (g_hist, l_hist)
            plot_single_experiment(name, g_hist, l_hist)
    
    plot_comparison(all_data)

if __name__ == "__main__":
    main()