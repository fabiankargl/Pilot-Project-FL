import torch
import time
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pilot.task import IncomeNet

model_path = '../pilot/results/final_model_fedprox_r5_e3_nn_lr0.005_mu1.0.pt'

try:
    state_dict = torch.load(model_path)
    first_layer_weight = state_dict['network.0.weight']
    input_dim = first_layer_weight.shape[1]
    print(f"Model loaded successfully. Input dimension: {input_dim}")
except FileNotFoundError:
    print(f"Warning: File '{model_path}' not found. Using initialized random weights for testing.")
    input_dim = 95 
    state_dict = None

model = IncomeNet(input_dim)
if state_dict:
    model.load_state_dict(state_dict)

model.eval() 

batch_sizes = [1, 16, 32, 64, 128, 256, 512, 1024, 2048]
num_warmup = 20
num_runs = 100 

results_data = []

print("\nStarting Benchmark...")
print(f"{'Batch Size':<15} | {'Avg Latency (ms)':<20} | {'Throughput (req/sec)':<25}")
print("-" * 65)

with torch.no_grad():
    for bs in batch_sizes:
        dummy_input = torch.randn(bs, input_dim)
        
        for _ in range(num_warmup):
            _ = model(dummy_input)
            
        batch_times = []
        for _ in range(num_runs):
            start_time = time.time()
            _ = model(dummy_input)
            end_time = time.time()
            
            batch_times.append((end_time - start_time) * 1000)
        
        avg_latency = np.mean(batch_times)
        throughput = bs / (avg_latency / 1000)
        
        print(f"{bs:<15} | {avg_latency:<20.4f} | {throughput:<25.2f}")
        
        for t in batch_times:
            results_data.append({'Batch Size': str(bs), 'Latency (ms)': t})

df_results = pd.DataFrame(results_data)

sns.set_theme(style="whitegrid", context="talk")

plt.figure(figsize=(12, 7))

ax = sns.boxplot(x="Batch Size", y="Latency (ms)", hue="Batch Size", data=df_results, 
                 palette="viridis", linewidth=1.5, fliersize=3)

plt.title('IncomeNet-66k Inference Latency Distribution', fontsize=18, fontweight='bold', pad=20)
plt.ylabel('Latency in Milliseconds (ms)', fontsize=14)
plt.xlabel('Batch Size', fontsize=14)

sns.despine(trim=True, left=True)

plt.grid(axis='y', linestyle='--', alpha=0.6)

plt.tight_layout()
plt.savefig('latency_boxplot.png', dpi=300)
print("\nPlot saved as 'latency_boxplot.png'")
plt.show()