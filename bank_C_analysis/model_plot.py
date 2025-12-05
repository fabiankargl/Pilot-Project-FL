import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import json 

FILE_NAME = 'model_results.json'

try:
    with open(FILE_NAME, 'r') as f:
        data_loaded = json.load(f)
        
    print(f"Results successfully loaded from {FILE_NAME}.") 

    df_results = pd.DataFrame.from_dict(data_loaded, orient='index').reset_index()
    df_results = df_results.rename(columns={'index': 'Model'})

except FileNotFoundError:
    print(f"File '{FILE_NAME}' not found. Make sure the training script ('if __name__ == \"__main__\"') was executed first.") 

    data = {
        'Model': ['Random Forest', 'Decision Tree', 'Logistic Regression', 'PyTorchNN'],
        'AUC_Score': [0.9033, 0.8986, 0.8944, 0.9008], 
        'Macro_F1_Score': [0.7584, 0.7592, 0.7493, 0.7544]
    }
    df_results = pd.DataFrame(data)
    print("Using static fallback data for plotting.")


plt.figure(figsize=(10, 6))
sns.scatterplot(data=df_results, x='AUC_Score', y='Macro_F1_Score', s=200, hue='Model', style='Model')

for i in range(df_results.shape[0]):
    plt.text(
        df_results.AUC_Score[i] + 0.0002, 
        df_results.Macro_F1_Score[i], 
        df_results.Model[i], 
        fontdict={'size':9, 'weight':'bold'}
    )

plt.title('Model Comparison: AUC vs Macro F1', fontsize=14)
plt.xlabel('AUC Score (Ranking Quality)', fontsize=12)
plt.ylabel('Macro F1 Score (Balanced Accuracy)', fontsize=12)

plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()

plt.savefig('model_comparison.png')
plt.show()