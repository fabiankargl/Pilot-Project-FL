import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Deine gesammelten Ergebnisse (manuell eingetragen)
data = {
    'Model': ['Random Forest', 'Decision Tree', 'Logistic Regression', 'Neural Network (sklearn)', 'Neural Network (PyTorch)'],
    'AUC_Score': [0.9033, 0.8841, 0.8943, 0.8966, 0.9009],
    'Macro_F1_Score': [0.7584, 0.7591, 0.7489, 0.7588, 0.7558]
}

df_results = pd.DataFrame(data)

# Plotting
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df_results, x='AUC_Score', y='Macro_F1_Score', s=200, hue='Model', style='Model')

# Beschriftungen hinzufügen
for i in range(df_results.shape[0]):
    plt.text(
        df_results.AUC_Score[i] + 0.0002, 
        df_results.Macro_F1_Score[i], 
        df_results.Model[i], 
        fontdict={'size':9, 'weight':'bold'}
    )

plt.title('Model Battle: AUC vs Macro F1', fontsize=14)
plt.xlabel('AUC Score (Ranking Quality)', fontsize=12)
plt.ylabel('Macro F1 Score (Balanced Accuracy)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()

# Speichern und Anzeigen
plt.savefig('model_comparison.png')
plt.show()