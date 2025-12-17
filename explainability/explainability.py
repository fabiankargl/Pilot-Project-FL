import torch
import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Set
from sklearn.preprocessing import StandardScaler
from pilot.task import IncomeNet

def calculate_feature_importance(model: torch.nn.Module, 
                                 filepath: str, 
                                 expected_dim: int = 95) -> List[Tuple[str, float]]:
    """
    Calculates and prints permutation feature importance for a given model and dataset.

    Args:
        model (torch.nn.Module): The trained PyTorch model to evaluate.
        filepath (str): The path to the CSV data file.
        expected_dim (int, optional): The expected input dimension for the model.
                                      Defaults to 95.

    Returns:
        List[Tuple[str, float]]: A list of  tuples, sorted in descending order of importance.
    """
    print(f"\n--- EXPLAINABILITY ANALYSIS: {filepath} ---")
    
    df: pd.DataFrame = pd.read_csv(filepath, na_values='?')
    df = df.drop_duplicates().dropna()
    df["income"] = df['income'].map({'<=50K': 0, '>50K': 1})
    y_true: np.ndarray = df["income"].values
    
    original_cols: List[str] = [c for c in df.columns if c != 'income' and 'institute' not in c]
    
    df_encoded = pd.get_dummies(df, drop_first=True)
    X = df_encoded.drop(columns=["income"])
    cols_to_drop = [c for c in X.columns if 'institute' in c]
    if cols_to_drop: X = X.drop(columns=cols_to_drop)
    
    current_dim = X.shape[1]
    if current_dim > expected_dim:
        X = X.iloc[:, :expected_dim]
    elif current_dim < expected_dim:
        padding = pd.DataFrame(0, index=X.index, columns=[f"pad_{i}" for i in range(expected_dim - current_dim)])
        X = pd.concat([X, padding], axis=1)
        
    feature_names: List[str] = [str(c) for c in X.columns]
    
    scaler = StandardScaler()
    X_scaled: np.ndarray = scaler.fit_transform(X)
    X_tensor: torch.Tensor = torch.FloatTensor(X_scaled)
    
    model.eval()
    with torch.no_grad():
        base_preds: np.ndarray = (model(X_tensor).numpy().flatten() > 0.5).astype(int)
    base_acc: float = (base_preds == y_true).mean()
    print(f"   Baseline Accuracy: {base_acc:.2%}")
    
    feature_groups: Dict[str, List[int]] = {}
    used_indices: Set[int] = set()
    
    for i, col in enumerate(feature_names):
        if col in original_cols:
            feature_groups[col] = [i]
            used_indices.add(i)
            
    for orig in original_cols:
        if orig in feature_groups: continue
        indices = []
        for i, col in enumerate(feature_names):
            if i in used_indices: continue
            if col.startswith(f"{orig}_"):
                indices.append(i)
        if indices:
            feature_groups[orig] = indices
            used_indices.update(indices)
            
    importances: Dict[str, float] = {}
    X_numpy = X_tensor.numpy()
    
    print("   ... Calculating Importance (this may take a moment) ...")
    for feature, indices in feature_groups.items():
        if not indices: continue
        
        saved_cols = X_numpy[:, indices].copy()
        
        perm_idx = np.random.permutation(X_numpy.shape[0])
        X_numpy[:, indices] = X_numpy[perm_idx][:, indices]
        
        with torch.no_grad():
            perm_tensor = torch.FloatTensor(X_numpy)
            preds = (model(perm_tensor).numpy().flatten() > 0.5).astype(int)
        
        perm_acc = (preds == y_true).mean()
        drop = base_acc - perm_acc
        importances[feature] = float(drop)
        
        X_numpy[:, indices] = saved_cols
        
    print("\n   TOP 5 DRIVERS FOR DECISIONS:")
    sorted_imps: List[Tuple[str, float]] = sorted(importances.items(), key=lambda x: x[1], reverse=True)
    for f, v in sorted_imps[:5]:
        print(f"   {f:<20}: {v:.4f} (Accuracy Drop)")
        
    return sorted_imps

if __name__ == "__main__":
    model_path = '../pilot/results/final_model_fedprox_r5_e3_nn_lr0.005_mu1.0.pt'
    model: torch.nn.Module = IncomeNet(95)
    try:
        state_dict = torch.load(model_path, map_location='cpu')
        model.load_state_dict(state_dict)
    except: pass
    
    calculate_feature_importance(model, "../data/BankA.csv")