import torch
import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Set
from sklearn.preprocessing import StandardScaler
from pilot.task import IncomeNet

def explain_local_prediction(model: torch.nn.Module, 
                             X_tensor: torch.Tensor, 
                             feature_groups: Dict[str, List[int]], 
                             feature_names: List[str],
                             customer_idx: int = 0) -> None:
    """
    Explains a single prediction for a specific customer using a gradient-based method.

    Args:
        model (torch.nn.Module): The trained model to explain.
        X_tensor (torch.Tensor): The preprocessed input data for all customers.
        feature_groups (Dict[str, List[int]]): Maps original feature names to their column indices.
        feature_names (List[str]): A list of all feature names after encoding/padding.
        customer_idx (int): The index of the customer in `X_tensor` to explain.
    """
    model.eval()
    
    single_input = X_tensor[customer_idx].unsqueeze(0).detach().clone()
    single_input.requires_grad_() 
    
    output = model(single_input)
    prediction_score = output.item()
    decision = "Approved (>50k)" if prediction_score > 0.5 else "Rejected (<=50k)"
    
    print(f"\n--- Local explainability (Customer #{customer_idx}) ---")
    print(f"   Prediction: {decision}")
    print(f"   Score:      {prediction_score:.4f}")
    
    output.backward()
    grads = single_input.grad.data.numpy().flatten()
    
    local_impacts = {}
    
    for feature, indices in feature_groups.items():
        if not indices: continue
        
        group_grads = grads[indices]
        input_vals = single_input.detach().numpy().flatten()[indices]
        
        impact = np.sum(np.abs(group_grads * input_vals))
        local_impacts[feature] = impact

    sorted_local = sorted(local_impacts.items(), key=lambda x: x[1], reverse=True)
    
    print("Top 3 Factors for THIS customer:")
    for f, v in sorted_local[:3]:
        print(f"   -> {f:<15}: Impact {v:.4f}")


def calculate_feature_importance(model: torch.nn.Module, 
                                 filepath: str, 
                                 expected_dim: int = 95) -> List[Tuple[str, float]]:
    """
    Calculates global feature importance and demonstrates local explanations.
    Args:
        model (torch.nn.Module): The trained model to be analyzed.
        filepath (str): The path to the raw CSV data file.
        expected_dim (int): The feature dimension the model expects as input.

    Returns:
        List[Tuple[str, float]]: A list of (feature_name, importance_score) tuples,
                                 sorted by importance in descending order.
    """
    print(f"--- Explainability analysis: {filepath} ---")
    
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
    
    print("   ... Calculating Global Importance (Permutation) ...")
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
        
    print("\n   Top 5 drivers (Global):")
    sorted_imps: List[Tuple[str, float]] = sorted(importances.items(), key=lambda x: x[1], reverse=True)
    for f, v in sorted_imps[:5]:
        print(f"   {f:<20}: {v:.4f} (Accuracy Drop)")

    explain_local_prediction(model, X_tensor, feature_groups, feature_names, customer_idx=0)
    
    explain_local_prediction(model, X_tensor, feature_groups, feature_names, customer_idx=67)
        
    return sorted_imps

if __name__ == "__main__":
    model_path = '../pilot/results/final_model_fedprox_r5_e3_nn_lr0.005_mu1.0.pt'
    model: torch.nn.Module = IncomeNet(95)
    try:
        state_dict = torch.load(model_path, map_location='cpu')
        model.load_state_dict(state_dict)
    except: pass
    
    calculate_feature_importance(model, "../data/BankA.csv")