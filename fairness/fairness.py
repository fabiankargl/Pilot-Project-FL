import torch
import pandas as pd
import numpy as np
from typing import Tuple, Optional, Dict
from sklearn.preprocessing import StandardScaler
import os
from pilot.task import IncomeNet

def prepare_data(filepath: str, 
                 expected_dim: int = 95) -> Tuple[Optional[torch.Tensor], Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Loads data from a CSV file, preprocesses it, and prepares it for model input.

    Args:
        filepath (str): The path to the input CSV file.
        expected_dim (int, optional): The target dimension for the feature set.
                                      Defaults to 95.

    Returns:
        A tuple containing:
        - Optional[torch.Tensor]: The processed feature tensor, ready for the model.
        - Optional[np.ndarray]: The array of true labels.
        - Optional[np.ndarray]: The array of binary gender labels (1 for Male, 0 for Female).
        Returns (None, None, None) if the file is not found or the 'gender' column is missing.
    """
    if not os.path.exists(filepath):
        print(f"Error: File not found: {filepath}")
        return None, None, None

    df = pd.read_csv(filepath, na_values='?')
    df = df.drop_duplicates().dropna()
    
    df["income"] = df['income'].map({'<=50K': 0, '>50K': 1})
    y_true: np.ndarray = df["income"].values

    if 'gender' in df.columns:
        gender_binary: np.ndarray = df['gender'].apply(lambda x: 1 if 'Male' in str(x).strip() else 0).values
    else:
        print("Warning: 'gender' column not found!")
        return None, None, None

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

    scaler = StandardScaler()
    X_scaled: np.ndarray = scaler.fit_transform(X)
    X_tensor: torch.Tensor = torch.FloatTensor(X_scaled)

    return X_tensor, y_true, gender_binary

def run_fairness_audit(model: torch.nn.Module, 
                       X_tensor: torch.Tensor, 
                       y_true: np.ndarray, 
                       gender_binary: np.ndarray, 
                       bank_name: str) -> Tuple[float, np.ndarray]:
    """
    Performs a fairness audit on the model's predictions based on gender.

    Args:
        model (torch.nn.Module): The trained model to be audited.
        X_tensor (torch.Tensor): The input feature tensor.
        y_true (np.ndarray): The ground truth labels.
        gender_binary (np.ndarray): The binary gender labels for each sample.
        bank_name (str): The name of the bank/dataset being audited, for logging.

    Returns:
        A tuple containing:
        - float: The True Positive Rate for the male subgroup.
        - np.ndarray: The raw model output probabilities for each sample.
    """
    print(f"\n--- FAIRNESS AUDIT: {bank_name} ---")
    
    model.eval()
    with torch.no_grad():
        y_probs: np.ndarray = model(X_tensor).numpy().flatten()
        y_pred: np.ndarray = (y_probs > 0.5).astype(int)

    analysis_df = pd.DataFrame({
        'gender': gender_binary,
        'true': y_true,
        'pred': y_pred
    })

    grp_male = analysis_df[analysis_df['gender'] == 1]
    grp_female = analysis_df[analysis_df['gender'] == 0]

    tpr_male: float = grp_male[grp_male['true'] == 1]['pred'].mean()
    tpr_female: float = grp_female[grp_female['true'] == 1]['pred'].mean()
    
    gap: float = abs(tpr_male - tpr_female)

    print(f"   True Positive Rate (Male):   {tpr_male:.2%}")
    print(f"   True Positive Rate (Female): {tpr_female:.2%}")
    print(f"   Equal Opportunity Gap:       {gap:.2%}")
    
    if gap > 0.05:
        print("   -> RESULT: Model shows significant BIAS.")
    else:
        print("   -> RESULT: Model is fair.")
    
    return tpr_male, y_probs

def optimize_threshold(y_probs: np.ndarray, 
                       y_true: np.ndarray, 
                       gender_binary: np.ndarray, 
                       target_tpr_male: float) -> None:
    """
    Finds an optimal prediction threshold for the female subgroup to mitigate bias.

    Args:
        y_probs (np.ndarray): The raw model output probabilities.
        y_true (np.ndarray): The ground truth labels.
        gender_binary (np.ndarray): The binary gender labels.
        target_tpr_male (float): The TPR of the male subgroup to use as a target.
    """
    print(f"--- OPTIMIZATION (Fairness Layer) ---")
    
    mask_female = (gender_binary == 0)
    best_thresh = 0.5
    best_gap = 1.0
    best_female_tpr = 0.0

    search_space: np.ndarray = np.linspace(0.3, 0.6, 100)
    
    for t in search_space:
        fem_preds: np.ndarray = (y_probs[mask_female] > t).astype(int)
        
        if sum(y_true[mask_female] == 1) > 0:
            fem_tpr: float = fem_preds[y_true[mask_female] == 1].mean()
            current_gap: float = abs(fem_tpr - target_tpr_male)
            
            if current_gap < best_gap:
                best_gap = current_gap
                best_thresh = t
                best_female_tpr = fem_tpr

    print(f"   Target TPR (Male):           {target_tpr_male:.2%}")
    print(f"   New Threshold (Female):      {best_thresh:.3f}")
    print(f"   New TPR (Female):            {best_female_tpr:.2%}")
    print(f"   New Gap:                     {best_gap:.2%}")

if __name__ == "__main__":
    MODEL_PATH = '../pilot/results/final_model_fedprox_r5_e3_nn_lr0.005_mu1.0.pt'
    DATA_FILES: Dict[str, str] = {
        "Bank A": "../data/BankA.csv",
        "Bank B": "../data/BankB.csv",
        "Bank C": "../data/BankC.csv"
    }

    print("... Loading model ...")
    model: torch.nn.Module = IncomeNet(input_dim=95)
    try:
        state_dict = torch.load(MODEL_PATH, map_location='cpu')
        model.load_state_dict(state_dict)
    except Exception as e:
        print(f"CRITICAL ERROR: Could not load model. Check path!\n{e}")
        exit()

    for bank_name, file_path in DATA_FILES.items():
        X, y, gender = prepare_data(file_path, expected_dim=95)
        
        if X is not None:
            tpr_male_baseline, probabilities = run_fairness_audit(model, X, y, gender, bank_name) # type: ignore
            
            optimize_threshold(probabilities, y, gender, tpr_male_baseline) # type: ignore
            print("-" * 50)