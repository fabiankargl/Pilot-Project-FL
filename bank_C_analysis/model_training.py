import pandas as pd
import numpy as np
import json 
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, f1_score, roc_auc_score
from typing import Tuple, Dict, Any, List


FILE_PATH = "../data/BankC.csv"
RANDOM_STATE = 42

results: Dict[str, Dict[str, float]] = {}


def load_preprocessing(filepath: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Loads the data, performs preprocessing (cleaning, encoding), and splits it.

    Args:
        filepath (str): The path to the CSV file.

    Returns:
        A tuple containing X_train, X_test, y_train, y_test DataFrames/Series.
    """
    print("\n--- Preprocessing ---")
    df = pd.read_csv(filepath, na_values='?')
    df = df.drop_duplicates().dropna()

    # Target Mapping
    df["income"] = df['income'].map({'<=50K': 0, '>50K': 1})
    print("Target distribution:\n", df["income"].value_counts())

    # One-Hot Encoding
    df_encoded = pd.get_dummies(df, drop_first=True)

    # Correlation Check
    correlations = df_encoded.corr()['income'].sort_values(ascending=False)
    print("\nTop 5 correlations:\n", correlations[:5])
    print("Flop 5 correlations:\n", correlations[-5:])

    # Feature Separation
    X = df_encoded.drop(columns=["income"])
    y = df_encoded["income"]

    # Remove 'institute' columns if present
    cols_to_drop = [c for c in X.columns if 'institute' in c]
    if cols_to_drop:
        X = X.drop(columns=cols_to_drop)

    return train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)


def evaluate_model(name: str, 
                   y_test: pd.Series, 
                   y_pred: np.ndarray, 
                   y_prob: np.ndarray = None) -> None:
    """
    Evaluates a model, prints the results, and stores them in the global 'results' dictionary,

    Args:
        name (str): The name of the model.
        y_test (pd.Series): The true target values.
        y_pred (np.ndarray): The predicted class labels.
        y_prob (np.ndarray): The predicted probabilities for the positive class (optional).
    """
    acc = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average="macro")
    auc_score = roc_auc_score(y_test, y_prob) if y_prob is not None else np.nan

    print(f"\nResult {name}:")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1-Score (Macro): {f1_macro:.4f}")
    if y_prob is not None:
        print(f"AUC-Score: {auc_score:.4f}")
    print(classification_report(y_test, y_pred))

    if "Base" in name or "PyTorch" in name:
        base_name = name.split(" ")[0].replace("(Base)", "").strip()
        results[base_name] = {
            'AUC_Score': float(auc_score),
            'Macro_F1_Score': float(f1_macro)
        }


def run_random_forest(X_train: pd.DataFrame, 
                      y_train: pd.Series,
                      X_test: pd.DataFrame, 
                      y_test: pd.Series) -> None:
    """
    Runs Random Forest classification and a corresponding GridSearchCV.

    Args:
        X_train, y_train, X_test, y_test (pd.DataFrame, pd.Series, pd.DataFrame, pd.Series): Training and test data sets.
    """
    print("\n--- Random Forest & GridSearch ---")

    model = RandomForestClassifier(n_estimators=50,
                                   random_state=RANDOM_STATE,
                                   class_weight='balanced',
                                   max_depth=20)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    evaluate_model("RandomForest (Base)", y_test, y_pred, y_prob)

    print("Running GridSearchCV for Random Forest...")
    rf = RandomForestClassifier(random_state=RANDOM_STATE)
    param_grid_rf: Dict[str, List[Any]] = {
        'n_estimators': [50, 100],
        'max_depth': [10, 20, None],
        'class_weight': ['balanced', None]
    }
    grid_rf = GridSearchCV(rf, param_grid_rf, cv=3, scoring='f1_macro', n_jobs=-1)
    grid_rf.fit(X_train, y_train)

    print("Best Random Forest F1-Score:", grid_rf.best_score_)
    print("Best Parameter:", grid_rf.best_params_)


def run_decision_tree(X_train: pd.DataFrame, 
                      y_train: pd.Series, 
                      X_test: pd.DataFrame, 
                      y_test: pd.Series, 
                      feature_names: List[str]) -> None:
    """
    Runs Decision Tree classification and a corresponding GridSearchCV.

    Args:
        X_train, y_train, X_test, y_test (pd.DataFrame, pd.Series, pd.DataFrame, pd.Series): Training and test data sets.
        feature_names (List[str]): List of feature names for potential rule export.
    """
    print("\n--- Decision Tree & GridSearch ---")

    tree_model = DecisionTreeClassifier(max_depth=10,
                                         random_state=RANDOM_STATE,
                                         min_samples_leaf=5,
                                         criterion='gini')
    tree_model.fit(X_train, y_train)
    y_pred = tree_model.predict(X_test)
    y_prob = tree_model.predict_proba(X_test)[:, 1]
    evaluate_model("DecisionTree (Base)", y_test, y_pred, y_prob)
    
    # Show Rules (Text)
    # rules = export_text(tree_model, feature_names=feature_names)
    # print(rules)

    print("Running GridSearchCV for Decision Tree...")
    tree = DecisionTreeClassifier(random_state=RANDOM_STATE)
    param_grid_tree: Dict[str, List[Any]] = {
        'max_depth': [5, 10, 15],
        'criterion': ['gini', 'entropy'],
        'min_samples_leaf': [1, 5]
    }
    grid_tree = GridSearchCV(tree, param_grid_tree, cv=3, scoring='f1_macro', n_jobs=-1)
    grid_tree.fit(X_train, y_train)

    print("Best Decision Tree F1-Score:", grid_tree.best_score_)
    print("Best Parameter:", grid_tree.best_params_)


def run_logistic_regression(X_train_scaled: np.ndarray, 
                            y_train: pd.Series, 
                            X_test_scaled: np.ndarray, 
                            y_test: pd.Series) -> None:
    """
    Runs Logistic Regression classification and a corresponding GridSearchCV on scaled data.

    Args:
        X_train_scaled, X_test_scaled (np.ndarray, np.ndarray): Scaled training and test feature arrays.
        y_train, y_test (pd.Series, pd.Series): Training and test target series.
    """
    print("\n--- Logistic Regression & GridSearch ---")

    log_model = LogisticRegression(random_state=RANDOM_STATE,
                                   max_iter=1000,
                                   C=10)

    log_model.fit(X_train_scaled, y_train)
    y_pred = log_model.predict(X_test_scaled)
    y_prob = log_model.predict_proba(X_test_scaled)[:, 1]
    evaluate_model("LogisticRegression (Base)", y_test, y_pred, y_prob)

    print("Running GridSearchCV for Logistic Regression...")
    log_reg = LogisticRegression(random_state=RANDOM_STATE)
    param_grid_log: Dict[str, List[Any]] = {
        'C': [0.1, 1, 10],
        'class_weight': ['balanced', None],
        'max_iter': [1000]
    }
    grid_log = GridSearchCV(log_reg, param_grid_log, cv=3, scoring='f1_macro', n_jobs=-1)
    grid_log.fit(X_train_scaled, y_train)

    print("Best Logistic Regression F1-Score:", grid_log.best_score_)
    print("Best Parameter:", grid_log.best_params_)


class BankNet(nn.Module):
    """
    A simple feed-forward neural network for binary classification.
    """
    def __init__(self, input_dim: int):
        super(BankNet, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, 64),
            nn.ReLU(),

            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def run_pytorch_nn(X_train_scaled: np.ndarray, 
                   y_train: pd.Series, 
                   X_test_scaled: np.ndarray, 
                   y_test: pd.Series) -> None:
    """
    Trains and evaluates a PyTorch neural network.

    Args:
        X_train_scaled, X_test_scaled (np.ndarray, np.ndarray): Scaled training and test feature arrays.
        y_train, y_test (pd.Series, pd.Series): Training and test target series.
    """
    print("\n--- PyTorch Neural Network ---")

    # Data Preparation
    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    input_dim = X_train_tensor.shape[1]
    model = BankNet(input_dim)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 20
    print(f"Starting training for {epochs} epochs...")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        if (epoch+1) % 5 == 0:
            print(f"Epoch {epoch+1}/{epochs} - Loss: {running_loss/len(train_loader):.4f}")

    model.eval()
    with torch.no_grad():
        y_pred_prob = model(X_test_tensor)
        y_pred_prob_np = y_pred_prob.numpy()
        y_pred_class = (y_pred_prob_np > 0.5).astype(int)

    evaluate_model("PyTorchNN", y_test, y_pred_class, y_pred_prob_np)


def save_results(results_dict: Dict[str, Dict[str, float]], 
                 filename: str = 'model_results.json') -> None:
    """
    Saves the model evaluation results (AUC and Macro F1) to a JSON file.

    Args:
        results_dict (Dict[str, Dict[str, float]]): Dictionary containing the model results.
        filename (str): Name of the file to save the results to.
    """
    print(f"\nSaving results to {filename}...")
    # Use float() to ensure numpy floats are converted to standard Python floats before serialization
    serializable_results = {k: {metric: float(v_metric) for metric, v_metric in v.items()} for k, v in results_dict.items()}
    with open(filename, 'w') as f:
        json.dump(serializable_results, f, indent=4)
    print("Results saved.")


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_preprocessing(FILE_PATH)

    # Scaling (Required for Logistic Regression & Neural Networks)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    run_random_forest(X_train, y_train, X_test, y_test)

    feature_names = list(X_train.columns)
    run_decision_tree(X_train, y_train, X_test, y_test, feature_names)

    run_logistic_regression(X_train_scaled, y_train, X_test_scaled, y_test)

    run_pytorch_nn(X_train_scaled, y_train, X_test_scaled, y_test)

    save_results(results, 'model_results.json')