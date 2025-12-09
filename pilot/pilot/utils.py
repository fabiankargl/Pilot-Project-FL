import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typing import Tuple

RANDOM_STATE = 42

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
        
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
    
    # Apply scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test