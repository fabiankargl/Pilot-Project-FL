import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from typing import NoReturn

FILE_PATH = "../data/BankC.csv" 

def load_and_analyze(filepath: str) -> pd.DataFrame:
    """
    Loads data from a CSV file, performs initial analysis and cleaning.

    Args:
        filepath (str): The path to the input CSV file.

    Returns:
        pd.DataFrame: A cleaned pandas DataFrame with no duplicates or missing values.
    """
    print("--- Loading Data ---")
    df_raw = pd.read_csv(filepath)
    print("First 5 rows (raw):")
    print(df_raw.head(5))
    print("\nInfo (raw):")
    print(df_raw.info())

    # Reload with correct NA handling
    df = pd.read_csv(filepath, na_values='?')
    print("\nFirst 5 rows (with NaNs):")
    print(df.head(5))
    print("\nMissing values per column:")
    print(df.isnull().sum())
    
    # Duplicate check
    print(f"\nNumber of duplicates: {df.duplicated().sum()}")
    print("Unique Institutes:", df['institute'].unique())

    # Cleaning
    df = df.drop_duplicates()
    print(f"Duplicates removed. Remaining rows: {len(df)}")
    
    df = df.dropna()
    print(f"NaNs removed. Final dataset length: {len(df)}")
    print(df.info())
    
    return df

def run_plots(df: pd.DataFrame) -> NoReturn:
    """
    Generates and displays a series of exploratory data analysis (EDA) plots.
    """
    sns.set_theme(style="whitegrid")

    # 1. Target Count
    plt.figure(figsize=(8, 6))
    ax = sns.countplot(data=df, x="income", hue="income", palette="viridis")
    plt.title('Count of target variable income')
    plt.xlabel('Income')
    plt.ylabel('Count')
    for container in ax.containers:
        ax.bar_label(container)
    plt.show()

    # 2. Target Percentage
    plt.figure(figsize=(8, 6))
    ax = sns.countplot(data=df, x="income", hue="income", stat="percent", palette="viridis")
    plt.title('Percentage of Customers by Income Level')
    plt.xlabel('Income')
    plt.ylabel('Percentage (%)')
    for container in ax.containers:
        ax.bar_label(container, fmt='%.2f%%')
    plt.show()

    # 3. Age Distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x="age", kde=True, color="blue")
    plt.title('Distribution of age')
    plt.xlabel('Age')
    plt.ylabel('Count')
    plt.show()

    # 4. Age vs Income Boxplot
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=df, x='income', y='age', hue='income', palette="viridis")
    plt.title('Age Distribution by Income Level')
    plt.xlabel('Income')
    plt.ylabel('Age')
    plt.show()

    # 5. Education vs Income
    plt.figure(figsize=(12, 8))
    sns.countplot(data=df, y='education', hue="income", palette="viridis")
    plt.title('Count of Education per income')
    plt.xlabel('Income')
    plt.ylabel('Education')
    plt.show()

if __name__ == "__main__":    
    df_clean = load_and_analyze(FILE_PATH)
    run_plots(df_clean)