import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier


BASE = os.path.dirname(os.path.abspath(__file__))

CSV_PATH = os.path.join(BASE, "data", "BankB.csv")

PLOT_DIR = os.path.join(BASE, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)


def safe_name(name: str) -> str:
    """macht aus Spaltennamen einfache Dateinamen"""
    return (
        str(name)
        .replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
        .replace(">", "gt")
        .replace("<", "lt")
        .replace("=", "eq")
        .replace(":", "_")
    )


def visualize_data(csv_path: str = CSV_PATH, output_dir: str = PLOT_DIR) -> None:
pd.read_csv(csv_path)

    if "income" in df.columns:
        plt.figure()
        df["income"].value_counts().plot(kind="bar")
        plt.title("Verteilung income")
        plt.xlabel("income")
        plt.ylabel("Anzahl")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "dist_income.png"))
        plt.close()

    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    for col in cat_cols:
        plt.figure()
        df[col].value_counts().plot(kind="bar")
        plt.title(f"Verteilung {col}")
        plt.xlabel(col)
        plt.ylabel("Anzahl")
        plt.tight_layout()
        fname = f"dist_cat_{safe_name(col)}.png"
        plt.savefig(os.path.join(output_dir, fname))
        plt.close()

    num_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()

    for col in num_cols:
        plt.figure()
        plt.hist(df[col].dropna(), bins=30)
        plt.title(f"Histogramm {col}")
        plt.xlabel(col)
        plt.ylabel("Anzahl")
        plt.tight_layout()
        fname = f"hist_{safe_name(col)}.png"
        plt.savefig(os.path.join(output_dir, fname))
        plt.close()

    if "income" in df.columns:
        for col in num_cols:
            if col == "income":
                continue
            plt.figure()
            df.boxplot(column=col, by="income")
            plt.title(f"{col} nach income")
            plt.suptitle("")
            plt.xlabel("income")
            plt.ylabel(col)
            plt.tight_layout()
            fname = f"box_{safe_name(col)}_by_income.png"
            plt.savefig(os.path.join(output_dir, fname))
            plt.close()

    if len(num_cols) > 1:
        corr = df[num_cols].corr()
        plt.figure(figsize=(8, 6))
        sns.heatmap(corr, annot=False, cmap="coolwarm", fmt=".2f")
        plt.title("Korrelations-Heatmap (numerische Features)")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "corr_heatmap_numerical.png"))
        plt.close()

    print(f"Plots gespeichert in: {output_dir}")


def load_data(csv_path: str = CSV_PATH):
    df = pd.read_csv(csv_path)

    target_col = "income"

    df[target_col] = (df[target_col].astype(str).str.strip() == ">50K").astype(int)

    drop_cols = []
    if "institute" in df.columns:
        drop_cols.append("institute")

    y = df[target_col]
    X = df.drop(columns=[target_col] + drop_cols)

    # Kategoriale Features in Dummies umwandeln
    X = pd.get_dummies(X, drop_first=True)

    return X, y

def plot_feature_importance(importances, feature_names, title, filename, top_n=20):
    """Speichert einen Bar-Plot der wichtigsten Features."""
    importances = np.array(importances)
    feature_names = np.array(feature_names)

    idx = np.argsort(importances)[::-1]  # absteigend
    idx = idx[:top_n]

    plt.figure(figsize=(8, 6))
    plt.barh(range(len(idx)), importances[idx][::-1])
    plt.yticks(range(len(idx)), feature_names[idx][::-1])
    plt.xlabel("Wichtigkeit")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, filename))
    plt.close()


def train_and_evaluate():
    X, y = load_data()
    feature_names = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        "SimpleNN": MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            solver="adam",
            max_iter=300,
            random_state=42,
        ),
        "LogisticRegression": LogisticRegression(
            max_iter=1000,
            random_state=42,
            n_jobs=-1,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            n_jobs=-1,
        ),
        "DecisionTree": DecisionTreeClassifier(random_state=42),
    }

    print(f"{'Model':<18} {'Accuracy':>10} {'F1':>10} {'AUC':>10}")
    print("-" * 50)

    for name, model in models.items():
        print(f"Training: {name}...")

        if name in ["SimpleNN", "LogisticRegression"]:
            model.fit(X_train_scaled, y_train)
            X_test_used = X_test_scaled
        else:
            model.fit(X_train, y_train)
            X_test_used = X_test

        y_pred = model.predict(X_test_used)

        if hasattr(model, "predict_proba"):
            y_score = model.predict_proba(X_test_used)[:, 1]
        else:
            y_score = None

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, pos_label=1)
        auc = roc_auc_score(y_test, y_score) if y_score is not None else float("nan")

        print(
            f"{name:<18} "
            f"{acc:>10.4f} "
            f"{f1:>10.4f} "
            f"{auc:>10.4f}"
        )

        if hasattr(model, "feature_importances_"):
            plot_feature_importance(
                importances=model.feature_importances_,
                feature_names=feature_names,
                title=f"Feature Importance {name}",
                filename=f"feature_importance_{name}.png",
                top_n=20,
            )

        if name == "LogisticRegression" and hasattr(model, "coef_"):
            importances = np.abs(model.coef_[0])
            plot_feature_importance(
                importances=importances,
                feature_names=feature_names,
                title="Feature Importance LogisticRegression (|coef|)",
                filename="feature_importance_LogisticRegression.png",
                top_n=20,
            )


def main():
    visualize_data()

    train_and_evaluate()


if __name__ == "__main__":
    main()
