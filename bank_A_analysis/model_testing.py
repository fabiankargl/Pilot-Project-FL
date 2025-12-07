import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve

from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Load Data
df = pd.read_csv('../data/BankA.csv')
print(df.head())

df = df.replace("?", pd.NA)

# Clean income field
df["income"] = df["income"].astype(str).str.strip()
df["income"] = df["income"].str.contains(">50K").astype(int)

# Feature sets
num_cols = ["age", "educational-num", "capital-gain", "capital-loss", "hours-per-week"]
cat_cols = [col for col in df.columns if col not in num_cols + ["income"]]

df[cat_cols] = df[cat_cols].fillna("Unknown")

X = df.drop("income", axis=1)
y = df["income"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# Preprocessors
preprocess_for_trees = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
], remainder='passthrough')

preprocess_for_nn = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
])

# For NN manual preprocessing
X_train_nn = preprocess_for_nn.fit_transform(X_train)
X_test_nn = preprocess_for_nn.transform(X_test)

# Helper function for metrics
def evaluate_model(name, y_test, y_pred, y_proba):
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)

    print(f"{name}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")

    return roc_auc

# Decision Tree
pipeline_dt = Pipeline([
    ("prep", preprocess_for_trees),
    ("model", DecisionTreeClassifier(max_depth=15, random_state=42, min_samples_leaf=5))
])

pipeline_dt.fit(X_train, y_train)
y_pred_dt = pipeline_dt.predict(X_test)
y_proba_dt = pipeline_dt.predict_proba(X_test)[:, 1]
roc_auc_dt = evaluate_model("Decision Tree", y_test, y_pred_dt, y_proba_dt)

# ROC curve
fpr_dt, tpr_dt, _ = roc_curve(y_test, y_proba_dt)

# Logistic Regression
pipeline_lr = Pipeline([
    ("prep", preprocess_for_trees),
    ("model", LogisticRegression(max_iter=1000))
])

pipeline_lr.fit(X_train, y_train)
y_pred_lr = pipeline_lr.predict(X_test)
y_proba_lr = pipeline_lr.predict_proba(X_test)[:, 1]
roc_auc_lr = evaluate_model("Logistic Regression", y_test, y_pred_lr, y_proba_lr)

fpr_lr, tpr_lr, _ = roc_curve(y_test, y_proba_lr)

# Random Forest
pipeline_rf = Pipeline([
    ("prep", preprocess_for_trees),
    ("model", RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1))
])

pipeline_rf.fit(X_train, y_train)
y_pred_rf = pipeline_rf.predict(X_test)
y_proba_rf = pipeline_rf.predict_proba(X_test)[:, 1]
roc_auc_rf = evaluate_model("Random Forest", y_test, y_pred_rf, y_proba_rf)

fpr_rf, tpr_rf, _ = roc_curve(y_test, y_proba_rf)

# Neural Network
input_dim = X_train_nn.shape[1]

model = Sequential([
    Dense(64, activation='relu', input_shape=(input_dim,)),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = model.fit(
    X_train_nn, y_train,
    validation_split=0.2,
    epochs=50,
    batch_size=32,
    callbacks=[early_stop],
    verbose=1
)

# Evaluate NN
y_proba_nn = model.predict(X_test_nn).flatten()
y_pred_nn = (y_proba_nn > 0.5).astype(int)

roc_auc_nn = evaluate_model("Neural Network", y_test, y_pred_nn, y_proba_nn)

fpr_nn, tpr_nn, _ = roc_curve(y_test, y_proba_nn)

# Combined ROC Curve Plot
plt.figure(figsize=(8,6))

plt.plot(fpr_dt, tpr_dt, label=f"Decision Tree (AUC = {roc_auc_dt:.3f})")
plt.plot(fpr_lr, tpr_lr, label=f"Logistic Regression (AUC = {roc_auc_lr:.3f})")
plt.plot(fpr_rf, tpr_rf, label=f"Random Forest (AUC = {roc_auc_rf:.3f})")
plt.plot(fpr_nn, tpr_nn, label=f"Neural Network (AUC = {roc_auc_nn:.3f})")

# Random baseline
plt.plot([0,1],[0,1], linestyle="--")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.4)
plt.show()
