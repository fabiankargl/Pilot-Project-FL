import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt

# --------- Analysis ---------
df_bankC = pd.read_csv("../data/BankC.csv")

print(df_bankC.head(5))

print(f"\n{df_bankC.info()}")

df_bankC = pd.read_csv("../data/BankC.csv", na_values='?')

print(df_bankC.head(5))

print(df_bankC.isnull().sum())

print("Number of duplicates: ", df_bankC.duplicated().sum())

print(df_bankC['institute'].unique())

df_bankC = df_bankC.drop_duplicates()

print("Number of duplicates after removing: ", df_bankC.duplicated().sum())
print("Length of dataset after removing duplicates", len(df_bankC))

df_bankC = df_bankC.dropna()
print("Length of dataset after removing NaN", len(df_bankC))

print(df_bankC.info())

ax = sns.countplot(data=df_bankC, x="income", hue="income")
plt.title('Count of target variable income')
plt.xlabel('Income')
plt.ylabel('Count')
for container in ax.containers:
    ax.bar_label(container)
plt.show()

ax = sns.countplot(data=df_bankC, x="income", hue="income", stat="percent")
plt.title('Percentage of Customers by Income Level')
plt.xlabel('Income')
plt.ylabel('Percentage (%)')
for container in ax.containers:
    ax.bar_label(container, fmt='%.2f%%')
plt.show()

sns.histplot(data=df_bankC, x="age")
plt.title('Distribution of age')
plt.xlabel('Age')
plt.ylabel('Count')
plt.show()

sns.boxplot(data=df_bankC, x='income', y='age')
plt.title('Age Distribution by Income Level')
plt.xlabel('Income')
plt.ylabel('Age')
plt.show()

sns.countplot(data=df_bankC, y='education', hue="income")
plt.title('Count of Education per income')
plt.xlabel('Income')
plt.ylabel('Education')
plt.show()

# --------- Preprocessing ---------
df_bankC_encoded = df_bankC.copy()

df_bankC_encoded["income"] = df_bankC_encoded['income'].map({'<=50K': 0, '>50K': 1})
print(df_bankC_encoded.head(5))
print(df_bankC_encoded["income"].value_counts())

df_bankC_encoded = pd.get_dummies(df_bankC_encoded, drop_first=True)
print(df_bankC_encoded.head(5))
print(df_bankC_encoded.info())

correlations = df_bankC_encoded.corr()['income'].sort_values(ascending=False)
print("Top 10 correlations: \n", correlations[:10])
print("Flop 10 correlations: \n", correlations[-10:])

from sklearn.model_selection import train_test_split

X = df_bankC_encoded.drop(columns=["income"])
print(X.info())
y= df_bankC_encoded["income"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Shape X_train: {X_train.shape}")
print(f"Shape X_test: {X_test.shape}")

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score, roc_auc_score

model = RandomForestClassifier(n_estimators=50, random_state=42, class_weight='balanced', max_depth=20)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

print("Result RandomForest:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("F1-Score:", f1_score(y_test, y_pred, average="macro"))
print("AUC-Score:", roc_auc_score(y_test, y_pred_proba))
print(classification_report(y_test, y_pred))

from sklearn.tree import DecisionTreeClassifier, export_text

tree_model = DecisionTreeClassifier(max_depth=15, random_state=42, min_samples_leaf=5, criterion='gini')
tree_model.fit(X_train, y_train)
y_pred_tree = tree_model.predict(X_test)
y_pred_tree_proba = tree_model.predict_proba(X_test)[:, 1]

print("Result DecisionTree:")
print("Accuracy:", accuracy_score(y_test, y_pred_tree))
print("F1-Score:", f1_score(y_test, y_pred_tree, average="macro"))
print("AUC-Score:", roc_auc_score(y_test, y_pred_tree_proba))
print(classification_report(y_test, y_pred_tree))

feature_names = list(X.columns)
rules = export_text(tree_model, feature_names=feature_names)
#print(rules)

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)

log_model = LogisticRegression(random_state=42, max_iter=1000, class_weight=None, C=1)
log_model.fit(X_train_scaled, y_train)
y_pred_log = log_model.predict(X_test_scaled)
y_pred_log_proba = log_model.predict_proba(X_test_scaled)[:, 1]

print("Result LogisticRegression:")
print("Accuracy:", accuracy_score(y_test, y_pred_log))
print("F1-Score:", f1_score(y_test, y_pred_log, average="macro"))
print("AUC-Score:", roc_auc_score(y_test, y_pred_log_proba))
print(classification_report(y_test, y_pred_log))

from sklearn.model_selection import GridSearchCV
# rf = RandomForestClassifier(random_state=42)

# param_grif_rf = {
#     'n_estimators': [50, 100, 200, 300],
#     'max_depth': [10, 20, None],
#     'class_weight': ['balanced', None]
# }

# grid_rf = GridSearchCV(rf, param_grif_rf, cv=3, scoring='f1_macro', n_jobs=-1)
# grid_rf.fit(X_train, y_train)
# print("Bester Random Forest F1-Score:", grid_rf.best_score_)
# print("Beste Parameter:", grid_rf.best_params_)

# tree = DecisionTreeClassifier(random_state=42)

# param_grid_tree = {
#     'max_depth': [3, 5, 7, 10, 15],
#     'criterion': ['gini', 'entropy'],
#     'min_samples_leaf': [1, 5, 10]
# }

# grid_tree = GridSearchCV(tree, param_grid_tree, cv=5, scoring='f1_macro', n_jobs=-1)
# grid_tree.fit(X_train, y_train)
# print("\nBester Decision Tree F1-Score:", grid_tree.best_score_)
# print("Beste Parameter:", grid_tree.best_params_)

# log_reg = LogisticRegression(random_state=42)

# param_grid_log = {
#     'C': [0.01, 0.1, 1, 10, 100],
#     'class_weight': ['balanced', None],
#     'max_iter': [1000, 2000]
# }

# grid_log = GridSearchCV(log_reg, param_grid_log, cv=5, scoring='f1_macro', n_jobs=-1)
# grid_log.fit(X_train_scaled, y_train)
# print("\nBeste Logistic Regression F1-Score:", grid_log.best_score_)
# print("Beste Parameter:", grid_log.best_params_)

from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(100, 50), 
                    activation='relu', 
                    solver='adam', 
                    max_iter=500, 
                    early_stopping=True,
                    random_state=42)

mlp.fit(X_train_scaled, y_train)

y_pred_mlp = mlp.predict(X_test_scaled)

print("Result Neural Network:")
print("Accuracy:", accuracy_score(y_test, y_pred_mlp))
print("F1-Score:", f1_score(y_test, y_pred_mlp, average='macro'))
print("AUC-Score:", roc_auc_score(y_test, mlp.predict_proba(X_test_scaled)[:, 1]))
print(classification_report(y_test, y_pred_mlp))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)

X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

print(f"Daten geladen. Features: {X_train_tensor.shape[1]}")

class BankNet(nn.Module):
    def __init__(self, input_dim):
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

    def forward(self, x):
        return self.network(x)

input_dim = X_train_tensor.shape[1]
model = BankNet(input_dim)

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 20 

print("Starte Training...")

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
    
    print(f"Epoch {epoch+1}/{epochs} - Loss: {running_loss/len(train_loader):.4f}")

print("Training beendet!")

from sklearn.metrics import classification_report, f1_score, roc_auc_score
model.eval()

with torch.no_grad(): 
    y_pred_prob = model(X_test_tensor)
    
    y_pred_prob_np = y_pred_prob.numpy()
    
    y_pred_class = (y_pred_prob_np > 0.5).astype(int)

acc = accuracy_score(y_test, y_pred_class)
f1 = f1_score(y_test, y_pred_class, average='macro')
auc = roc_auc_score(y_test, y_pred_prob_np)

print(f"PyTorch Neural Net Results:")
print(f"Accuracy: {acc}")
print(f"F1-Score (Macro): {f1}")
print(f"AUC-Score: {auc}")
print(classification_report(y_test, y_pred_class))