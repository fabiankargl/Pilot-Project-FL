from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from pilot.utils import load_preprocessing
from typing import Tuple, Dict

class IncomeNet(nn.Module):
    def __init__(self, input_dim: int):
        super(IncomeNet, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(in_features=input_dim,
                      out_features=256),
            nn.BatchNorm1d(num_features=256),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            
            nn.Linear(in_features=256,
                      out_features=128),
            nn.BatchNorm1d(num_features=128),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            
            nn.Linear(in_features=128,
                      out_features=64),
            nn.ReLU(),
            
            nn.Linear(in_features=64,
                      out_features=1),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)
    
class LogisticRegression(nn.Module):
    def __init__(self, input_dim: int):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_dim , 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.sigmoid(self.linear(x))


def load_data(partition_id: int, 
              num_partitions: int) -> Tuple[DataLoader, DataLoader]:
    data_dir = Path(__file__).parent.parent.parent / "data"
    data_filepath = str(data_dir / "BankA.csv")
    if partition_id == 0:
        data_filepath = str(data_dir / "BankA.csv")
    elif partition_id == 1:
        data_filepath = str(data_dir / "BankB.csv")
    elif partition_id == 2:
        data_filepath = str(data_dir / "BankC.csv")
    else:
        raise ValueError("Only 3 partitions (BankA, BankB, BankC) available.")
        
    X_train, X_test, y_train, y_test = load_preprocessing(filepath=data_filepath)
    
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)
    
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    
    trainloader = DataLoader(dataset=train_dataset,
                             batch_size=64,
                             shuffle=True)
    testloader = DataLoader(dataset=test_dataset,
                            batch_size=64,
                            shuffle=False)
    
    return trainloader, testloader

def train(model: nn.Module, 
          trainloader: DataLoader, 
          epochs: int, 
          lr: float, 
          device: torch.device,
          proximal_mu: float = 0.0):
    model.to(device)  
    criterion = nn.BCELoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    global_params = [param.detach().clone() for param in model.parameters()]
    
    model.train()
    running_loss = 0.0
    for _ in range(epochs):
        for inputs, labels in trainloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            preds = model(inputs)
            loss = criterion(preds, labels)
            
            if proximal_mu > 0.0:
                proximal_term = 0.0
                for local_weights, global_weights in zip(model.parameters(), global_params):
                    proximal_term += (local_weights - global_weights).norm(2)**2
                loss += (proximal_mu / 2) * proximal_term
            
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
    avg_trainloss = running_loss / (len(trainloader) * epochs)
    return avg_trainloss

def test(model: nn.Module, 
         testloader: DataLoader, 
         device: torch.device) -> Tuple[float, float, Dict[str, float]]:
    model.to(device)
    criterion = nn.BCELoss().to(device)
    
    all_labels = []
    all_preds = []
    all_probs = []
    
    model.eval()
    loss = 0.0
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            preds = model(inputs)
            loss += criterion(preds, labels).item()
            
            all_probs.extend(preds.cpu().numpy())
            
            predicted = (preds > 0.5).float()
            all_preds.extend(predicted.cpu().numpy())
            
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = loss / len(testloader) 
    
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = 0.0
    
    metrics = {
        "accuracy": accuracy,
        "f1_macro": f1,
        "auc": auc
    }
    
    return avg_loss, accuracy, metrics
