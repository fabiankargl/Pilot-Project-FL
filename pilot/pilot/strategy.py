from typing import List, Any
from flwr.serverapp.strategy import FedAvg, FedProx

class HistoryStrategy:
    """
    A mixin class for Flower strategies to record metrics.

    This class provides functionality to track both global (aggregated) and local
    (per-client) metrics across federated learning rounds. It is designed to be
    inherited by a primary Flower strategy class.

    Attributes:
        global_history (dict): A dictionary storing the history of aggregated metrics
            for each round. Includes training loss, evaluation loss, and various
            evaluation metrics (accuracy, F1, AUC).
        local_history (dict): A dictionary storing the history of metrics reported by
            individual clients for each round. Captures both training and evaluation
            results on a per-client basis.
        current_round (int): The current federated learning round number.
    """
    def __init__(self):
        self.global_history = {
            "round": [],
            "train_loss": [],
            "eval_loss": [],
            "eval_acc": [],
            "eval_f1": [],
            "eval_auc": [],
        }
        self.local_history = {
            "round": [],
            "client_results": []  
        }
        self.current_round = 0
        
    def _calculate_weighted_metric(self, results: List[Any], metric_name: str) -> float:
        if not results:
            return 0.0
            
        total_examples = sum(r.content["metrics"]["num-examples"] for r in results)
        
        if total_examples == 0:
            return 0.0
            
        weighted_sum = sum(
            r.content["metrics"][metric_name] * r.content["metrics"]["num-examples"]
            for r in results
        )
        return float(weighted_sum / total_examples)
        
    def _update_train_history(self, train_results: List[Any]) -> None:
        if train_results:
            total_examples = sum(r.content["metrics"]["num-examples"] for r in train_results)
            avg_train_loss = sum(
                r.content["metrics"]["train_loss"] * r.content["metrics"]["num-examples"]
                for r in train_results
            ) / total_examples
            
            self.global_history["train_loss"].append(float(avg_train_loss))
            
            local_train_results = []
            for idx, r in enumerate(train_results):
                client_metrics = {
                    "client_id": idx, 
                    "train_loss": float(r.content["metrics"]["train_loss"]),
                    "num_examples": int(r.content["metrics"]["num-examples"])
                }
                local_train_results.append(client_metrics)
            
            if not self.local_history["round"] or self.local_history["round"][-1] != self.current_round + 1:
                self.local_history["round"].append(self.current_round + 1)
                self.local_history["client_results"].append({
                    "train": local_train_results,
                    "eval": []  
                })
            else:
                self.local_history["client_results"][-1]["train"] = local_train_results
                 
    def _update_eval_history(self, evaluate_results: List[Any]) -> None:
        if evaluate_results:
            self.current_round += 1
            self.global_history["round"].append(self.current_round)
            
            self.global_history["eval_loss"].append(self._calculate_weighted_metric(evaluate_results, "eval_loss"))
            self.global_history["eval_acc"].append(self._calculate_weighted_metric(evaluate_results, "eval_acc"))
            self.global_history["eval_f1"].append(self._calculate_weighted_metric(evaluate_results, "eval_f1"))
            self.global_history["eval_auc"].append(self._calculate_weighted_metric(evaluate_results, "eval_auc"))
            
            local_eval_results = []
            for idx, r in enumerate(evaluate_results):
                client_metrics = {
                    "client_id": idx, 
                    "eval_loss": float(r.content["metrics"]["eval_loss"]),
                    "eval_acc": float(r.content["metrics"]["eval_acc"]),
                    "eval_f1": float(r.content["metrics"]["eval_f1"]),
                    "eval_auc": float(r.content["metrics"]["eval_auc"]),
                    "num_examples": int(r.content["metrics"]["num-examples"])
                }
                local_eval_results.append(client_metrics)
            
            if self.local_history["client_results"]:
                self.local_history["client_results"][-1]["eval"] = local_eval_results

class FedAvgWithHistory(FedAvg, HistoryStrategy):
    def __init__(self, *args, **kwargs):
        FedAvg.__init__(self, *args, **kwargs)
        HistoryStrategy.__init__(self)
    
    def aggregate_train(self, grid, train_results):
        result = super().aggregate_train(grid, train_results)
        
        self._update_train_history(train_results)
        
        return result
    
    def aggregate_evaluate(self, grid, evaluate_results):
        result = super().aggregate_evaluate(grid, evaluate_results)
        
        self._update_eval_history(evaluate_results)
        
        return result
    
class FedProxWithHistory(FedProx, HistoryStrategy):
    def __init__(self, *args, **kwargs):
        FedProx.__init__(self, *args, **kwargs)
        HistoryStrategy.__init__(self)
        
    def aggregate_train(self, grid, train_results):
        result = super().aggregate_train(grid, train_results)

        self._update_train_history(train_results)
        
        return result
    
    def aggregate_evaluate(self, grid, evaluate_results):
        result = super().aggregate_evaluate(grid, evaluate_results)
        
        self._update_eval_history(evaluate_results)
        
        return result