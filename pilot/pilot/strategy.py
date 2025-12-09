from flwr.serverapp.strategy import FedAvg

class FedAvgWithHistory(FedAvg):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
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
    
    def aggregate_train(self, grid, train_results):
        result = super().aggregate_train(grid, train_results)
        
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
        
        return result
    
    def aggregate_evaluate(self, grid, evaluate_results):
        result = super().aggregate_evaluate(grid, evaluate_results)
        
        if evaluate_results:
            self.current_round += 1
            self.global_history["round"].append(self.current_round)
            
            total_examples = sum(r.content["metrics"]["num-examples"] for r in evaluate_results)
            
            avg_eval_loss = sum(
                r.content["metrics"]["eval_loss"] * r.content["metrics"]["num-examples"]
                for r in evaluate_results
            ) / total_examples
            
            avg_eval_acc = sum(
                r.content["metrics"]["eval_acc"] * r.content["metrics"]["num-examples"]
                for r in evaluate_results
            ) / total_examples
            
            avg_eval_f1 = sum(
                r.content["metrics"]["eval_f1"] * r.content["metrics"]["num-examples"]
                for r in evaluate_results
            ) / total_examples
            
            avg_eval_auc = sum(
                r.content["metrics"]["eval_auc"] * r.content["metrics"]["num-examples"]
                for r in evaluate_results
            ) / total_examples
            
            self.global_history["eval_loss"].append(float(avg_eval_loss))
            self.global_history["eval_acc"].append(float(avg_eval_acc))
            self.global_history["eval_f1"].append(float(avg_eval_f1))
            self.global_history["eval_auc"].append(float(avg_eval_auc))
            
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
        
        return result