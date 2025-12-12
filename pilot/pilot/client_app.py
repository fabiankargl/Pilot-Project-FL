import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from pilot.task import IncomeNet, LogisticRegression, load_data, train as train_fn, test as test_fn

app = ClientApp()

@app.train()
def train(msg: Message, context: Context):
    partition_id = context.node_config["partition-id"]
    model_type = context.run_config.get("model-type", "nn")
    trainloader, _ = load_data(partition_id=partition_id,
                               num_partitions=0)
    sample_batch = next(iter(trainloader))
    input_dim = sample_batch[0].shape[1]
    
    if model_type == "logreg":
        model = LogisticRegression(input_dim=input_dim)
    else:
        model = IncomeNet(input_dim=input_dim)
        
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    lr = msg.content["config"].get("lr", 0.01)
    epochs = context.run_config.get("local-epochs", 1)
    
    proximal_mu = msg.content["config"].get("proximal_mu", 0.0)

    # Call the training function
    train_loss = train_fn(
        model,
        trainloader,
        epochs,
        lr,
        device,
        proximal_mu
    )

    # Construct and return reply Message
    model_record = ArrayRecord(model.state_dict())
    metrics = {
        "train_loss": train_loss,
        "num-examples": len(trainloader.dataset),
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"arrays": model_record, "metrics": metric_record})
    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context):
    partition_id = context.node_config["partition-id"]
    model_type = context.run_config.get("model-type", "nn")
    _, valloader = load_data(partition_id=partition_id, 
                             num_partitions=0)
    sample_batch = next(iter(valloader))
    input_dim = sample_batch[0].shape[1]

    if model_type == "logreg":
        model = LogisticRegression(input_dim=input_dim)
    else:
        model = IncomeNet(input_dim=input_dim)
        
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Call the evaluation function
    eval_loss, eval_acc, extended_metrics = test_fn(
        model,
        valloader,
        device,
    )

    # Construct and return reply Message
    metrics = {
        "eval_loss": eval_loss,
        "eval_acc": eval_acc,
        "eval_f1": extended_metrics["f1_macro"],
        "eval_auc": extended_metrics["auc"],
        "num-examples": len(valloader.dataset),
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"metrics": metric_record})
    return Message(content=content, reply_to=msg)
