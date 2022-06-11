import torch
from sklearn.metrics import classification_report


def testing_model(dataloader, model, num_labels=12, device="cpu"):
    total_labels = []
    total_predictions = []

    clr_labels = [str(i) for i in range(num_labels)]
    print(next(model.parameters()).device)
    model.to("cuda")
    model.eval()
    print(next(model.parameters()).device)

    for batch in dataloader:
        batch.to("cuda")
        with torch.no_grad():
            predictions = model(batch)
            total_labels.extend(batch.y.tolist())
            total_predictions.extend(torch.max(predictions, axis=1).indices.tolist())
        batch.to("cpu")
        torch.cuda.empty_cache()
    print(classification_report(total_labels, total_predictions, target_names=clr_labels))
