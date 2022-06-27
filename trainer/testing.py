import torch
from sklearn.metrics import classification_report


def testing_model(dataloader, model, num_labels=12):
    total_labels = []
    total_predictions = []

    clr_labels = [str(i) for i in range(num_labels)]
    model.to("cuda")
    model.eval()

    for batch in dataloader:
        batch.to("cuda")
        with torch.no_grad():
            predictions = model(batch)
            total_labels.extend(batch.y.tolist())
            total_predictions.extend(torch.max(predictions, axis=1).indices.tolist())
        batch.to("cpu")
        torch.cuda.empty_cache()
    print(classification_report(total_labels, total_predictions, target_names=clr_labels))

def testing_one(batch, model):
    model.to("cuda")
    model.eval()
    batch.to("cuda")

    print("original: ", batch.y)
    with torch.no_grad():
        predictions = model(batch)
        batch.to("cpu")
    _, indices = torch.max(predictions, dim=1)
    print("Prediction: ", indices)


