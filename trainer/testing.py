import torch
from sklearn.metrics import classification_report


def testing_model(dataloader, model, num_labels=12, device="cpu"):
    total_labels = []
    total_predictions = []

    clr_labels = [str(i) for i in range(num_labels)]

    model.eval()
    #model.to("cpu")
    for batch in dataloader:
        batch.to("cuda")
        predictions = model(batch)
        #mientras = cambio(batch.y)

        #predictions.to("cpu")
        total_labels.extend(batch.y.tolist())  # batch.y.tolist() batch.y.tolist()

        total_predictions.extend(torch.max(predictions, axis=1).indices.tolist())
    print(classification_report(total_labels, total_predictions, target_names=clr_labels))


def cambio(lista):
    labels = {6: 0, 7: 1, 8: 2, 9: 3, 10: 4, 11: 5}  # 9:3, 10:4, 11:5
    lb = []
    # print(lista)
    for l in lista:
        lb.append(labels[int(l)])
    return torch.tensor(lb)#.to("cpu")