import torch
import torch.nn.functional as F
import numpy as np
from src.data.MergedDataset import to_device
from src.net.cnn import Net
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    roc_auc_score
)

def save_model(model_path, model):
    torch.save(model.state_dict(), model_path)

def load_model(model_path, conv_size, layer_size, dropout):
    try:
        model = Net(conv_size, layer_size, dropout)
        model.load_state_dict(torch.load(model_path, weights_only=True))
        model.eval()
        return model
    except FileNotFoundError:
        raise FileNotFoundError(f"Model file not found: {model_path}")

def train(wandb, net, device, train_dl, learning_rate, epochs, weight):
    net.train()
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    weight = to_device(torch.FloatTensor(weight), device)

    net.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(train_dl, 0):
            inputs, labels = data
            inputs, labels = to_device(inputs, device), to_device(labels, device)
            optimizer.zero_grad()
            outputs = net(inputs)

            loss = F.cross_entropy(outputs, labels, weight=weight)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        wandb.log({"epoch": epoch, "loss": (running_loss / len(train_dl))})

def test(wandb, net, device, datasetName, test_dl):
    net.eval()
    y_test = []
    prob = []
    with torch.no_grad():
        for data in test_dl:
            images, labels = data
            images, labels = to_device(images, device), to_device(labels, device)

            outputs = net(images)
            y_test.extend(labels.cpu().numpy())
            prob.extend(torch.nn.functional.softmax(outputs.cpu(), dim=1)[:, 1])
        fpr, tpr, thresholds = roc_curve(y_test, prob)
        roc_auc = roc_auc_score(y_test, prob)

        distances = np.sqrt(fpr ** 2 + (1 - tpr) ** 2)
        best_threshold = thresholds[np.argmin(distances)]
        new_preds = [1 if score > best_threshold else 0 for score in prob]

        accuracy = accuracy_score(y_test, new_preds)
        precision = precision_score(y_test, new_preds)
        recall = recall_score(y_test, new_preds)
        f1 = f1_score(y_test, new_preds)
        wandb.log({datasetName+"-AUROC": roc_auc, datasetName+"-Accuracy": accuracy, datasetName+"-Precision": precision, datasetName+"-Recall": recall, datasetName+"-F1": f1})