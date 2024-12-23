import torch
from cnn import Net
from matplotlib import colors
import matplotlib.pyplot as plt
import numpy as np

def save_model(model_path, model):
    torch.save(model.state_dict(), model_path)

def load_model(model_path):
    try:
        model = Net()
        model.load_state_dict(torch.load(model_path, weights_only=True))
        model.eval()
        return model
    except FileNotFoundError:
        return None

def display_metrics(model, dataset, data):
    print(f'\nModel trained on\t{model}\nModel tested on\t\t{dataset}\nMetrics:\n\tAuroc:\t\t{data[0]}\n\tAccuracy:\t{data[1]}\n\tPrecision:\t{data[2]}\n\tRecall:\t\t{data[3]}\n\tF1:\t\t{data[4]}')

def plot_graphs(datasetName, roc_auc, fpr, tpr, disp, values):
    fig, ax = plt.subplots(ncols=3, figsize=(20, 4))  # ax is a single Axes object
    metrics = ['accuracy', 'precision', 'recall', 'f1']

    ax[0].bar(metrics, values, color='skyblue')
    ax[0].set_title('Performance Metrics')
    ax[0].set_ylim(0, 1)  # Assuming all metrics are between 0 and 1
    ax[0].set_ylabel('Score')

    disp.plot(cmap=plt.cm.Blues, ax=ax[1], colorbar=False)
    ax[1].set_title('Confusion Matrix')

    # Use ax directly instead of ax[0]
    ax[2].plot(fpr, tpr, lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    ax[2].plot([0, 1], [0, 1], 'k--', lw=1, label='Random Guess')
    ax[2].set_title(datasetName)
    ax[2].set_ylabel('True Positive Rate')
    ax[2].set_xlabel('False Positive Rate')
    ax[2].legend(loc='lower right')

    # Show the plot
    plt.show()

def plot_graph(name, data, center):
    plt.figure(figsize=(5, 3))
    divnorm=colors.TwoSlopeNorm(vmin=.5, vcenter=center, vmax=1)
    heatmap = plt.imshow(data, cmap="coolwarm_r", aspect="auto", norm=divnorm)
    
    cbar = plt.colorbar(heatmap)
    cbar.set_label(name)
    
    plt.xticks(ticks=[0, 1, 2], labels=["CXr", "CheX", "RSNA"])
    plt.yticks(ticks=[0, 1, 2], labels=["CXr", "CheX", "RSNA"])
    
    for (i, j), val in np.ndenumerate(data):
        plt.text(j, i, f"{val:.2f}", ha='center', va='center', color='white')
    
    plt.title(name)
    
    plt.tight_layout()
    plt.show()