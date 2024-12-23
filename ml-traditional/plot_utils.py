import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay, 
    precision_score, 
    recall_score, 
    f1_score, 
    roc_curve, 
    roc_auc_score, 
    precision_recall_curve, 
    auc, 
    brier_score_loss
)
from sklearn.calibration import calibration_curve
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, f1_score
import seaborn as sns

from sklearn.metrics import roc_curve, roc_auc_score

def plot_classification_metrics(y_test, y_pred, class_labels=["NEGATIVE", "POSITIVE"]):
    """
    Plots a confusion matrix, comparison of true and predicted class frequencies, 
    and class-specific performance metrics (Precision, Recall, F1-Score).

    Parameters:
        y_test (array-like): True labels.
        y_pred (array-like): Predicted labels.
        class_labels (list): List of class names.

    Returns:
        None
    """
    # Compute confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)

    # Compute class frequencies
    unique, real_counts = np.unique(y_test, return_counts=True)
    _, pred_counts = np.unique(y_pred, return_counts=True)

    # Compute metrics for each class
    precision_scores = []
    recall_scores = []
    f1_scores = []

    for i, label in enumerate(class_labels):
        precision_scores.append(precision_score(y_test, y_pred, pos_label=i))
        recall_scores.append(recall_score(y_test, y_pred, pos_label=i))
        f1_scores.append(f1_score(y_test, y_pred, pos_label=i))

    # Plot settings
    x = np.arange(len(class_labels))
    width = 0.25  # Bar width

    # Create a figure with three subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Confusion Matrix subplot
    disp.plot(cmap=plt.cm.Blues, ax=axes[0], colorbar=False)
    axes[0].set_title('Confusion Matrix')

    # Comparison of true and predicted class frequencies subplot
    axes[1].bar(x - width / 2, real_counts, width, label='True Classes', color='blue', alpha=0.7)
    axes[1].bar(x + width / 2, pred_counts, width, label='Predicted Classes', color='red', alpha=0.7)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(class_labels)
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Comparison of True and Predicted Classes')
    axes[1].legend()
    axes[1].grid(axis='y', linestyle='--', alpha=0.7)

    # Class-Specific Performance Metrics subplot
    axes[2].bar(x - width, precision_scores, width=width, label='Precision', color='blue', alpha=0.7)
    axes[2].bar(x, recall_scores, width=width, label='Recall', color='orange', alpha=0.7)
    axes[2].bar(x + width, f1_scores, width=width, label='F1-Score', color='green', alpha=0.7)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(class_labels)
    axes[2].set_ylabel('Score')
    axes[2].set_title('Class-Specific Performance Metrics')
    axes[2].legend()
    axes[2].grid(axis='y', linestyle='--', alpha=0.7)

    # Adjust layout
    plt.tight_layout()
    plt.show()

def plot_grid_search_results(grid_search, param_grid):
    """
    Extracts results from a GridSearchCV object and plots the performance of mean test scores 
    against each hyperparameter.

    Parameters:
        grid_search (GridSearchCV): A fitted GridSearchCV object.
        param_grid (dict): The parameter grid used in the GridSearchCV.

    Returns:
        pd.DataFrame: A DataFrame containing parameters and their mean test scores.
    """
    # Extract results
    mean_scores = grid_search.cv_results_['mean_test_score']
    params = grid_search.cv_results_['params']

    # Create a DataFrame for analysis
    df_results = pd.DataFrame(params)
    df_results['mean_test_score'] = mean_scores

    # Plot performance for each parameter
    param_names = list(param_grid.keys())

    plt.figure(figsize=(15, 10))
    for i, param in enumerate(param_names, 1):
        plt.subplot((len(param_names) + 1) // 2, 2, i)  # Adjust layout for up to 2 plots per row
        unique_values = sorted(df_results[param].unique())
        scores = [df_results[df_results[param] == val]['mean_test_score'].mean() for val in unique_values]
        plt.plot(unique_values, scores, marker='o', linestyle='-', color='purple')
        plt.xlabel(param)
        plt.ylabel('Mean Accuracy')
        plt.title(f'Accuracy vs {param}')
        plt.grid(True)
        plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.show()

def plot_roc_curve(y_test, y_probs, model_name="KNN Model"):
    """
    Plots the ROC curve and calculates the AUROC value.

    Parameters:
        y_test (array-like): True binary labels.
        y_probs (array-like): Predicted probabilities for the positive class.
        model_name (str): Name of the model for display purposes.
    """
    # Calculate the ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, y_probs)
    
    # Calculate the AUROC
    auroc = roc_auc_score(y_test, y_probs)
    
    # Plot the ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'{model_name} (AUROC = {auroc:.2f})', color='darkorange', lw=2)
    plt.plot([0, 1], [0, 1], 'k--', lw=2, label="Random Guessing")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve', fontsize=15)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(alpha=0.5)
    plt.show()

def plot_auroc_heatmap(datasets, auroc_scores, model_name):
    """
    Plots a heatmap of AUROC scores for different datasets and models.

    Parameters:
        datasets (list): List of dataset names (for the y-axis).
        models (list): List of model names (for the x-axis).
        auroc_scores (2D list): 2D list of AUROC scores, where each row corresponds
                                to a dataset and each column corresponds to a model.
    """
    # Create a DataFrame for the heatmap
    heatmap_data = pd.DataFrame(auroc_scores, index=datasets, columns=datasets)

    # Plot the heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="YlGnBu", cbar=True, linewidths=0.5)
    plt.title(f"{model_name} AUROC Heatmap", fontsize=16)
    plt.xlabel("x", fontsize=12)
    plt.ylabel("y", fontsize=12)
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.show()