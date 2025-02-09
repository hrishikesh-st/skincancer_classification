import torch
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import numpy as np
import matplotlib.pyplot as plt

def evaluate_on_test(model, test_loader, device='cuda'):
    """
    Test set evaluation

    :param model: Model type
    :type model: torch.nn.Module
    :param test_loader: Test dataloader
    :type test_loader: torch.utils.data.DataLoader
    :param device: Device to evaluate, defaults to 'cuda'
    :type device: str, optional
    :return: Metrics dict
    :rtype: dict
    """
    model.eval()
    model.to(device)

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            preds = torch.sigmoid(outputs).squeeze(1) > 0.5  
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # convert to numpy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # accuracy 
    accuracy = (all_preds == all_labels).mean() * 100.0

    # precision, recall, F1
    precision = precision_score(all_labels, all_preds, pos_label=1, zero_division=0)
    recall = recall_score(all_labels, all_preds, pos_label=1, zero_division=0)
    f1 = f1_score(all_labels, all_preds, pos_label=1, zero_division=0)

    # confusion matrix
    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1])
    TN, FP, FN, TP = cm.ravel()
    specificity = TN / (TN + FP + 1e-10)

    # confusion matrix plot
    cm_fig = plot_confusion_matrix(cm, class_names=["Benign", "Malignant"])

    # metrics dict
    metrics_dict = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1score": f1,
        "cm": cm,
        "cm_fig": cm_fig
    }
    return metrics_dict

def plot_confusion_matrix(cm, class_names=["Class0", "Class1"]):
    """
    Plot confusion matrix

    :param cm: Confusion matrix
    :type cm: np.ndarray
    :param class_names: Class names, defaults to ["Class0", "Class1"]
    :type class_names: list, optional
    :return: Figure
    :rtype: matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(5,5))
    ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.set_title("Confusion Matrix")
    ax.set_xticks([0,1])
    ax.set_yticks([0,1])
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')

    # Annotate each cell
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > cm.max()/2. else "black")

    fig.tight_layout()
    return fig
