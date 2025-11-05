import torch
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import List, Dict, Tuple
import logging

logging.basicConfig(level=logging.INFO)


def evaluate_model(model: torch.nn.Module, test_loader: 'torch.utils.data.DataLoader',
                   cat_cardinalities: List[int], device: torch.device, target_mapping: Dict[str, int]
                  ) -> Tuple[List[int], List[int]]:
    """
    Evaluate the model on a test set and plot the confusion matrix.

    Args:
        model (nn.Module): Trained PyTorch model.
        test_loader (DataLoader): DataLoader for test dataset.
        cat_cardinalities (List[int]): Cardinalities for categorical embeddings.
        device (torch.device): Device to run evaluation on.
        target_mapping (Dict[str, int]): Mapping from target labels to numeric indices.

    Returns:
        Tuple[List[int], List[int]]: y_true and y_pred lists.
    """
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for (X_num, X_cat, cat_mask), y in test_loader:
            X_num, X_cat, cat_mask, y = X_num.to(device), X_cat.to(device), cat_mask.to(device), y.to(device)
            for i, n_cat in enumerate(cat_cardinalities):
                X_cat[:, :, i] = torch.clamp(X_cat[:, :, i], 0, n_cat-1)
            preds = model(X_num, X_cat, cat_mask)
            predicted = preds.argmax(1)
            y_true.extend(y.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    acc = accuracy_score(y_true, y_pred)
    logging.info(f"Final Test Accuracy: {acc:.2%}")
    logging.info(f"Per-class metrics:\n{classification_report(y_true, y_pred, target_names=[str(k) for k in target_mapping.keys()])}")

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=[str(k) for k in target_mapping.keys()],
                yticklabels=[str(k) for k in target_mapping.keys()])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix on Test Set')
    plt.show()

    return y_true, y_pred

def one_vs_all_accuracy(y_true: List[int], y_pred: List[int], target_label: str, class_lookup: Dict[str, int]) -> float:
    """
    Compute one-vs-all accuracy for a specific target class.

    Args:
        y_true (List[int]): True labels.
        y_pred (List[int]): Predicted labels.
        target_label (str): Label to compute one-vs-all accuracy for.
        class_lookup (Dict[str, int]): Mapping from label names to numeric indices.

    Returns:
        float: Accuracy for the target label vs all others.
    """
    target_class = class_lookup[target_label]
    y_true_binary = (np.array(y_true) == target_class).astype(int)
    y_pred_binary = (np.array(y_pred) == target_class).astype(int)
    accuracy = (y_true_binary == y_pred_binary).mean()
    logging.info(f"Accuracy for '{target_label}' vs all others: {accuracy:.2%}")
    return accuracy
