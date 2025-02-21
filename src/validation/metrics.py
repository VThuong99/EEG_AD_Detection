import numpy as np

"""Confusion matrix
|True Positive (TP) (1,1)|False Positive (FP) (0,1)|
|False Negative (FN) (1,0)|True Negative (TN) (0,0)|
"""
def accuracy(confusion_matrix: np.ndarray) -> float:
    """ Calculate accuracy from confusion matrix. """
    tp = confusion_matrix[1, 1]
    tn = confusion_matrix[0, 0]
    return (tp + tn) / np.sum(confusion_matrix) if np.sum(confusion_matrix) > 0 else 0

def sensitivity(confusion_matrix: np.ndarray) -> float:
    """Calculate sensitivity (recall) from confusion matrix."""
    tp = confusion_matrix[1, 1]
    fn = confusion_matrix[1, 0]
    return tp / (tp + fn) if (tp + fn) > 0 else 0  # Avoid division by zero

def specificity(confusion_matrix: np.ndarray) -> float:
    """Calculate specificity from confusion matrix."""
    tn = confusion_matrix[0, 0]
    fp = confusion_matrix[0, 1]
    return tn / (tn + fp) if (tn + fp) > 0 else 0  # Avoid division by zero

def f1(confusion_matrix: np.ndarray) -> float:
    """Calculate F1 score from confusion matrix."""
    prec = precision(confusion_matrix)
    rec = sensitivity(confusion_matrix)
    return 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0  # Avoid division by zero

def precision(confusion_matrix: np.ndarray) -> float:
    """Calculate precision from confusion matrix."""
    tp = confusion_matrix[1, 1]
    fp = confusion_matrix[0, 1]
    return tp / (tp + fp) if (tp + fp) > 0 else 0  # Avoid division by zero  