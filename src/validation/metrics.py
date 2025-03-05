import numpy as np

"""Confusion matrix
|True Positive (TP) (1,1)|False Positive (FP) (0,1)|
|False Negative (FN) (1,0)|True Negative (TN) (0,0)|
"""
def accuracy(confusion_matrix: np.ndarray) -> float:
    """ Calculate accuracy from confusion matrix, handling cases with missing classes. """
    if confusion_matrix.shape[0] < 2 or confusion_matrix.shape[1] < 2:
        # Handle cases where confusion_matrix is too small (e.g., only one class predicted)
        tn = confusion_matrix[0, 0] if confusion_matrix.shape[0] > 0 and confusion_matrix.shape[1] > 0 else 0
        tp = 0 # If class 1 is missing in confusion_matrix, assume tp = 0 and fn = 0
    else: # Normal case: confusion_matrix has full size
        tp = confusion_matrix[1, 1]
        tn = confusion_matrix[0, 0]
    return (tp + tn) / np.sum(confusion_matrix) if np.sum(confusion_matrix) > 0 else 0

def sensitivity(confusion_matrix: np.ndarray) -> float:
    """Calculate sensitivity (recall) from confusion matrix, handling missing classes."""
    if confusion_matrix.shape[0] < 2 or confusion_matrix.shape[1] < 2:
        tp = 0 # Assume tp = 0 and fn = 0 if class 1 is missing
        fn = 0
    else:
        tp = confusion_matrix[1, 1]
        fn = confusion_matrix[1, 0]
    return tp / (tp + fn) if (tp + fn) > 0 else 0  # Avoid division by zero

def specificity(confusion_matrix: np.ndarray) -> float:
    """Calculate specificity from confusion matrix, handling missing classes."""
    if confusion_matrix.shape[0] < 2 or confusion_matrix.shape[1] < 2:
        tn = confusion_matrix[0, 0] if confusion_matrix.shape[0] > 0 and confusion_matrix.shape[1] > 0 else 0
        fp = 0 # Assume tn = 0 and fp = 0 if class 0 is missing
    else:
        tn = confusion_matrix[0, 0]
        fp = confusion_matrix[0, 1]
    return tn / (tn + fp) if (tn + fp) > 0 else 0  # Avoid division by zero

def f1(confusion_matrix: np.ndarray) -> float:
    """Calculate F1 score from confusion matrix, handling missing classes. """
    prec = precision(confusion_matrix)
    rec = sensitivity(confusion_matrix)
    return 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0  # Avoid division by zero

def precision(confusion_matrix: np.ndarray) -> float:
    """Calculate precision from confusion matrix, handling missing classes. """
    if confusion_matrix.shape[0] < 2 or confusion_matrix.shape[1] < 2:
        tp = 0
        fp = 0
    else:
        tp = confusion_matrix[1, 1]
        fp = confusion_matrix[0, 1]
    return tp / (tp + fp) if (tp + fp) > 0 else 0  # Avoid division by zero