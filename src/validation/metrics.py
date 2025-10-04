# import numpy as np

# """Confusion matrix
# |True Positive (TP) (1,1)|False Positive (FP) (0,1)|
# |False Negative (FN) (1,0)|True Negative (TN) (0,0)|
# """
# def accuracy(confusion_matrix: np.ndarray) -> float:
#     """ Calculate accuracy from confusion matrix, handling cases with missing classes. """
#     if confusion_matrix.shape[0] < 2 or confusion_matrix.shape[1] < 2:
#         # Handle cases where confusion_matrix is too small (e.g., only one class predicted)
#         tn = confusion_matrix[0, 0] if confusion_matrix.shape[0] > 0 and confusion_matrix.shape[1] > 0 else 0
#         tp = 0 # If class 1 is missing in confusion_matrix, assume tp = 0 and fn = 0
#     else: # Normal case: confusion_matrix has full size
#         tp = confusion_matrix[1, 1]
#         tn = confusion_matrix[0, 0]
#     return (tp + tn) / np.sum(confusion_matrix) if np.sum(confusion_matrix) > 0 else 0

# def sensitivity(confusion_matrix: np.ndarray) -> float:
#     """Calculate sensitivity (recall) from confusion matrix, handling missing classes."""
#     if confusion_matrix.shape[0] < 2 or confusion_matrix.shape[1] < 2:
#         tp = 0 # Assume tp = 0 and fn = 0 if class 1 is missing
#         fn = 0
#     else:
#         tp = confusion_matrix[1, 1]
#         fn = confusion_matrix[1, 0]
#     return tp / (tp + fn) if (tp + fn) > 0 else 0  # Avoid division by zero

# def specificity(confusion_matrix: np.ndarray) -> float:
#     """Calculate specificity from confusion matrix, handling missing classes."""
#     if confusion_matrix.shape[0] < 2 or confusion_matrix.shape[1] < 2:
#         tn = confusion_matrix[0, 0] if confusion_matrix.shape[0] > 0 and confusion_matrix.shape[1] > 0 else 0
#         fp = 0 # Assume tn = 0 and fp = 0 if class 0 is missing
#     else:
#         tn = confusion_matrix[0, 0]
#         fp = confusion_matrix[0, 1]
#     return tn / (tn + fp) if (tn + fp) > 0 else 0  # Avoid division by zero

# def f1(confusion_matrix: np.ndarray) -> float:
#     """Calculate F1 score from confusion matrix, handling missing classes. """
#     prec = precision(confusion_matrix)
#     rec = sensitivity(confusion_matrix)
#     return 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0  # Avoid division by zero

# def precision(confusion_matrix: np.ndarray) -> float:
#     """Calculate precision from confusion matrix, handling missing classes. """
#     if confusion_matrix.shape[0] < 2 or confusion_matrix.shape[1] < 2:
#         tp = 0
#         fp = 0
#     else:
#         tp = confusion_matrix[1, 1]
#         fp = confusion_matrix[0, 1]
#     return tp / (tp + fp) if (tp + fp) > 0 else 0  # Avoid division by zero

import numpy as np

def accuracy(confusion_matrix: np.ndarray) -> float:
    """Accuracy: đúng / tổng (binary hoặc multi-class)."""
    total = np.sum(confusion_matrix)
    if total == 0:
        return 0.0
    return np.trace(confusion_matrix) / total


def precision(confusion_matrix: np.ndarray) -> float:
    """Precision (binary hoặc macro-average)."""
    n_classes = confusion_matrix.shape[0]
    precisions = []
    for i in range(n_classes):
        tp = confusion_matrix[i, i]
        fp = confusion_matrix[:, i].sum() - tp
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        precisions.append(prec)
    return np.mean(precisions)


def sensitivity(confusion_matrix: np.ndarray) -> float:
    """Recall / Sensitivity (binary hoặc macro-average)."""
    n_classes = confusion_matrix.shape[0]
    recalls = []
    for i in range(n_classes):
        tp = confusion_matrix[i, i]
        fn = confusion_matrix[i, :].sum() - tp
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        recalls.append(rec)
    return np.mean(recalls)


def specificity(confusion_matrix: np.ndarray) -> float:
    """Specificity (binary hoặc macro-average)."""
    n_classes = confusion_matrix.shape[0]
    specs = []
    for i in range(n_classes):
        tp = confusion_matrix[i, i]
        fn = confusion_matrix[i, :].sum() - tp
        fp = confusion_matrix[:, i].sum() - tp
        tn = confusion_matrix.sum() - (tp + fn + fp)
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        specs.append(spec)
    return np.mean(specs)


def f1(confusion_matrix: np.ndarray) -> float:
    """F1-score (binary hoặc macro-average)."""
    n_classes = confusion_matrix.shape[0]
    f1s = []
    for i in range(n_classes):
        tp = confusion_matrix[i, i]
        fn = confusion_matrix[i, :].sum() - tp
        fp = confusion_matrix[:, i].sum() - tp
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_val = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        f1s.append(f1_val)
    return np.mean(f1s)
