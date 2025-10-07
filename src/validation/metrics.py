import numpy as np

def accuracy(confusion_matrix: np.ndarray) -> float:
    total = np.sum(confusion_matrix)
    if total == 0:
        return 0.0
    return np.trace(confusion_matrix) / total


def precision(confusion_matrix: np.ndarray) -> float:
    """Precision for binary/macro-average."""
    n_classes = confusion_matrix.shape[0]
    precisions = []
    for i in range(n_classes):
        tp = confusion_matrix[i, i]
        fp = confusion_matrix[:, i].sum() - tp
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        precisions.append(prec)
    return np.mean(precisions)


def sensitivity(confusion_matrix: np.ndarray) -> float:
    """Recall / Sensitivity for binary/macro-average."""
    n_classes = confusion_matrix.shape[0]
    recalls = []
    for i in range(n_classes):
        tp = confusion_matrix[i, i]
        fn = confusion_matrix[i, :].sum() - tp
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        recalls.append(rec)
    return np.mean(recalls)


def specificity(confusion_matrix: np.ndarray) -> float:
    """Specificity for binary/macro-average."""
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
    """F1-score for binary/macro-average."""
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
    # print(f"F1-scores per class: {f1s}")
    return np.mean(f1s)
