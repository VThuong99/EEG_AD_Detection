import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

from src.validation.metrics import accuracy, sensitivity, specificity, f1, precision # Assuming metrics.py is in src/validation

METRIC_FUNCTIONS = {
    'accuracy': accuracy,
    'sensitivity': sensitivity,
    'specificity': specificity,
    'f1': f1,
    'precision': precision
}

def _prepare_data(features, targets, indices, flatten_final, use_subject_normalization=True):
    from sklearn.preprocessing import StandardScaler
    X_list, y_list = [], []
    for idx in indices:
        subject_X = features[idx]
        subject_y = targets[idx]
        if use_subject_normalization:
            scaler = StandardScaler()
            subject_X = scaler.fit_transform(subject_X.reshape(-1, subject_X.shape[-1])).reshape(subject_X.shape)
        X_list.append(subject_X)
        # Broadcast scalar target to match number of epochs
        if np.isscalar(subject_y):
            subject_y = subject_y * np.ones(subject_X.shape[0], dtype=np.int32)
        y_list.append(subject_y)
    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    if flatten_final:
        X = X.reshape(X.shape[0], -1)
    return X, y

def _plot_history(history):
    """
    Plot training and validation loss/accuracy.
    
    Args:
        history (dict): Dictionary containing 'train_loss', 'val_loss', 'train_acc', 'val_acc' lists.
    """
    epochs = range(1, len(history['train_loss']) + 1)
    plt.figure(figsize=(12, 5))
    
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], 'b', label='Training loss')
    plt.plot(epochs, history['val_loss'], 'r', label='Validation loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_acc'], 'b', label='Training accuracy')
    plt.plot(epochs, history['val_acc'], 'r', label='Validation accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

class LOSOCV:
    def __init__(self, model, metrics=None, n_folds=None, random_state=None, use_subject_normalization=False):
        self.model = model
        self.metrics = metrics if metrics else ['accuracy', 'sensitivity', 'specificity', 'f1']
        self.n_folds = n_folds
        self.random_state = random_state
        self.use_subject_normalization = use_subject_normalization
        self.METRIC_FUNCTIONS = METRIC_FUNCTIONS # Use the defined metric functions

    def run(self, features, targets, flatten_final=True, verbose=0):
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import confusion_matrix
        n_subjects = len(features)
        train_confusion_matrices, test_confusion_matrices = [], []

        if self.n_folds is None:
            subject_indices = range(n_subjects)
        else:
            rng = np.random.default_rng(self.random_state)
            subject_indices = rng.choice(n_subjects, size=self.n_folds, replace=False)

        for fold_index, subject_index in enumerate(subject_indices):
            train_indices = [i for i in range(n_subjects) if i != subject_index]
            train_X, train_y = _prepare_data(features, targets, train_indices, flatten_final, self.use_subject_normalization)
            test_X, test_y = _prepare_data(features, targets, [subject_index], flatten_final, self.use_subject_normalization)

            if flatten_final and not self.use_subject_normalization:
                scaler = StandardScaler()
                train_X = scaler.fit_transform(train_X)
                test_X = scaler.transform(test_X)

            for layer in self.model.model.modules():
                if hasattr(layer, 'reset_parameters'):
                    layer.reset_parameters()

            trained_model, epoch_losses = self.model.fit(train_X, train_y, calculate_epoch_loss=(verbose >= 2))
            self.model = trained_model

            train_pred = self.model.predict(train_X)
            test_pred = self.model.predict(test_X)

            labels = np.unique(np.concatenate([np.array([t] if np.isscalar(t) else t) for t in targets]))
            train_cm = confusion_matrix(train_y, train_pred, labels=labels)
            test_cm = confusion_matrix(test_y, test_pred, labels=labels)
            train_confusion_matrices.append(train_cm)
            test_confusion_matrices.append(test_cm)

            if verbose >= 1:
                print(f"\nFold {fold_index + 1}/{len(subject_indices)} - Subject {subject_index}:")
                fold_train_metrics = {m: self.METRIC_FUNCTIONS[m](train_cm) for m in self.metrics}
                fold_test_metrics = {m: self.METRIC_FUNCTIONS[m](test_cm) for m in self.metrics}
                print(f"  Train Metrics: {fold_train_metrics}")
                print(f"  Test Metrics: {fold_test_metrics}")

        train_confusion_matrix = np.sum(train_confusion_matrices, axis=0)
        test_confusion_matrix = np.sum(test_confusion_matrices, axis=0)

        train_metrics = {m: self.METRIC_FUNCTIONS[m](train_confusion_matrix) for m in self.metrics}
        test_metrics = {m: self.METRIC_FUNCTIONS[m](test_confusion_matrix) for m in self.metrics}

        return train_metrics, test_metrics
    
def subject_dependent_eval(model, features, targets, test=[1, 2], val=None, flatten_final=True, verbose=2, plot=False, patience=5):
    """
    Perform subject-dependent evaluation with early stopping and plotting.
    
    Train the model on subjects not in the test or validation lists, validate on the validation subjects,
    and evaluate on the test subjects. Supports early stopping based on validation loss.
    
    Args:
        model: A model wrapper instance with fit and predict methods.
        features (list of np.array): List of feature arrays per subject.
            Each array is expected to have shape (n_epochs, T, B, C).
        targets (list): List of target labels per subject.
        test (list of int): List of subject IDs to use for testing (1-based indexing).
        val (list of int): List of subject IDs to use for validating (1-based indexing).
            If None, no validation is performed.
        flatten_final (bool): If True, reshape data into 2D before training.
        verbose (int): Verbosity level (0: silent, 1: summary, 2: detailed logs).
        plot (bool): If True, plot training and validation loss/accuracy.
    Returns:
        Tuple[dict, dict]: A tuple containing train metrics and test metrics.
    """
    # Convert subject IDs in to zero-based indices
    test_indices = [s - 1 for s in test]
    val_indices = [s - 1 for s in val] if val else []

    # Determine training indices (exclude test and validation subjects)
    train_indices = [i for i in range(len(features)) if i not in test_indices and i not in val_indices]
    
    # Prepare training, validation, and test data
    train_X, train_y = _prepare_data(features, targets, train_indices, flatten_final)
    val_X, val_y = _prepare_data(features, targets, val_indices, flatten_final) if val else (None, None)
    test_X, test_y = _prepare_data(features, targets, test_indices, flatten_final)    

    if verbose >= 1:
        print("Train set shape:", train_X.shape)
        if val:
            print("Validation set shape:", val_X.shape)
        print("Test set shape:", test_X.shape)
    
    # Normalize data if flattening is applied
    if flatten_final:
        scaler = StandardScaler()
        train_X = scaler.fit_transform(train_X)
        if val:
            val_X = scaler.transform(val_X)
        test_X = scaler.transform(test_X)
    
    # Reset model parameters before training
    for layer in model.model.modules():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()
    
    # Train the model with early stopping
    trained_model, history = model.fit_with_validation(train_X, train_y, val_X, val_y, patience=patience, verbose=verbose)
    model = trained_model  
    
    # Make predictions 
    train_pred = model.predict(train_X)
    test_pred = model.predict(test_X)
    val_pred = model.predict(val_X) if val else None
    
    # Compute confusion matrices
    labels = np.unique(targets)
    train_cm = confusion_matrix(train_y, train_pred, labels=labels)
    test_cm = confusion_matrix(test_y, test_pred, labels=labels)
    val_cm = confusion_matrix(val_y, val_pred, labels=labels) if val else None
    
    # Compute metrics 
    train_metrics = {metric: METRIC_FUNCTIONS[metric](train_cm) for metric in METRIC_FUNCTIONS}
    test_metrics = {metric: METRIC_FUNCTIONS[metric](test_cm) for metric in METRIC_FUNCTIONS}
    val_metrics = {m: METRIC_FUNCTIONS[m](val_cm) for m in METRIC_FUNCTIONS} if val else None
    
    if verbose >= 1:
        print(f"Train Metrics: {train_metrics}")
        if val:
            print(f"Validation Metrics: {val_metrics}")
        print(f"Test Metrics: {test_metrics}")

    # Plot training history if requested
    if plot:
        _plot_history(history)
    
    return train_metrics, test_metrics