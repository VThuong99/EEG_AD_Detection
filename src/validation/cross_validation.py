import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from src.validation.metrics import accuracy, sensitivity, specificity, f1, precision 

METRIC_FUNCTIONS = {
    'accuracy': accuracy,
    'sensitivity': sensitivity,
    'specificity': specificity,
    'f1': f1,
    'precision': precision
}

def _prepare_data(features, targets, include_indices=None, flatten_final=True):
    """
    Prepare data by including subjects specified by include_indices.
    
    Args:
        features (list of np.array): Feature arrays for each subject.
        targets (list): Target label for each subject.
        include_indices (list of int): List of subject indices to include.
        flatten_final (bool): If True, reshape the data to 2D.
        
    Returns:
        Tuple[np.array, np.array]: The concatenated features and targets for included subjects.
    """
    if include_indices is None:
        include_indices = range(len(features))
    all_features = [features[i] for i in include_indices]
    all_targets = [targets[i] * np.ones(features[i].shape[0]) for i in include_indices]
    features_array = np.concatenate(all_features)
    targets_array = np.concatenate(all_targets)
    
    if flatten_final:
        features_array = features_array.reshape((features_array.shape[0], -1))
    
    return features_array, targets_array

def _normalize_data(train_X, val_X=None, test_X=None, flatten_final=True):
    """
    Normalize data using StandardScaler.
    
    Args:
        train_X (np.array): Training features.
        val_X (np.array, optional): Validation features.
        test_X (np.array, optional): Test features.
        flatten_final (bool): If True, apply normalization.
        
    Returns:
        Tuple[np.array, np.array, np.array]: Normalized train, validation, and test features.
    """
    if not flatten_final:
        return train_X, val_X, test_X
    
    scaler = StandardScaler()
    train_X = scaler.fit_transform(train_X)
    if val_X is not None:
        val_X = scaler.transform(val_X)
    if test_X is not None:
        test_X = scaler.transform(test_X)
    
    return train_X, val_X, test_X

def _compute_metrics(confusion_matrix, metrics, metric_functions=METRIC_FUNCTIONS):
    """
    Compute metrics from a confusion matrix.
    
    Args:
        confusion_matrix (np.array): Confusion matrix.
        metrics (list): List of metric names to compute.
        metric_functions (dict): Dictionary of metric functions.
        
    Returns:
        dict: Dictionary of metric names and their values.
    """
    return {m: metric_functions[m](confusion_matrix) for m in metrics}

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

def _reset_model(model):
    """
    Reset model parameters if applicable.
    
    Args:
        model: Model instance.
        
    Returns:
        Model instance with reset parameters.
    """
    if hasattr(model, 'model'): # is Pytorch model?
        try:
            for layer in model.model.modules():
                if hasattr(layer, 'reset_parameters'):
                    layer.reset_parameters()
        except AttributeError:
            pass
    elif hasattr(model, '__class__') and hasattr(model, 'get_params'): # is Scikit-learn model?
        model_class = model.__class__
        model_params = model.get_params()
        model = model_class(**model_params)
    return model

class BaseCV:
    """Base class for cross-validation methods."""
    def __init__(self, model, metrics=None, random_state=None):
        """
        Args:
            model: An object with 'fit' and 'predict' methods.
            metrics (list): List of metric names to calculate.
            random_state (int, optional): Random state for reproducibility.
        """
        if metrics and not all(m in METRIC_FUNCTIONS for m in metrics):
            raise ValueError(f"Metrics must be in {list(METRIC_FUNCTIONS.keys())}")
        self.model = model
        self.metrics = metrics if metrics else ['accuracy', 'sensitivity', 'specificity', 'f1']
        self.random_state = random_state
        self.METRIC_FUNCTIONS = METRIC_FUNCTIONS

    def _train_and_evaluate(self, train_X, train_y, val_X, val_y, test_X, test_y, labels, flatten_final, verbose, patience=None):
        """
        Train model and evaluate on train, validation, and test sets.
        
        Args:
            train_X, train_y: Training data.
            val_X, val_y: Validation data (optional).
            test_X, test_y: Test data.
            labels: Unique target labels.
            flatten_final (bool): If True, normalize data.
            verbose (int): Verbosity level.
            patience (int, optional): Patience for early stopping.
            
        Returns:
            Tuple: Train and test confusion matrices, history (if applicable).
        """
        train_X, val_X, test_X = _normalize_data(train_X, val_X, test_X, flatten_final)
        self.model = _reset_model(self.model)
        
        history = None
        if hasattr(self.model, 'fit_with_validation') and (val_X is not None or patience):
            self.model, history = self.model.fit_with_validation(
                train_X, train_y, val_X, val_y, patience=patience, verbose=(verbose >= 2)
            )
        else:
            self.model.fit(train_X, train_y)
        
        train_pred = self.model.predict(train_X)
        test_pred = self.model.predict(test_X)
        
        train_cm = confusion_matrix(train_y, train_pred, labels=labels)
        test_cm = confusion_matrix(test_y, test_pred, labels=labels)
        
        return train_cm, test_cm, history

class LOSOCV(BaseCV):
    """Perform Leave-One-Subject-Out Cross-Validation."""
    def __init__(self, model, metrics=None, n_folds=None, random_state=None):
        super().__init__(model, metrics, random_state)
        self.n_folds = n_folds

    def run(self, features, targets, flatten_final=True, verbose=0, plot=False): 
        """ 
        Performs the LOSO cross-validation.
        """
        n_subjects = len(features)
        subject_indices = range(n_subjects) if self.n_folds is None else \
            np.random.default_rng(self.random_state).choice(n_subjects, size=self.n_folds, replace=False)

        labels = np.unique(targets)
        train_confusion_matrices, test_confusion_matrices = [], []
        histories = []

        # LOCOCV code
        for fold_index, subject_index in enumerate(subject_indices): 
            # Prepare training and testing data for this fold
            train_indices = [i for i in range(n_subjects) if i != subject_index]
            train_X, train_y = _prepare_data(features, targets, train_indices, flatten_final)
            test_X, test_y = _prepare_data(features, targets, [subject_index], flatten_final)

            train_cm, test_cm, history = self._train_and_evaluate(
                train_X, train_y, None, None, test_X, test_y, labels, flatten_final, verbose
            )
            
            train_confusion_matrices.append(train_cm)
            test_confusion_matrices.append(test_cm)
            if history:
                histories.append(history)
            
            if verbose >= 1:
                print(f"\nFold {fold_index + 1}/{len(subject_indices)} - Subject {subject_index}:")
                print(f"  Train Metrics: {_compute_metrics(train_cm, self.metrics)}")
                print(f"  Test Metrics: {_compute_metrics(test_cm, self.metrics)}")
        
        total_train_cm = np.sum(train_confusion_matrices, axis=0)
        total_test_cm = np.sum(test_confusion_matrices, axis=0)
        
        train_metrics = _compute_metrics(total_train_cm, self.metrics)
        test_metrics = _compute_metrics(total_test_cm, self.metrics)
        
        if plot and histories:
            avg_history = {
                'train_loss': np.mean([h['train_loss'] for h in histories], axis=0).tolist(),
                'val_loss': np.mean([h['val_loss'] for h in histories], axis=0).tolist(),
                'train_acc': np.mean([h['train_acc'] for h in histories], axis=0).tolist(),
                'val_acc': np.mean([h['val_acc'] for h in histories], axis=0).tolist()
            }
            _plot_history(avg_history, 'losocv_history.png')
        
        return train_metrics, test_metrics
    
class MCCV(BaseCV):
    """Perform Monte-Carlo Cross-Validation with subject-dependent settings."""
    def __init__(self, model, metrics=None, n_iter=10, test_size=0.2, val_size=0.2, random_state=None, patience=5):
        super().__init__(model, metrics, random_state)
        if not (0 < test_size < 1) or not (0 < val_size < 1):
            raise ValueError("test_size and val_size must be between 0 and 1")
        self.n_iter = n_iter
        self.test_size = test_size
        self.val_size = val_size
        self.patience = patience

    def run(self, features, targets, flatten_final=True, verbose=0, plot=False):
        """Performs MCCV with subject-dependent splits."""
        n_subjects = len(features)
        subject_indices = np.arange(n_subjects)
        train_confusion_matrices, test_confusion_matrices = [], []
        histories = []
        test_indices_per_fold = []
        
        rng = np.random.default_rng(self.random_state)
        labels = np.unique(targets)
        
        for iter in range(self.n_iter):
            iter_seed = rng.integers(0, 10000)
            train_val_indices, test_indices = train_test_split(subject_indices, test_size=self.test_size, random_state=iter_seed)
            val_proportion = self.val_size / (1 - self.test_size)
            train_indices, val_indices = train_test_split(train_val_indices, test_size=val_proportion, random_state=iter_seed)
            
            train_X, train_y = _prepare_data(features, targets, train_indices, flatten_final)
            val_X, val_y = _prepare_data(features, targets, val_indices, flatten_final)
            test_X, test_y = _prepare_data(features, targets, test_indices, flatten_final)
            
            train_cm, test_cm, history = self._train_and_evaluate(
                train_X, train_y, val_X, val_y, test_X, test_y, labels, flatten_final, verbose, self.patience
            )
            
            train_confusion_matrices.append(train_cm)
            test_confusion_matrices.append(test_cm)
            test_indices_per_fold.append(test_indices.tolist())
            if history:
                histories.append(history)
            
            if verbose >= 1:
                print(f"\nIteration {iter + 1}/{self.n_iter}:")
                print(f"  Test Subjects: {test_indices}")
                print(f"  Train Metrics: {_compute_metrics(train_cm, self.metrics)}")
                print(f"  Test Metrics: {_compute_metrics(test_cm, self.metrics)}")
        
        total_train_cm = np.sum(train_confusion_matrices, axis=0)
        total_test_cm = np.sum(test_confusion_matrices, axis=0)
        
        train_metrics = _compute_metrics(total_train_cm, self.metrics)
        test_metrics = _compute_metrics(total_test_cm, self.metrics)
        
        if plot and histories:
            avg_history = {
                'train_loss': np.mean([h['train_loss'] for h in histories], axis=0).tolist(),
                'val_loss': np.mean([h['val_loss'] for h in histories], axis=0).tolist(),
                'train_acc': np.mean([h['train_acc'] for h in histories], axis=0).tolist(),
                'val_acc': np.mean([h['val_acc'] for h in histories], axis=0).tolist()
            }
            _plot_history(avg_history, 'mccv_history.png')
        
        return train_metrics, test_metrics

def subject_independent_eval(model, features, targets, test=[1, 2], val=None, flatten_final=True, verbose=2, plot=False, patience=5):
    """
    Perform subject-independent evaluation with early stopping and plotting.
    """
    if not all(1 <= s <= len(features) for s in test) or (val and not all(1 <= s <= len(features) for s in val)):
        raise ValueError("Test and validation indices must be valid subject IDs")
    
    test_indices = [s - 1 for s in test]
    val_indices = [s - 1 for s in val] if val else []
    train_indices = [i for i in range(len(features)) if i not in test_indices and i not in val_indices]
    
    train_X, train_y = _prepare_data(features, targets, train_indices, flatten_final)
    val_X, val_y = _prepare_data(features, targets, val_indices, flatten_final) if val else (None, None)
    test_X, test_y = _prepare_data(features, targets, test_indices, flatten_final)
    
    if verbose >= 1:
        print("Train set shape:", train_X.shape)
        if val:
            print("Validation set shape:", val_X.shape)
        print("Test set shape:", test_X.shape)
    
    labels = np.unique(targets)
    base_cv = BaseCV(model, METRIC_FUNCTIONS.keys())  # Use all metrics for subject-dependent eval
    train_cm, test_cm, history = base_cv._train_and_evaluate(
        train_X, train_y, val_X, val_y, test_X, test_y, labels, flatten_final, verbose, patience
    )
    
    val_cm = None
    if val_X is not None:
        val_pred = model.predict(val_X)
        val_cm = confusion_matrix(val_y, val_pred, labels=labels)
    
    train_metrics = _compute_metrics(train_cm, METRIC_FUNCTIONS.keys())
    test_metrics = _compute_metrics(test_cm, METRIC_FUNCTIONS.keys())
    val_metrics = _compute_metrics(val_cm, METRIC_FUNCTIONS.keys()) if val_cm is not None else None
    
    if verbose >= 1:
        print(f"Train Metrics: {train_metrics}")
        if val_metrics:
            print(f"Validation Metrics: {val_metrics}")
        print(f"Test Metrics: {test_metrics}")
    
    if plot and history:
        _plot_history(history)
    
    return train_metrics, test_metrics