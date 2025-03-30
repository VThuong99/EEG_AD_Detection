import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

from src.validation.metrics import accuracy, sensitivity, specificity, f1, precision # Assuming metrics.py is in src/validation

# Define metric functions here if 'from src.validation.metrics import ...' causes issues
METRIC_FUNCTIONS = {
    'accuracy': accuracy,
    'sensitivity': sensitivity,
    'specificity': specificity,
    'f1': f1,
    'precision': precision
}

def train_prep(features, targets, exclude=None, flatten_final=True):
    """ Prepares a list of feature arrays with corresponding labels in targets for training."""
    total_subjects = len(targets)
    targets_list = []
    for i in range(total_subjects):
        num_epochs = features[i].shape[0]
        targets_list.append(targets[i] * np.ones(num_epochs))
    if exclude is None:
        features_array = np.concatenate(features)
        targets_array = np.concatenate(targets_list)
    else:
        features_array = np.concatenate(features[:exclude] + features[exclude + 1:])
        targets_array = np.concatenate(targets_list[:exclude] + targets_list[exclude + 1:])
    if flatten_final:
        features_array = features_array.reshape((features_array.shape[0], -1))
    return features_array, targets_array

def _train_prep(features, targets, exclude_indices=None, flatten_final=True):
    """
    Prepare training data by excluding subjects specified by exclude_indices.
    
    Args:
        features (list of np.array): Feature arrays for each subject.
        targets (list): Target label for each subject.
        exclude_indices (list of int, optional): List of subject indices to exclude (for test).
        flatten_final (bool): If True, reshape the data to 2D.
        
    Returns:
        Tuple[np.array, np.array]: The concatenated training features and targets.
    """
    all_features = []
    all_targets = []
    total_subjects = len(targets)
    
    for i in range(total_subjects):
        if exclude_indices is not None and i in exclude_indices:
            continue  # Skip test subjects
        n_epochs = features[i].shape[0]
        all_features.append(features[i])
        all_targets.append(targets[i] * np.ones(n_epochs))
    
    features_array = np.concatenate(all_features)
    targets_array = np.concatenate(all_targets)
    
    if flatten_final:
        features_array = features_array.reshape((features_array.shape[0], -1))
    
    return features_array, targets_array

class LOSOCV:
    """Perform Leave-One-Subject-Out Cross-Validation."""
    def __init__(self, model, metrics=None, n_folds=None, random_state=None):
        """
        Args:
            model: An object with 'fit' and 'predict' methods.
            metrics: A list of metric names (strings) to calculate.
                Must be keys in the METRIC_FUNCTIONS dictionary.
                Defaults to ['accuracy', 'sensitivity', 'specificity', 'f1'].
            n_folds (int, optional): The number of folds to run the model on. Defaults to None for all runs
            random_state (int, optional): The random state for reproducability. Defaults to None
        """
        self.model = model
        self.metrics = metrics if metrics else ['accuracy', 'sensitivity', 'specificity', 'f1']
        self.n_folds = n_folds
        self.random_state = random_state

        self.METRIC_FUNCTIONS = METRIC_FUNCTIONS # Use the defined metric functions

    def run(self, features, targets, flatten_final=True, verbose=0): # verbose now accepts integer levels
        """ Performs the LOSO cross-validation.

        Args:
            features: List of feature arrays, one for each subject.
            targets: List of target labels, one for each subject.
            flatten_final (bool, optional): Whether to flatten the feature array. Defaults to True.
            verbose (int, optional): Verbosity level. 0: No detailed output, 1: Fold metrics, 2: Fold metrics and epoch losses. Defaults to 0. # Updated docstring for verbose levels
        """
        n_subjects = len(features)

        train_confusion_matrices = []
        test_confusion_matrices = []

        if (self.n_folds == None):
            subject_indices = range(n_subjects)
        else:
            rng = np.random.default_rng(self.random_state)
            subject_indices = rng.choice(n_subjects, size=self.n_folds, replace=False) # Generate from a uniform distribution

        # LOCOCV code
        for fold_index, subject_index in enumerate(subject_indices): # Added fold_index for verbose output
            # Prepare training and testing data for this fold
            train_X, train_y = train_prep(features, targets, exclude=subject_index, flatten_final=flatten_final)
            if flatten_final:
                test_X = features[subject_index].reshape(features[subject_index].shape[0], -1)
            else:
                test_X = features[subject_index]

            test_y = targets[subject_index] * np.ones(features[subject_index].shape[0])

            if flatten_final:
                scaler = StandardScaler()
                train_X = scaler.fit_transform(train_X)
                test_X = scaler.transform(test_X)

            for layer in self.model.model.modules():
                if hasattr(layer, 'reset_parameters'):
                    layer.reset_parameters()

            # Fit the model on the training data
            trained_model, epoch_losses = self.model.fit(train_X, train_y, calculate_epoch_loss=(verbose >= 2)) 
            self.model = trained_model # Update self.model with trained model for prediction

            # Make predictions on training and testing data
            train_pred = self.model.predict(train_X)
            test_pred = self.model.predict(test_X)

            # Calculate confusion matrices
            labels = np.unique(targets)
            train_cm = confusion_matrix(train_y, train_pred, labels=labels)
            test_cm = confusion_matrix(test_y, test_pred, labels=labels)
            train_confusion_matrices.append(train_cm)
            test_confusion_matrices.append(test_cm)

            if verbose >= 1: # Print fold-wise information if verbose level is 1 or higher
                print(f"\nFold {fold_index + 1}/{len(subject_indices)} - Subject {subject_index}:") # Added fold index and subject index
                fold_train_metrics = {}
                fold_test_metrics = {}
                for metric in self.metrics:
                    fold_train_metrics[metric] = self.METRIC_FUNCTIONS[metric](train_cm)
                    fold_test_metrics[metric] = self.METRIC_FUNCTIONS[metric](test_cm)
                print(f"  Train Metrics: {fold_train_metrics}")
                print(f"  Test Metrics: {fold_test_metrics}")


        # Calculate the average metrics across all folds
        train_confusion_matrix = np.sum(train_confusion_matrices, axis=0)
        test_confusion_matrix = np.sum(test_confusion_matrices, axis=0)

        # Calculate the metrics
        train_metrics = {}
        test_metrics = {}
        for metric in self.metrics: #Loop through all of the strings in the metrics function
            train_metrics[metric] = self.METRIC_FUNCTIONS[metric](train_confusion_matrix)
            test_metrics[metric] = self.METRIC_FUNCTIONS[metric](test_confusion_matrix)

        return train_metrics, test_metrics 
    
def subject_dependent_eval(model, features, targets, sub=['1', '2'], flatten_final=True, verbose=2):
    """
    Perform subject-dependent evaluation.
    
    Train the model on subjects not in the test list and evaluate on the subjects specified by `sub`.
    
    Args:
        model: A model wrapper instance (e.g., DeepLearningModel) with fit and predict methods.
        features (list of np.array): List of feature arrays per subject.
            Each array is expected to have shape (n_epochs, T, B, C).
        targets (list): List of target labels per subject.
        sub (list of str): List of subject IDs (as strings) to use for testing.
            Assumes subject '1' corresponds to index 0, etc.
        flatten_final (bool): If True, reshape data into 2D before training.
            (Set to False if the model expects 4D input.)
        verbose (int): Verbosity level (0: silent, 1: summary, 2: detailed logs).
    
    Returns:
        Tuple[dict, dict]: A tuple containing train metrics and test metrics.
    """
    # Convert subject IDs in 'sub' to zero-based indices
    test_indices = [int(s) - 1 for s in sub]
    # Determine training subject indices (all indices not in test_indices)
    train_indices = [i for i in range(len(features)) if i not in test_indices]
    
    # Prepare training data by excluding test subjects
    train_X, train_y = _train_prep(features, targets, exclude_indices=test_indices, flatten_final=flatten_final)
    
    # Prepare test data by concatenating test subject features
    test_list = []
    test_targets_list = []
    for i in test_indices:
        test_list.append(features[i])
        n_epochs = features[i].shape[0]
        test_targets_list.append(targets[i] * np.ones(n_epochs))
    test_X = np.concatenate(test_list)
    test_y = np.concatenate(test_targets_list)
    if flatten_final:
        test_X = test_X.reshape((test_X.shape[0], -1))
    
    if verbose >= 1:
        print("Train set shape:", train_X.shape)
        print("Test set shape:", test_X.shape)
    
    # Normalize data if flattening is applied
    if flatten_final:
        scaler = StandardScaler()
        train_X = scaler.fit_transform(train_X)
        test_X = scaler.transform(test_X)
    
    # Reset model parameters before training (if applicable)
    # Here we iterate over the internal model modules.
    for layer in model.model.modules():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()
    
    # Train the model on the training data
    trained_model, epoch_losses = model.fit(train_X, train_y, calculate_epoch_loss=(verbose >= 2))
    model = trained_model  # Update model reference
    
    # Make predictions on training and test data
    train_pred = model.predict(train_X)
    test_pred = model.predict(test_X)
    
    # Compute confusion matrices
    labels = np.unique(targets)
    train_cm = confusion_matrix(train_y, train_pred, labels=labels)
    test_cm = confusion_matrix(test_y, test_pred, labels=labels)
    
    # Compute evaluation metrics using the provided metric functions
    train_metrics = {metric: METRIC_FUNCTIONS[metric](train_cm) for metric in METRIC_FUNCTIONS}
    test_metrics = {metric: METRIC_FUNCTIONS[metric](test_cm) for metric in METRIC_FUNCTIONS}
    
    if verbose >= 1:
        print(f"Train Metrics: {train_metrics}")
        print(f"Test Metrics: {test_metrics}")
    
    return train_metrics, test_metrics