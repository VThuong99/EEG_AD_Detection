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
        # all_epoch_losses = None # No longer need to initialize or return all_epoch_losses

        # if verbose >= 2: # No longer need to initialize all_epoch_losses conditionally
        #     all_epoch_losses = [] # List to store epoch losses for all folds


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
                if verbose >= 2: # Conditionally print epoch losses if verbose level is 2 or higher
                    print(f"  Epoch Losses: {epoch_losses}") # Print epoch losses
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