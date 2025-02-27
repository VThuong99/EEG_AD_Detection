import numpy as np  
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler 
from sklearn.metrics import confusion_matrix 

from src.validation.metrics import accuracy, sensitivity, specificity, f1, precision

def train_prep(features, targets, exclude=None, flatten_final=True):
    """ Prepares a list of feature arrays with corresponding labels in targets for training."""
    total_subjects = len(targets)
    targets_list = []
    for i in range(total_subjects):
        num_epochs = features[i].shape[0]
        targets_list.append(targets[i] * np.ones(num_epochs))
    if exclude == None:
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

        self.METRIC_FUNCTIONS = {
            'accuracy': accuracy,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'f1': f1,
            'precision': precision
        }

    def run(self, features, targets, flatten_final=True):
        """ Performs the LOSO cross-validation. """
        n_subjects = len(features)

        train_confusion_matrices = []
        test_confusion_matrices = []  

        if (self.n_folds == None):
            subject_indices = range(n_subjects)
        else:
            rng = np.random.default_rng(self.random_state)
            subject_indices = rng.choice(n_subjects, size=self.n_folds, replace=False) # Generate from a uniform distribution

        # LOCOCV code
        for i in subject_indices:
            # Prepare training and testing data for this fold
            train_X, train_y = train_prep(features, targets, exclude=i, flatten_final=True)
            test_X = features[i].reshape(features[i].shape[0], -1)
            test_y = targets[i] * np.ones(features[i].shape[0])

            # Scale the training and testing data using StandardScaler
            scaler = StandardScaler()
            train_X = scaler.fit_transform(train_X)
            test_X = scaler.transform(test_X)

            # Fit the model on the training data
            self.model.fit(train_X, train_y)

            # Make predictions on training and testing data
            train_pred = self.model.predict(train_X)
            test_pred = self.model.predict(test_X)

            # Calculate confusion matrices
            labels = np.unique(targets)
            train_confusion_matrices.append(confusion_matrix(train_y, train_pred, labels=labels))
            test_confusion_matrices.append(confusion_matrix(test_y, test_pred, labels=labels))

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


