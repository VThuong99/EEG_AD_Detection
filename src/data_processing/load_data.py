import mne
import pandas as pd
import os
import pickle

from config import DATA_PATH, PROCESSED_DATA_PATH



def load_subject(subject_id: int, path: str = DATA_PATH) -> mne.io.Raw:
    """loads subject using their numeric id in the data folders"""
    return mne.io.read_raw_eeglab(path + '/derivatives/sub-' + str(subject_id).zfill(3)
                                  + '/eeg/sub-' + str(subject_id).zfill(3) + '_task-eyesclosed_eeg.set', preload=True, verbose='CRITICAL')


def process_data(duration: float, overlap: float, classes: dict = {'A': 1, 'F': 2, 'C': 0}, data_path=DATA_PATH, processed_path=PROCESSED_DATA_PATH) -> None:
    """
    Loads raw EEG data for all subjects from the specified classes, divides their recordings into epochs,
    and save the epochs along with the assigned class labels.

    Parameters
    ----------
    duration : float
        Duration of each epoch in seconds.
    overlap : float
        Overlap between epochs, in seconds.
    classes : dict, optional
        Dictionary whose keys are the classes to include and values are the numeric labels.
        By default {'A': 1, 'F': 2, 'C': 0}.
    data_path : str, optional
        Filepath to the data folder. Defaults to PATH in config.py.
    processed_path : str, optional
        Filepath to the folder where the processed data will be saved. Defaults to PROCESSED_DATA_PATH in config.py.

    Returns
    -------
    subject_datas : list
        List of numpy arrays, each array has shape (num_epochs, num_channels, num_samples_per_epoch) của từng subject.
    targets : list
        List contain target label with each subject.
    """

    subject_table = pd.read_csv(data_path + '/participants.tsv', sep='\t')
    target_labels = subject_table['Group']

    # Create subdirectory based on duration and overlap
    processed_subdir = f"{duration}s-{overlap}o"
    processed_path = os.path.join(processed_path, processed_subdir)
    os.makedirs(processed_path, exist_ok=True)  # Create the directory if it doesn't exist


    for subject_id in range(1, len(target_labels) + 1):
        if target_labels.iloc[subject_id - 1] not in classes:
            continue

        raw = load_subject(subject_id, path=data_path)
        epochs = mne.make_fixed_length_epochs(
            raw,
            duration=duration,
            overlap=overlap,
            preload=True
        )

        epochs_array = epochs.get_data()  # Shape: (num_epochs, num_channels, num_samples_per_epoch)

        filename = f"sub-{subject_id:03d}.pickle"  # Generate filename like "sub-001.pickle"
        filepath = os.path.join(processed_path, filename)
        with open(filepath, 'wb') as file:
            pickle.dump({'subject_data': epochs_array, 'targets': classes[target_labels.iloc[subject_id - 1]]}, file)


def load_processed_data(duration: float, overlap: float, processed_path=PROCESSED_DATA_PATH):
    """
    Loads the data and targets from all the pickle files in the specified directory.

    Args:
        duration (float): Duration used during the epoching.
        overlap (float): Overlap used during the epoching.
        processed_path (str, optional): The root directory where the pickle files are stored.
                               Defaults to PROCESSED_DATA_PATH.

    Returns:
        subject_data (list): A list where each element is the data for a subject.
        targets (list): A list where each element is the label for the corresponding
                       subject in subject_data.
    """

    # Determine the correct subdirectory
    processed_subdir = f"{duration}s-{overlap}o"
    processed_path = os.path.join(processed_path, processed_subdir)

    subject_data = []
    targets = []

    for filename in sorted(os.listdir(processed_path)):
        filepath = os.path.join(processed_path, filename)
        with open(filepath, 'rb') as file:
            data = pickle.load(file)
            subject_data.append(data['subject_data'])
            targets.append(data['targets'])

    return subject_data, targets


def load_large_data(data_path: str, datasets: list, duration: float = 4.0, overlap: float = 2.0, target_channels: int = 19, target_sampling_rate: int = 128, classes: dict = {"HC": 0, "AD": 1}):
    """
    Loads EEG data and labels from specified datasets (e.g., ["ADFTD", "ADSZ", "ADFSU"]) in the data_path.
    Preprocesses the data to match the target format (n_epochs, n_channels, n_samples) with specified duration and overlap in seconds.
    Filters subjects based on the provided classes dictionary and remaps labels (e.g., HC=0, AD=2 to AD=1).

    Args:
        data_path (str): Path to the directory containing dataset folders (e.g., "ADFTD", "ADSZ").
        datasets (list): List of dataset folder names to load (e.g., ["ADFTD", "ADSZ", "ADFSU"]).
        duration (float): Duration of each epoch in seconds (default: 4.0).
        overlap (float): Overlap between epochs in seconds (default: 2.0).
        target_channels (int): Number of channels to select (default: 19).
        target_sampling_rate (int): Sampling rate in Hz (default: 128).
        classes (dict): Dictionary mapping class names to numeric labels (default: {"HC": 0, "AD": 1}).

    Returns:
        subject_data (list): List of numpy arrays, each with shape (n_epochs, target_channels, n_samples).
        targets (list): List of numeric labels (0 for HC, 1 for AD) corresponding to each subject.
    """
    subject_data = []
    targets = []

    # Standard 10-20 EEG channels for reference
    standard_channels = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 
                        'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'Fz', 'Cz', 'Pz']

    # Mapping from original labels to new labels (e.g., 2 -> 1 for AD)
    label_mapping = {0: classes["HC"], 2: classes["AD"]}

    # Iterate through specified datasets
    for dataset in datasets:
        folder_path = os.path.join(data_path, dataset)
        if not os.path.isdir(folder_path):
            print(f"Warning: Dataset folder {folder_path} does not exist. Skipping.")
            continue

        feature_path = os.path.join(folder_path, "Feature")
        label_path = os.path.join(folder_path, "Label", "label.npy")

        # Check if "Feature" directory and "label.npy" exist
        if not os.path.exists(feature_path) or not os.path.exists(label_path):
            print(f"Warning: Missing 'Feature' or 'Label/label.npy' in {folder_path}. Skipping.")
            continue

        # Load label.npy
        labels = np.load(label_path)  # Shape: (n_subjects, 2), col 0: label, col 1: subject_id

        for label, subject_id in labels:
            # Skip subjects not in specified classes (HC=0, AD=2)
            if label not in label_mapping:
                continue

            # Generate feature file name, e.g., "feature_01.npy" for subject_id=1
            feature_file = f"feature_{subject_id:02d}.npy"
            feature_file_path = os.path.join(feature_path, feature_file)

            if os.path.exists(feature_file_path):
                # Load feature data
                subject_array = np.load(feature_file_path)  # Shape: (n_epochs, 128, n_channels)
                n_epochs, current_samples, n_channels = subject_array.shape

                # Verify sample rate matches expected input
                if current_samples != target_sampling_rate:
                    print(f"Warning: Subject {subject_id} in dataset {dataset} has {current_samples} samples per epoch, expected {target_sampling_rate}. Skipping.")
                    continue

                # Transpose to (n_epochs, n_channels, 128)
                subject_array = np.transpose(subject_array, (0, 2, 1))

                # Select target number of channels
                if n_channels > target_channels:
                    subject_array = subject_array[:, :target_channels, :]
                elif n_channels < target_channels:
                    print(f"Warning: Subject {subject_id} in dataset {dataset} has {n_channels} channels, padding with zeros to reach {target_channels}.")
                    padding = np.zeros((n_epochs, target_channels - n_channels, current_samples))
                    subject_array = np.concatenate([subject_array, padding], axis=1)

                # Calculate target samples and step size based on duration and overlap in seconds
                target_samples = int(duration * target_sampling_rate)  # e.g., 4 * 128 = 512
                step = int((duration - overlap) * target_sampling_rate)  # e.g., (4 - 2) * 128 = 256

                if step <= 0:
                    print(f"Warning: Invalid step size for subject {subject_id} in dataset {dataset} (duration: {duration}, overlap: {overlap}). Skipping epoch restructuring.")
                    continue

                # Combine epochs into a continuous signal
                total_samples = (n_epochs - 1) * current_samples + current_samples
                continuous_signal = np.zeros((target_channels, total_samples))
                for ch in range(target_channels):
                    for i in range(n_epochs):
                        start = i * current_samples
                        end = start + current_samples
                        continuous_signal[ch, start:end] = subject_array[i, ch, :]

                # Re-segment into new epochs
                new_epochs = []
                for start in range(0, total_samples - target_samples + 1, step):
                    new_epoch = continuous_signal[:, start:start + target_samples]
                    new_epochs.append(new_epoch)
                if new_epochs:
                    subject_array = np.array(new_epochs)  # Shape: (new_n_epochs, target_channels, target_samples)
                    subject_data.append(subject_array)
                    targets.append(label_mapping[label])  # Remap label (e.g., 2 -> 1 for AD)
                else:
                    print(f"Warning: No epochs created for subject {subject_id} in dataset {dataset}.")
            else:
                print(f"Warning: Missing feature file {feature_file} in {feature_path}.")

    return subject_data, targets