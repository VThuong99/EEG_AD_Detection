import mne
import pandas as pd
import os
import pickle

DATA_PATH = "./data"  # Replace with your actual data path
PROCESSED_DATA_PATH = "./processed_data"  # Replace with your actual processed data path


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

    for filename in os.listdir(processed_path):
        filepath = os.path.join(processed_path, filename)
        with open(filepath, 'rb') as file:
            data = pickle.load(file)
            subject_data.append(data['subject_data'])
            targets.append(data['targets'])

    return subject_data, targets