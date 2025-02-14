#filepath to the data folder
DATA_PATH = 'datasets/ds004504'
#filepath to the folder storing data processed with mne
PROCESSED_DATA_PATH = 'datasets/processed_data'
#sample frequency of the data
sfreq = 500
#define frequency bands
BANDS = {
    'delta': (0.5, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta': (13, 25),
    'gamma': (25, 45)
}
