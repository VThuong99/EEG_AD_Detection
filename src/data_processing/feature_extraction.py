import numpy as np
import mne
from mne_connectivity import spectral_connectivity_epochs

from config import sfreq, BANDS
from src.data_processing.bandpower import relative_band_power

class FeatureExtractor:
    """Base class for all feature extractors."""
    def extract(self, data: np.ndarray, sfreq=None) -> np.ndarray:
        raise NotImplementedError('Must implement extract method.')
    
class PsdFeature(FeatureExtractor):
    """Calcuate the power spectral density of the data for each subject."""
    def __init__(self, sfreq=sfreq, fmin=0.5, fmax=45):
        self.sfreq = sfreq 
        self.fmin = fmin 
        self.fmax = fmax
        self.freqs = None
    def extract(self, data: np.ndarray, sfreq=sfreq) -> np.ndarray:
        """
        Extracts psd features using welch method.
        Assumes input data of shape (eeg signal): (n_epochs, n_channels, n_samples_per_epoch).
        """

        n_epochs, n_channels, n_samples = data.shape
        all_psds = []

        for epoch in range(n_epochs):
            psds, freqs = mne.time_frequency.psd_array_welch(
                data[epoch],
                sfreq=self.sfreq,
                fmin=self.fmin,
                fmax=self.fmax,
                n_fft=int(sfreq * 4), #Fixed window length for frequency resolution. e.g. 2s window for 500Hz data frequency resolution of 0.5Hz
                verbose=False 
            )
            all_psds.append(psds)

        processed_data = np.array(all_psds)
        self.freqs = freqs # Store freqs for later use
        return processed_data
    
    def get_freqs(self):
        return self.freqs

class RbpFeature(FeatureExtractor):
    """Calculate the relative band power of the data for each subject."""

    def __init__(self, freq_bands=None, sfreq=sfreq, fmin=0.5, fmax=45):
        self.freq_bands = sorted(list(set(freq for band in BANDS.values() for freq in band)))
        self.psd_feature = PsdFeature(sfreq=sfreq, fmin=fmin, fmax=fmax)

    def extract(self, data: np.ndarray, sfreq=None) -> np.ndarray:
        # Calculate psd features 
        psds = self.psd_feature.extract(data)
        freqs = self.psd_feature.get_freqs()

        # Calculate relative band power
        rbps = relative_band_power(psds, freqs, self.freq_bands)
        return rbps 


class SccFeature(FeatureExtractor):
    """ Calculate the spectral coherence connectivity of the data for each subject. """

    def __init__(self, sfreq=sfreq, fmin=0.5, fmax=45, method='coh'):
        self.sfreq = sfreq 
        self.fmin = fmin 
        self.fmax = fmax 
        self.method = method 
        self.ch_names = ['Fp1' , 'Fp2' , 'F3', 'F4' , 'C3', 'C4', 'P3', 'P4' , 'O1' , 'O2', 
                'F7' , 'F8', 'T3' , 'T4' , 'T5' , 'T6' , 'Fz', 'Cz','Pz'] # 19 channel names 10-20 system

    def extract(self, data: np.ndarray, sfreq=None, ch_names=None) -> np.ndarray:
        sfreq = self.sfreq 
        fmin = self.fmin 
        fmax = self.fmax 
        cwt_freqs = np.arange(fmin, fmax, 1) # 1Hz frequency resolution
        
        info = mne.create_info(ch_names=self.ch_names, sfreq=sfreq, ch_types='eeg')
        epochs = mne.EpochsArray(data, info)
        con = spectral_connectivity_epochs(epochs,
                                            method=self.method,
                                            mode='cwt_morlet', 
                                            sfreq=sfreq, 
                                            cwt_freqs=cwt_freqs, 
                                            verbose=False)
        
        return con.get_data(output='dense')
    
    def get_ch_names(self):
        return self.ch_names
    
class MeanFeature(FeatureExtractor):
    """Calculates the mean of the data for each subject."""
    def extract(self, data: np.ndarray, sfreq=None) -> np.ndarray:
        return np.mean(data, axis=2)

class VarianceFeature(FeatureExtractor):
    """Calculates the variance of the data for each subject."""
    def extract(self, data, sfreq=None):
        return np.var(data, axis=2)
    
class FlattenFeature:
    """Flatten features of shape (n_epochs, n_channels, n_samples) to (n_epochs, n_channels * n_samples)."""
    @staticmethod
    def process(features: np.ndarray) -> np.ndarray:
        n_epochs = features.shape[0]
        n_features = np.prod(features.shape[1:])
        
        return features.reshape(n_epochs, n_features)
    
class FeaturePipeline:
    """Pipeline to run multiple feature extractors. """
    def __init__(self, feature_extractors: list, post_processors=None):
        self.feature_extractors = feature_extractors
        self.post_processors = post_processors if post_processors else []

    def run(self, data: np.ndarray, sfreq):
        features = []
        for extractor in self.feature_extractors:
            extracted = extractor.extract(data, sfreq)
            features.append(extracted)

        feature_vector = combine_features(features)

        for processor in self.post_processors:
            feature_vector = processor.process(feature_vector)

        return feature_vector
    
def combine_features(features: list[np.ndarray]) -> np.ndarray:
    """Combines a list of feature arrays along the last axis, reshaping 2D arrays to 3D."""
    processed_features = []
    for feature in features:
        if feature.ndim == 2:
            feature = feature[:, :, np.newaxis] # Reshape to (n_epochs, n_channels, 1)
        processed_features.append(feature)

    return np.concatenate(processed_features, axis=2)