import numpy as np
import mne
from mne_connectivity import spectral_connectivity_epochs

from config import sfreq, BANDS
from src.data_processing.bandpower import relative_band_power, mean_band_power, absolute_band_power

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
    
class MbpFeature(FeatureExtractor):
    """Calculate the mean band power (MBP) of the data for each subject."""
    
    def __init__(self, freq_bands=None, sfreq=sfreq, fmin=0.5, fmax=45):
        """
        Initialize the MbpFeature extractor.
        
        Parameters:
        - freq_bands (dict, optional): Dictionary of frequency bands, e.g., {'Delta': (0.5, 4), ...}.
                                      If None, default bands are used.
        - sfreq (float): Sampling frequency.
        - fmin (float): Minimum frequency for PSD calculation.
        - fmax (float): Maximum frequency for PSD calculation.
        """
        if freq_bands is None:
            self.freq_bands = {
                'Delta': (0.5, 4),
                'Theta': (4, 8),
                'Alpha': (8, 13),
                'Beta': (13, 25),
                'Gamma': (25, 45)
            }
        else:
            self.freq_bands = freq_bands
        self.freq_bands_list = sorted(list(set(
            freq for band in self.freq_bands.values() for freq in band
        )))
        self.psd_feature = PsdFeature(sfreq=sfreq, fmin=fmin, fmax=fmax)
    
    def extract(self, data: np.ndarray, sfreq=None) -> np.ndarray:
        """
        Extract MBP features from EEG data.
        
        Parameters:
        - data (np.ndarray): EEG data with shape (n_epochs, n_channels, n_samples).
        - sfreq (float, optional): Sampling frequency. If None, uses default from config.
        
        Returns:
        - mbps (np.ndarray): Mean band power with shape (n_epochs, n_channels, n_bands).
        """
        # Calculate PSD using PsdFeature
        psds = self.psd_feature.extract(data)
        freqs = self.psd_feature.get_freqs()
        
        # Calculate Mean Band Power using band_power.mean_band_power
        mbps = mean_band_power(psds, freqs, self.freq_bands_list)
        return mbps
    
class AbpFeature(FeatureExtractor):
    """Calculate the absolute band power (ABP) of the data for each subject with normalization options."""
    
    def __init__(self, freq_bands=None, sfreq=sfreq, fmin=0.5, fmax=45, normalize='zscore'):
        """
        Initialize the AbpFeature extractor.
        
        Parameters:
        - freq_bands (dict, optional): Dictionary of frequency bands, e.g., {'Delta': (0.5, 4), ...}.
                                      If None, default bands are used.
        - sfreq (float): Sampling frequency.
        - fmin (float): Minimum frequency for PSD calculation.
        - fmax (float): Maximum frequency for PSD calculation.
        - normalize (str, optional): Normalization method. Options:
                                    - 'zscore': Z-score normalization per band.
                                    - 'relative': Normalize by total power (like RBP).
                                    - None: No normalization (raw ABP).
        """
        if freq_bands is None:
            self.freq_bands = {
                'Delta': (0.5, 4),
                'Theta': (4, 8),
                'Alpha': (8, 13),
                'Beta': (13, 25),
                'Gamma': (25, 45)
            }
        else:
            self.freq_bands = freq_bands
        self.freq_bands_list = sorted(list(set(
            freq for band in self.freq_bands.values() for freq in band
        )))
        self.psd_feature = PsdFeature(sfreq=sfreq, fmin=fmin, fmax=fmax)
        self.normalize = normalize
    
    def _zscore_normalize(self, abps: np.ndarray) -> np.ndarray:
        """
        Apply z-score normalization to ABP per band.
        
        Parameters:
        - abps (np.ndarray): ABP data with shape (n_epochs, n_channels, n_bands).
        
        Returns:
        - abps_normalized (np.ndarray): Z-score normalized ABP.
        """
        abps_normalized = np.zeros_like(abps)
        for band in range(abps.shape[-1]):
            band_data = abps[:, :, band]
            mean = np.mean(band_data)
            std = np.std(band_data)
            abps_normalized[:, :, band] = (band_data - mean) / (std + 1e-8)
        return abps_normalized
    
    def _relative_normalize(self, abps: np.ndarray) -> np.ndarray:
        """
        Normalize ABP by total power (similar to RBP).
        
        Parameters:
        - abps (np.ndarray): ABP data with shape (n_epochs, n_channels, n_bands).
        
        Returns:
        - abps_normalized (np.ndarray): Relative power normalized ABP.
        """
        total_power = np.sum(abps, axis=-1, keepdims=True)
        return abps / (total_power + 1e-8)
    
    def extract(self, data: np.ndarray, sfreq=None) -> np.ndarray:
        """
        Extract ABP features from EEG data with optional normalization.
        
        Parameters:
        - data (np.ndarray): EEG data with shape (n_epochs, n_channels, n_samples).
        - sfreq (float, optional): Sampling frequency. If None, uses default from config.
        
        Returns:
        - abps (np.ndarray): Absolute band power with shape (n_epochs, n_channels, n_bands),
                            optionally normalized.
        """
        # Calculate PSD using PsdFeature
        psds = self.psd_feature.extract(data)
        freqs = self.psd_feature.get_freqs()
        
        # Calculate Absolute Band Power using band_power.absolute_band_power
        abps = absolute_band_power(psds, freqs, self.freq_bands_list)
        
        # Apply normalization if specified
        if self.normalize == 'zscore':
            abps = self._zscore_normalize(abps)
        elif self.normalize == 'relative':
            abps = self._relative_normalize(abps)
        # If normalize=None, return raw ABP
        
        return abps


class NoGammaRbpFeature(FeatureExtractor):
    """Calculate the relative band power of the data for each subject, excluding gamma band."""

    def __init__(self, sfreq=sfreq, fmin=0.5, fmax=45):
        self.freq_bands_dict = {
            'Delta': (0.5, 4),
            'Theta': (4, 8),
            'Alpha': (8, 13),
            'Beta': (13, 25)
        }
        self.freq_bands = sorted(list(set(
            freq for band in self.freq_bands_dict.values() for freq in band
        )))
        self.psd_feature = PsdFeature(sfreq=sfreq, fmin=fmin, fmax=fmax)

    def extract(self, data: np.ndarray, sfreq=None) -> np.ndarray:
        psds = self.psd_feature.extract(data)
        freqs = self.psd_feature.get_freqs()

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

class DICE_NetFeature:
    """
    Class to extract features for DICE-Net using only RBP.
    
    For each 30-second epoch (input shape: (n_channels, n_samples)), the epoch is divided 
    into T one-second segments. For each segment and each channel, the relative band power 
    is computed for B frequency bands (e.g. Delta, Theta, Alpha, Beta, Gamma). 
    The resulting feature for each epoch has shape (T, B, n_channels).
    
    Parameters:
        T (int): Number of segments (default 30).
        sfreq (float): Sampling frequency (e.g. 500 Hz).
        fmin (float): Minimum frequency (default 0.5 Hz).
        fmax (float): Maximum frequency (default 45 Hz).
        freq_bands (dict): Dictionary of frequency bands.
                           Default: {'Delta': (0.5, 4), 'Theta': (4, 8), 'Alpha': (8, 13),
                                     'Beta': (13, 25), 'Gamma': (25, 45)}
    """
    def __init__(self, T=30, sfreq=500, fmin=0.5, fmax=45, freq_bands=None):
        self.T = T
        self.sfreq = sfreq
        self.fmin = fmin
        self.fmax = fmax
        if freq_bands is None:
            self.freq_bands = {
                'Delta': (0.5, 4),
                'Theta': (4, 8),
                'Alpha': (8, 13),
                'Beta': (13, 25),
                'Gamma': (25, 45)
            }
        else:
            self.freq_bands = freq_bands

    def extract(self, data: np.ndarray, sfreq=None) -> np.ndarray:
        """
        Extracts RBP features for DICE-Net model.
        Paper link: https://ieeexplore.ieee.org/document/10179900/
        
        Parameters:
            data (np.ndarray): Raw EEG data with shape (n_epochs, n_channels, n_samples).
            sfreq (float, optional): Sampling frequency. If None, uses self.sfreq.
        
        Returns:
            features_all (np.ndarray): Array of shape (n_epochs, T, B, n_channels) where
                                       T = number of segments (e.g. 30),
                                       B = number of frequency bands (e.g. 5).
        """
        if sfreq is None:
            sfreq = self.sfreq
        
        n_epochs, n_channels, n_samples = data.shape
        T = self.T
        seg_length = n_samples // T  
        B = len(self.freq_bands)
        features_all = []

        # Change 2D data in to 3D
        for epoch in range(n_epochs):
            epoch_data = data[epoch]  # shape: (n_channels, n_samples)
            features_epoch = np.zeros((T, B, n_channels))
            for t in range(T):
                start = t * seg_length
                end = start + seg_length
                segment = epoch_data[:, start:end]  # shape: (n_channels, seg_length)

                psds, freqs = mne.time_frequency.psd_array_welch(
                    segment,
                    sfreq=sfreq,
                    fmin=self.fmin,
                    fmax=self.fmax,
                    n_fft=seg_length,
                    verbose=False
                )
                # psds: shape (n_channels, n_freqs)
                for b_idx, (band, (f_low, f_high)) in enumerate(self.freq_bands.items()):
                    band_idx = np.where((freqs >= f_low) & (freqs <= f_high))[0]
                    psd_band = psds[:, band_idx].sum(axis=1)
                    psd_total = psds.sum(axis=1) + 1e-8
                    rbp = psd_band / psd_total  # shape: (n_channels,)
                    features_epoch[t, b_idx, :] = rbp
            features_all.append(features_epoch)
        features_all = np.array(features_all)  # shape: (n_epochs, T, B, n_channels)
        return features_all
    

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