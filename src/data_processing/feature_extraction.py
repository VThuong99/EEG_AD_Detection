import numpy as np
import torch
import mne
from mne_connectivity import spectral_connectivity_epochs
from mne.time_frequency import tfr_array_morlet

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
        self.freqs = freqs
        return processed_data
    
    def get_freqs(self):
        return self.freqs

class RbpFeature(FeatureExtractor):
    """Calculate the relative band power of the data for each subject."""
    def __init__(self, freq_bands=None, sfreq=sfreq, fmin=0.5, fmax=45):
        self.freq_bands = sorted(list(set(freq for band in BANDS.values() for freq in band)))
        self.psd_feature = PsdFeature(sfreq=sfreq, fmin=fmin, fmax=fmax)

    def extract(self, data: np.ndarray, sfreq=None) -> np.ndarray:
        psds = self.psd_feature.extract(data)
        freqs = self.psd_feature.get_freqs()
        rbps = relative_band_power(psds, freqs, self.freq_bands)
        return rbps 
    
class MbpFeature(FeatureExtractor):
    """Calculate the mean band power (MBP) of the data for each subject."""
    def __init__(self, freq_bands=None, sfreq=sfreq, fmin=0.5, fmax=45):
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
        psds = self.psd_feature.extract(data)
        freqs = self.psd_feature.get_freqs()
        mbps = mean_band_power(psds, freqs, self.freq_bands_list)
        return mbps
    
class AbpFeature(FeatureExtractor):
    """Calculate the absolute band power (ABP) of the data for each subject with normalization options."""
    def __init__(self, freq_bands=None, sfreq=sfreq, fmin=0.5, fmax=45, normalize='minmax'):
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
        abps_normalized = np.zeros_like(abps)
        for band in range(abps.shape[-1]):
            band_data = abps[:, :, band]
            mean = np.mean(band_data)
            std = np.std(band_data)
            abps_normalized[:, :, band] = (band_data - mean) / (std + 1e-8)
        return abps_normalized
    
    def _relative_normalize(self, abps: np.ndarray) -> np.ndarray:
        total_power = np.sum(abps, axis=-1, keepdims=True)
        return abps / (total_power + 1e-8)
    
    def _minmax_normalize(self, abps: np.ndarray) -> np.ndarray:
        abps_normalized = np.zeros_like(abps)
        for i in range(abps.shape[0]):
            for j in range(abps.shape[1]):
                sample = abps[i, j, :]
                min_val = np.min(sample)
                max_val = np.max(sample)
                abps_normalized[i, j, :] = (sample - min_val) / (max_val - min_val + 1e-8)
        return abps_normalized
    
    def extract(self, data: np.ndarray, sfreq=None) -> np.ndarray:
        psds = self.psd_feature.extract(data)
        freqs = self.psd_feature.get_freqs()
        abps = absolute_band_power(psds, freqs, self.freq_bands_list)
        
        if self.normalize == 'zscore':
            abps = self._zscore_normalize(abps)
        elif self.normalize == 'relative':
            abps = self._relative_normalize(abps)
        elif self.normalize == 'minmax':
            abps = self._minmax_normalize(abps)
        return abps

class SccFeature(FeatureExtractor):
    """Calculate the spectral coherence connectivity of the data for each subject."""
    def __init__(self, sfreq=sfreq, fmin=0.5, fmax=45, method='coh'):
        self.sfreq = sfreq 
        self.fmin = fmin 
        self.fmax = fmax 
        self.method = method 
        self.ch_names = ['Fp1' , 'Fp2' , 'F3', 'F4' , 'C3', 'C4', 'P3', 'P4' , 'O1' , 'O2', 
                'F7' , 'F8', 'T3' , 'T4' , 'T5' , 'T6' , 'Fz', 'Cz','Pz']

    def extract(self, data: np.ndarray, sfreq=None, ch_names=None) -> np.ndarray:
        sfreq = self.sfreq 
        fmin = self.fmin 
        fmax = self.fmax 
        cwt_freqs = np.arange(fmin, fmax, 1)
        
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
    
class STFTFeature(FeatureExtractor):
    def __init__(self, sfreq=500, nperseg=500, noverlap=250):
        """
        Extract STFT features from EEG signals using MNE.
        
        Args:
            sfreq (float): Sampling frequency (Hz), default is 500.
            nperseg (int): Length of each segment (number of samples), default is 500.
            noverlap (int): Number of samples to overlap between segments, default is 250.
        """
        self.sfreq = sfreq
        self.nperseg = nperseg
        self.noverlap = noverlap

    def extract(self, data: np.ndarray, sfreq=None) -> np.ndarray:
        """
        Extracts STFT features from EEG data.
        
        Args:
            data (np.ndarray): EEG data with shape (n_epochs, n_channels, n_times).
            sfreq (float, optional): Sampling frequency. If None, uses self.sfreq.
        
        Returns:
            np.ndarray: STFT spectrogram with shape (n_epochs, n_channels, n_freqs, n_times).
        """
        if sfreq is None:
            sfreq = self.sfreq

        if data.ndim == 2:
            data = data[np.newaxis, ...] # For feature extracting with one sample
        
        n_epochs, n_channels, n_times = data.shape
        stft_results = []
        for epoch in range(n_epochs):
            epoch_data = data[epoch]
            stft_data = mne.time_frequency.stft(epoch_data, 
                                                wsize=self.nperseg, 
                                                tstep=self.nperseg - self.noverlap, 
                                                verbose=False)
            magnitude = np.abs(stft_data) # Shape: (n_channels, n_freqs, n_times)
            stft_results.append(magnitude)
        return np.stack(stft_results, axis=0) # (n_epochs, n_channels, n_freqs, n_times)

class CWTFeature(FeatureExtractor):
    def __init__(self, sfreq=500, freqs=np.arange(1, 30, 1), n_cycles=2):
        """
        Extract CWT features from EEG signals using Morlet wavelet with MNE.
        
        Args:
            sfreq (float): Sampling frequency (Hz), default is 500.
            freqs (np.ndarray): Frequency range of interest, default is 0.5-45 Hz with 2 Hz steps.
            n_cycles (int): Number of cycles for the Morlet wavelet, default is 2.
        """
        self.sfreq = sfreq
        self.freqs = freqs
        self.n_cycles = n_cycles

    def extract(self, data: np.ndarray, sfreq=None) -> np.ndarray:
        """
        Extracts CWT features from EEG data.
        
        Args:
            data (np.ndarray): EEG data with shape (n_epochs, n_channels, n_times).
            sfreq (float, optional): Sampling frequency. If None, uses self.sfreq.
        
        Returns:
            np.ndarray: CWT scalogram with shape (n_epochs, n_channels, n_freqs, n_times).
        """
        if sfreq is None:
            sfreq = self.sfreq

        if data.ndim == 2:
            data = data[np.newaxis, ...]
        power = tfr_array_morlet(data, sfreq=sfreq, freqs=self.freqs, n_cycles=self.n_cycles, output='power')
        return power
    
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
    Paper link: https://ieeexplore.ieee.org/document/10179900/
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
        if sfreq is None:
            sfreq = self.sfreq
        
        n_epochs, n_channels, n_samples = data.shape
        T = self.T
        seg_length = n_samples // T  
        B = len(self.freq_bands)
        features_all = []

        for epoch in range(n_epochs):
            epoch_data = data[epoch]
            features_epoch = np.zeros((T, B, n_channels))
            for t in range(T):
                start = t * seg_length
                end = start + seg_length
                segment = epoch_data[:, start:end]

                psds, freqs = mne.time_frequency.psd_array_welch(
                    segment,
                    sfreq=sfreq,
                    fmin=self.fmin,
                    fmax=self.fmax,
                    n_fft=seg_length,
                    verbose=False
                )
                for b_idx, (band, (f_low, f_high)) in enumerate(self.freq_bands.items()):
                    band_idx = np.where((freqs >= f_low) & (freqs <= f_high))[0]
                    psd_band = psds[:, band_idx].sum(axis=1)
                    psd_total = psds.sum(axis=1) + 1e-8
                    rbp = psd_band / psd_total
                    features_epoch[t, b_idx, :] = rbp
            features_all.append(features_epoch)
        features_all = np.array(features_all)
        return features_all
    

class FeaturePipeline:
    """Pipeline to run multiple feature extractors."""
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
            feature = feature[:, :, np.newaxis]
        processed_features.append(feature)
    return np.concatenate(processed_features, axis=2)