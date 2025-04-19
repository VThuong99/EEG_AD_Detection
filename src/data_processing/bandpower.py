import numpy as np
from scipy.integrate import simpson 

def freq_ind(freqs,freq_bands):
    """returns list of indices in the freqs array corresponding to the frequencies in freq_bands"""
    indices = []
    for i in range(len(freq_bands)):
        indices.append(np.argmin(np.abs(freqs - freq_bands[i])))
    return indices

def absolute_band_power(psds,freqs,freq_bands,endpoints=freq_ind):
    """
    Computes absolute band power in each frequency band of each EEG channel of row in the psds array.

    Parameters
    ----------
    psds : ndarray
        Array of psds of shape (num_rows) x (num_channels) x len(freqs)
    freqs : ndarray
        1-D array of frequencies.
    freq_bands : array_like
        List of frequencies defining the boundaries of the frequency bands.
    endpoints : 
        Function used to match freq_bands to freqs.

    Returns
    -------
    abps: ndarray
        Array of absolute band power values of shape (num_rows) x (num_channels) x (len(freq_bands)-1)
    """    
    indices = endpoints(freqs,freq_bands)
    absolute_bands_list = []
    for i in range(len(indices)-1):
        absolute_bands_list.append(simpson(psds[...,indices[i]:indices[i+1]+1],x=freqs[indices[i]:indices[i+1]+1],axis=-1))
    return np.transpose(np.array(absolute_bands_list),(1,2,0))

def mean_band_power(psds, freqs, freq_bands, endpoints=freq_ind):
    """
    Computes mean band power (average PSD) in each frequency band of each EEG channel of row in the psds array.

    Parameters
    ----------
    psds : ndarray
        Array of psds of shape (num_rows) x (num_channels) x len(freqs).
    freqs : ndarray
        1-D array of frequencies.
    freq_bands : array_like
        List of frequencies defining the boundaries of the frequency bands.
    endpoints : 
        Function used to match freq_bands to freqs.

    Returns
    -------
    mbps: ndarray
        Array of mean band power values of shape (num_rows) x (num_channels) x (len(freq_bands)-1).
    """
    indices = endpoints(freqs, freq_bands)
    mean_bands_list = []
    for i in range(len(indices)-1):
        # Calculate mean PSD within the frequency band
        mean_band = np.mean(psds[..., indices[i]:indices[i+1]+1], axis=-1)
        mean_bands_list.append(mean_band)
    return np.transpose(np.array(mean_bands_list), (1, 2, 0))

def relative_band_power(psds,freqs,freq_bands,endpoints=freq_ind):
    """
    Computes relative band power in each frequency band of each EEG channel of row in the psds array.

    Parameters
    ----------
    psds : ndarray
        Array of psds of shape (num_rows) x (num_channels) x len(freqs).
    freqs : ndarray
        1-D array of frequencies.
    freq_bands : array_like
        List of frequencies defining the boundaries of the frequency bands.
    endpoints : 
        Function used to match freq_bands to freqs.

    Returns
    -------
    rbps: ndarray
        Array of relative band power values of shape (num_rows) x (num_channels) x (len(freq_bands)-1).
    """    
    indices = endpoints(freqs,freq_bands)
    total_power = np.expand_dims(simpson(psds[...,indices[0]:indices[-1]+1],x=freqs[indices[0]:indices[-1]+1],axis=-1),axis=-1)
    return np.divide(absolute_band_power(psds,freqs,freq_bands,endpoints=endpoints),total_power)