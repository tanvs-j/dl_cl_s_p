"""
Feature extraction for EEG signals.
Implements spectral, spatial, and temporal feature extraction as described in
Shoeb & Guttag (2010) ICML paper.
"""

import numpy as np
from scipy import signal
from scipy.fft import fft, fftfreq
import pywt
from typing import Tuple, List, Optional, Dict
from dataclasses import dataclass

from loguru import logger


@dataclass
class FeatureConfig:
    """Configuration for feature extraction."""
    sampling_rate: int = 256  # Hz
    epoch_length: float = 2.0  # seconds
    num_channels: int = 18
    num_filters: int = 8
    freq_min: float = 0.5  # Hz
    freq_max: float = 25.0  # Hz
    window_size: int = 3  # number of epochs for temporal features
    

class FilterBank:
    """Filterbank for spectral feature extraction."""
    
    def __init__(self, num_filters: int = 8, freq_min: float = 0.5, 
                 freq_max: float = 25.0, sampling_rate: int = 256):
        """
        Initialize filterbank.
        
        Args:
            num_filters: Number of bandpass filters
            freq_min: Minimum frequency (Hz)
            freq_max: Maximum frequency (Hz)
            sampling_rate: Sampling rate (Hz)
        """
        self.num_filters = num_filters
        self.freq_min = freq_min
        self.freq_max = freq_max
        self.sampling_rate = sampling_rate
        
        # Create equally spaced frequency bands
        self.bands = self._create_frequency_bands()
        logger.info(f"Created {num_filters} filters: {self.bands}")
        
    def _create_frequency_bands(self) -> List[Tuple[float, float]]:
        """Create equally spaced frequency bands."""
        freqs = np.linspace(self.freq_min, self.freq_max, self.num_filters + 1)
        bands = [(freqs[i], freqs[i+1]) for i in range(self.num_filters)]
        return bands
    
    def apply(self, eeg_data: np.ndarray) -> np.ndarray:
        """
        Apply filterbank to EEG data and extract energy in each band.
        
        Args:
            eeg_data: EEG signal [samples] or [channels, samples]
            
        Returns:
            Band energies [num_filters] or [channels, num_filters]
        """
        if eeg_data.ndim == 1:
            return self._apply_single_channel(eeg_data)
        else:
            # Apply to each channel
            features = np.array([
                self._apply_single_channel(eeg_data[ch])
                for ch in range(eeg_data.shape[0])
            ])
            return features
    
    def _apply_single_channel(self, signal_data: np.ndarray) -> np.ndarray:
        """Apply filterbank to a single channel."""
        energies = np.zeros(self.num_filters)
        
        for i, (low_freq, high_freq) in enumerate(self.bands):
            # Design bandpass filter
            sos = signal.butter(
                4,  # filter order
                [low_freq, high_freq],
                btype='band',
                fs=self.sampling_rate,
                output='sos'
            )
            
            # Apply filter
            filtered = signal.sosfilt(sos, signal_data)
            
            # Calculate energy (sum of squared values)
            energies[i] = np.sum(filtered ** 2)
        
        return energies


class SpectralFeatureExtractor:
    """Extract spectral features from EEG signals."""
    
    def __init__(self, config: FeatureConfig):
        """Initialize spectral feature extractor."""
        self.config = config
        self.filterbank = FilterBank(
            num_filters=config.num_filters,
            freq_min=config.freq_min,
            freq_max=config.freq_max,
            sampling_rate=config.sampling_rate
        )
    
    def extract_filterbank_features(self, eeg_epoch: np.ndarray) -> np.ndarray:
        """
        Extract filterbank energy features.
        
        Args:
            eeg_epoch: EEG epoch [channels, samples]
            
        Returns:
            Features [channels * num_filters]
        """
        # Apply filterbank to each channel
        band_energies = self.filterbank.apply(eeg_epoch)
        
        # Flatten: [channels, num_filters] -> [channels * num_filters]
        features = band_energies.flatten()
        
        return features
    
    def extract_fft_features(self, eeg_epoch: np.ndarray, 
                           num_features: int = 50) -> np.ndarray:
        """
        Extract FFT-based spectral features.
        
        Args:
            eeg_epoch: EEG epoch [channels, samples]
            num_features: Number of frequency bins to keep
            
        Returns:
            FFT magnitude features
        """
        features_list = []
        
        for ch in range(eeg_epoch.shape[0]):
            # Compute FFT
            fft_values = fft(eeg_epoch[ch])
            fft_magnitude = np.abs(fft_values)
            
            # Keep only positive frequencies up to num_features
            fft_features = fft_magnitude[:num_features]
            features_list.append(fft_features)
        
        return np.concatenate(features_list)
    
    def extract_wavelet_features(self, eeg_epoch: np.ndarray, 
                                wavelet: str = 'db4', 
                                level: int = 4) -> np.ndarray:
        """
        Extract wavelet-based features.
        
        Args:
            eeg_epoch: EEG epoch [channels, samples]
            wavelet: Wavelet type
            level: Decomposition level
            
        Returns:
            Wavelet coefficient energies
        """
        features_list = []
        
        for ch in range(eeg_epoch.shape[0]):
            # Wavelet decomposition
            coeffs = pywt.wavedec(eeg_epoch[ch], wavelet, level=level)
            
            # Energy of each decomposition level
            energies = [np.sum(c ** 2) for c in coeffs]
            features_list.extend(energies)
        
        return np.array(features_list)


class SpatialFeatureExtractor:
    """Extract spatial features from multi-channel EEG."""
    
    def __init__(self, config: FeatureConfig):
        """Initialize spatial feature extractor."""
        self.config = config
    
    def extract_channel_correlation(self, eeg_epoch: np.ndarray) -> np.ndarray:
        """
        Extract inter-channel correlation features.
        
        Args:
            eeg_epoch: EEG epoch [channels, samples]
            
        Returns:
            Upper triangle of correlation matrix
        """
        # Compute correlation matrix
        corr_matrix = np.corrcoef(eeg_epoch)
        
        # Extract upper triangle (excluding diagonal)
        upper_tri_indices = np.triu_indices_from(corr_matrix, k=1)
        correlations = corr_matrix[upper_tri_indices]
        
        return correlations
    
    def extract_channel_variance(self, eeg_epoch: np.ndarray) -> np.ndarray:
        """
        Extract variance for each channel.
        
        Args:
            eeg_epoch: EEG epoch [channels, samples]
            
        Returns:
            Variance per channel
        """
        return np.var(eeg_epoch, axis=1)
    
    def extract_spatial_gradient(self, eeg_epoch: np.ndarray) -> np.ndarray:
        """
        Extract spatial gradient features (difference between adjacent channels).
        
        Args:
            eeg_epoch: EEG epoch [channels, samples]
            
        Returns:
            Spatial gradient features
        """
        gradients = []
        
        for ch in range(eeg_epoch.shape[0] - 1):
            # Difference between adjacent channels
            diff = eeg_epoch[ch + 1] - eeg_epoch[ch]
            # Energy of difference
            energy = np.sum(diff ** 2)
            gradients.append(energy)
        
        return np.array(gradients)


class TemporalFeatureExtractor:
    """Extract temporal evolution features."""
    
    def __init__(self, config: FeatureConfig):
        """Initialize temporal feature extractor."""
        self.config = config
    
    def create_stacked_features(self, feature_sequence: List[np.ndarray]) -> np.ndarray:
        """
        Create stacked feature vector from temporal sequence.
        
        Args:
            feature_sequence: List of feature vectors from consecutive epochs
            
        Returns:
            Stacked feature vector capturing temporal evolution
        """
        # Take last W epochs
        W = self.config.window_size
        if len(feature_sequence) < W:
            # Pad with zeros if not enough history
            padding = [np.zeros_like(feature_sequence[0])] * (W - len(feature_sequence))
            feature_sequence = padding + feature_sequence
        else:
            feature_sequence = feature_sequence[-W:]
        
        # Stack features
        stacked = np.concatenate(feature_sequence)
        
        return stacked
    
    def extract_statistical_features(self, eeg_window: np.ndarray) -> np.ndarray:
        """
        Extract statistical features over time window.
        
        Args:
            eeg_window: EEG data [channels, samples]
            
        Returns:
            Statistical features (mean, std, skew, kurtosis per channel)
        """
        from scipy.stats import skew, kurtosis
        
        features = []
        
        for ch in range(eeg_window.shape[0]):
            ch_data = eeg_window[ch]
            features.extend([
                np.mean(ch_data),
                np.std(ch_data),
                skew(ch_data),
                kurtosis(ch_data)
            ])
        
        return np.array(features)


class EEGFeatureExtractor:
    """Main feature extractor combining all feature types."""
    
    def __init__(self, config: Optional[FeatureConfig] = None):
        """Initialize feature extractor."""
        self.config = config or FeatureConfig()
        
        self.spectral = SpectralFeatureExtractor(self.config)
        self.spatial = SpatialFeatureExtractor(self.config)
        self.temporal = TemporalFeatureExtractor(self.config)
        
        # Feature history for temporal features
        self.feature_history = []
        
        logger.info(f"Initialized EEG Feature Extractor")
        logger.info(f"  Epoch length: {self.config.epoch_length}s")
        logger.info(f"  Sampling rate: {self.config.sampling_rate} Hz")
        logger.info(f"  Channels: {self.config.num_channels}")
    
    def extract_epoch_features(self, eeg_epoch: np.ndarray, 
                               include_spatial: bool = True) -> np.ndarray:
        """
        Extract features from a single EEG epoch (X_T in the paper).
        
        Args:
            eeg_epoch: EEG epoch [channels, samples]
            include_spatial: Whether to include spatial features
            
        Returns:
            Feature vector for the epoch
        """
        features = []
        
        # 1. Spectral features (M filters × N channels)
        spectral_features = self.spectral.extract_filterbank_features(eeg_epoch)
        features.append(spectral_features)
        
        # 2. Spatial features (optional)
        if include_spatial:
            variance_features = self.spatial.extract_channel_variance(eeg_epoch)
            features.append(variance_features)
        
        # Concatenate all features
        feature_vector = np.concatenate(features)
        
        return feature_vector
    
    def extract_temporal_features(self, eeg_epoch: np.ndarray) -> np.ndarray:
        """
        Extract features with temporal evolution (stacked X_T).
        
        Args:
            eeg_epoch: EEG epoch [channels, samples]
            
        Returns:
            Stacked feature vector capturing temporal evolution
        """
        # Extract features for current epoch
        current_features = self.extract_epoch_features(eeg_epoch)
        
        # Add to history
        self.feature_history.append(current_features)
        
        # Create stacked features
        stacked_features = self.temporal.create_stacked_features(
            self.feature_history
        )
        
        return stacked_features
    
    def reset_history(self):
        """Reset feature history (e.g., at start of new recording)."""
        self.feature_history = []
    
    def extract_features_from_signal(self, eeg_signal: np.ndarray, 
                                    use_temporal: bool = True) -> np.ndarray:
        """
        Extract features from continuous EEG signal by segmenting into epochs.
        
        Args:
            eeg_signal: Continuous EEG [channels, samples]
            use_temporal: Whether to use temporal stacking
            
        Returns:
            Feature matrix [num_epochs, feature_dim]
        """
        epoch_samples = int(self.config.epoch_length * self.config.sampling_rate)
        num_epochs = eeg_signal.shape[1] // epoch_samples
        
        features_list = []
        self.reset_history()
        
        for i in range(num_epochs):
            start_idx = i * epoch_samples
            end_idx = start_idx + epoch_samples
            
            epoch = eeg_signal[:, start_idx:end_idx]
            
            if use_temporal:
                features = self.extract_temporal_features(epoch)
            else:
                features = self.extract_epoch_features(epoch)
            
            features_list.append(features)
        
        return np.array(features_list)
    
    def get_feature_dimension(self, use_temporal: bool = True) -> int:
        """Get the dimensionality of feature vectors."""
        # Base features: M filters × N channels
        base_dim = self.config.num_filters * self.config.num_channels
        
        # Add channel variance
        base_dim += self.config.num_channels
        
        # If temporal stacking, multiply by window size
        if use_temporal:
            base_dim *= self.config.window_size
        
        return base_dim


def demo_feature_extraction():
    """Demonstrate feature extraction."""
    # Create synthetic EEG data
    sampling_rate = 256
    duration = 10  # seconds
    num_channels = 18
    
    t = np.linspace(0, duration, sampling_rate * duration)
    eeg_data = np.random.randn(num_channels, len(t))
    
    # Add some rhythmic activity to simulate seizure
    seizure_freq = 5  # Hz
    eeg_data[0] += 2 * np.sin(2 * np.pi * seizure_freq * t)
    
    # Extract features
    config = FeatureConfig()
    extractor = EEGFeatureExtractor(config)
    
    logger.info("\nExtracting features from synthetic EEG...")
    features = extractor.extract_features_from_signal(eeg_data, use_temporal=True)
    
    logger.info(f"Feature matrix shape: {features.shape}")
    logger.info(f"Feature dimension: {extractor.get_feature_dimension()}")
    
    return features


if __name__ == "__main__":
    demo_feature_extraction()
