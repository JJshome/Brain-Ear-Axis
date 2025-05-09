#!/usr/bin/env python3
"""
Neural Signal Analysis Module

This module implements methods for processing and analyzing neural signals,
including preprocessing, feature extraction, and time-frequency analysis.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal
from scipy.stats import zscore
import pywt
from sklearn.decomposition import PCA, FastICA

class NeuralSignalProcessor:
    """
    Class for preprocessing and analyzing neural signal data.
    """
    
    def __init__(self, output_dir="results/signal_analysis"):
        """
        Initialize the signal processor.
        
        Parameters:
        -----------
        output_dir : str
            Directory to save analysis results
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.raw_data = None
        self.filtered_data = None
        self.sampling_rate = None
        self.channel_names = None
    
    def load_data(self, data, sampling_rate, channel_names=None):
        """
        Load neural signal data.
        
        Parameters:
        -----------
        data : numpy.ndarray or pandas.DataFrame
            Neural signal data with shape (n_samples, n_channels)
        sampling_rate : float
            Sampling rate in Hz
        channel_names : list, optional
            Names of channels/electrodes
            
        Returns:
        --------
        data : numpy.ndarray
            Loaded data
        """
        if isinstance(data, pd.DataFrame):
            if channel_names is None:
                channel_names = data.columns.tolist()
            data = data.values
        
        self.raw_data = data
        self.sampling_rate = sampling_rate
        self.channel_names = channel_names if channel_names is not None else [f"Channel_{i}" for i in range(data.shape[1])]
        
        print(f"Loaded data with shape {data.shape} and sampling rate {sampling_rate} Hz")
        return data
    
    def preprocess(self, notch_freq=60, bandpass_range=(0.5, 100), 
                  normalize=True, remove_artifacts=True, artifact_threshold=5):
        """
        Preprocess neural signal data with filtering and artifact removal.
        
        Parameters:
        -----------
        notch_freq : float or None
            Frequency to apply notch filter (e.g., 60 Hz for line noise)
        bandpass_range : tuple
            Frequency range for bandpass filter (low, high)
        normalize : bool
            Whether to normalize the data using z-score
        remove_artifacts : bool
            Whether to remove artifacts using amplitude thresholding
        artifact_threshold : float
            Z-score threshold for artifact removal
            
        Returns:
        --------
        filtered_data : numpy.ndarray
            Preprocessed data
        """
        if self.raw_data is None:
            raise ValueError("No data loaded. Call load_data first.")
        
        # Make a copy of the raw data
        processed_data = self.raw_data.copy()
        
        # Apply notch filter to remove line noise
        if notch_freq is not None:
            b_notch, a_notch = signal.iirnotch(
                notch_freq, 
                Q=30, 
                fs=self.sampling_rate
            )
            processed_data = signal.filtfilt(b_notch, a_notch, processed_data, axis=0)
            print(f"Applied notch filter at {notch_freq} Hz")
        
        # Apply bandpass filter
        if bandpass_range is not None:
            low, high = bandpass_range
            nyquist = 0.5 * self.sampling_rate
            low_normalized = low / nyquist
            high_normalized = high / nyquist
            b_bandpass, a_bandpass = signal.butter(
                4, 
                [low_normalized, high_normalized], 
                btype='bandpass'
            )
            processed_data = signal.filtfilt(b_bandpass, a_bandpass, processed_data, axis=0)
            print(f"Applied bandpass filter from {low} to {high} Hz")
        
        # Normalize data
        if normalize:
            processed_data = zscore(processed_data, axis=0)
            print("Applied z-score normalization")
        
        # Remove artifacts
        if remove_artifacts:
            # Convert to z-scores for thresholding
            if not normalize:  # If not already normalized
                z_scores = zscore(processed_data, axis=0)
            else:
                z_scores = processed_data
            
            # Identify artifacts
            artifact_mask = np.abs(z_scores) > artifact_threshold
            
            # Replace artifacts with interpolated values
            for chan in range(processed_data.shape[1]):
                chan_artifacts = artifact_mask[:, chan]
                if np.any(chan_artifacts):
                    # Get indices of artifacts
                    artifact_indices = np.where(chan_artifacts)[0]
                    
                    # For each artifact point, interpolate from neighboring points
                    for idx in artifact_indices:
                        # Find nearest non-artifact points before and after
                        before_indices = np.where(~chan_artifacts[:idx])[0]
                        after_indices = np.where(~chan_artifacts[idx+1:])[0] + idx + 1
                        
                        # If we can find points before and after for interpolation
                        if len(before_indices) > 0 and len(after_indices) > 0:
                            before_idx = before_indices[-1]
                            after_idx = after_indices[0]
                            
                            # Linear interpolation
                            before_val = processed_data[before_idx, chan]
                            after_val = processed_data[after_idx, chan]
                            interp_val = before_val + (after_val - before_val) * (idx - before_idx) / (after_idx - before_idx)
                            
                            processed_data[idx, chan] = interp_val
                        elif len(before_indices) > 0:
                            # If only points before, use the last valid point
                            processed_data[idx, chan] = processed_data[before_indices[-1], chan]
                        elif len(after_indices) > 0:
                            # If only points after, use the first valid point
                            processed_data[idx, chan] = processed_data[after_indices[0], chan]
            
            artifact_count = np.sum(artifact_mask)
            print(f"Removed {artifact_count} artifact points ({artifact_count/processed_data.size*100:.2f}% of data)")
        
        self.filtered_data = processed_data
        return processed_data
    
    def extract_frequency_bands(self, data=None, band_ranges=None):
        """
        Extract frequency band powers using bandpass filtering.
        
        Parameters:
        -----------
        data : numpy.ndarray, optional
            Signal data to analyze. If None, uses filtered_data
        band_ranges : dict, optional
            Dictionary mapping band names to frequency ranges
            
        Returns:
        --------
        band_powers : dict
            Dictionary mapping band names to power values
        """
        if data is None:
            if self.filtered_data is None:
                raise ValueError("No filtered data available. Call preprocess first.")
            data = self.filtered_data
        
        if band_ranges is None:
            # Default frequency bands in Hz
            band_ranges = {
                'delta': (0.5, 4),
                'theta': (4, 8),
                'alpha': (8, 13),
                'beta': (13, 30),
                'gamma': (30, 100)
            }
        
        band_powers = {}
        
        for band_name, (low_freq, high_freq) in band_ranges.items():
            # Bandpass filter for the specific band
            nyquist = 0.5 * self.sampling_rate
            low_normalized = low_freq / nyquist
            high_normalized = high_freq / nyquist
            
            b, a = signal.butter(4, [low_normalized, high_normalized], btype='bandpass')
            band_data = signal.filtfilt(b, a, data, axis=0)
            
            # Compute band power using RMS
            power = np.sqrt(np.mean(band_data**2, axis=0))
            
            band_powers[band_name] = power
        
        # Create a DataFrame with band powers
        band_powers_df = pd.DataFrame(band_powers, index=self.channel_names)
        
        # Save to CSV
        band_powers_df.to_csv(os.path.join(self.output_dir, "frequency_band_powers.csv"))
        
        # Visualize band powers
        plt.figure(figsize=(12, 8))
        band_powers_df.plot(kind='bar')
        plt.title("Frequency Band Powers Across Channels")
        plt.xlabel("Channel")
        plt.ylabel("Power")
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "frequency_band_powers.png"))
        
        return band_powers_df
    
    def compute_spectrogram(self, channel=0, window_size=256, overlap=128, 
                           nfft=512, log_scale=True):
        """
        Compute and plot a spectrogram for a specific channel.
        
        Parameters:
        -----------
        channel : int or str
            Channel index or name
        window_size : int
            Window size for STFT
        overlap : int
            Overlap between windows
        nfft : int
            FFT size
        log_scale : bool
            Whether to use log scale for the frequency axis
            
        Returns:
        --------
        f : numpy.ndarray
            Frequency values
        t : numpy.ndarray
            Time values
        Sxx : numpy.ndarray
            Spectrogram values
        """
        if self.filtered_data is None:
            raise ValueError("No filtered data available. Call preprocess first.")
        
        # Get the channel data
        if isinstance(channel, str):
            if channel in self.channel_names:
                channel_idx = self.channel_names.index(channel)
            else:
                raise ValueError(f"Channel name {channel} not found")
        else:
            channel_idx = channel
        
        channel_data = self.filtered_data[:, channel_idx]
        channel_name = self.channel_names[channel_idx]
        
        # Compute spectrogram
        f, t, Sxx = signal.spectrogram(
            channel_data,
            fs=self.sampling_rate,
            window='hann',
            nperseg=window_size,
            noverlap=overlap,
            nfft=nfft,
            scaling='density'
        )
        
        # Convert to dB
        if log_scale:
            Sxx = 10 * np.log10(Sxx + 1e-10)  # Add small constant to avoid log(0)
        
        # Plot spectrogram
        plt.figure(figsize=(12, 8))
        plt.pcolormesh(t, f, Sxx, shading='gouraud', cmap='viridis')
        plt.title(f"Spectrogram of {channel_name}")
        plt.xlabel("Time (s)")
        plt.ylabel("Frequency (Hz)")
        plt.colorbar(label="Power/Frequency (dB/Hz)" if log_scale else "Power/Frequency (V²/Hz)")
        
        # Use log scale for frequency if requested
        if log_scale:
            plt.yscale('log')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f"spectrogram_channel_{channel_idx}.png"))
        
        return f, t, Sxx
    
    def compute_wavelet_transform(self, channel=0, wavelet='cmor1.5-1.0', 
                                scales=None, scale_type='log'):
        """
        Compute continuous wavelet transform for a specific channel.
        
        Parameters:
        -----------
        channel : int or str
            Channel index or name
        wavelet : str
            Wavelet to use (e.g., 'cmor1.5-1.0', 'morl', 'mexh')
        scales : numpy.ndarray, optional
            Scales for wavelet transform
        scale_type : str
            Scale spacing: 'log' or 'linear'
            
        Returns:
        --------
        scales : numpy.ndarray
            Scale values
        frequencies : numpy.ndarray
            Corresponding frequency values
        coefficients : numpy.ndarray
            Wavelet coefficients
        """
        if self.filtered_data is None:
            raise ValueError("No filtered data available. Call preprocess first.")
        
        # Get the channel data
        if isinstance(channel, str):
            if channel in self.channel_names:
                channel_idx = self.channel_names.index(channel)
            else:
                raise ValueError(f"Channel name {channel} not found")
        else:
            channel_idx = channel
        
        channel_data = self.filtered_data[:, channel_idx]
        channel_name = self.channel_names[channel_idx]
        
        # Generate scales if not provided
        if scales is None:
            if scale_type == 'log':
                scales = np.logspace(0, np.log10(channel_data.shape[0]//4), 100)
            else:
                scales = np.linspace(1, channel_data.shape[0]//4, 100)
        
        # Compute CWT
        coefficients, frequencies = pywt.cwt(channel_data, scales, wavelet, 1/self.sampling_rate)
        
        # Plot scalogram
        plt.figure(figsize=(12, 8))
        time = np.arange(channel_data.shape[0]) / self.sampling_rate
        plt.pcolormesh(time, frequencies, np.abs(coefficients), shading='gouraud', cmap='viridis')
        plt.title(f"Wavelet Transform of {channel_name}")
        plt.xlabel("Time (s)")
        plt.ylabel("Frequency (Hz)")
        plt.colorbar(label="Magnitude")
        plt.yscale('log')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f"wavelet_transform_channel_{channel_idx}.png"))
        
        return scales, frequencies, coefficients
    
    def compute_power_spectral_density(self, method='welch', nperseg=256):
        """
        Compute power spectral density for all channels.
        
        Parameters:
        -----------
        method : str
            Method for PSD estimation: 'welch' or 'periodogram'
        nperseg : int
            Length of each segment for Welch's method
            
        Returns:
        --------
        f : numpy.ndarray
            Frequency values
        psd : numpy.ndarray
            PSD values for each channel
        """
        if self.filtered_data is None:
            raise ValueError("No filtered data available. Call preprocess first.")
        
        # Initialize arrays
        psd_list = []
        
        # Compute PSD for each channel
        for chan_idx in range(self.filtered_data.shape[1]):
            channel_data = self.filtered_data[:, chan_idx]
            
            if method == 'welch':
                f, psd_chan = signal.welch(
                    channel_data, 
                    fs=self.sampling_rate, 
                    nperseg=nperseg
                )
            elif method == 'periodogram':
                f, psd_chan = signal.periodogram(
                    channel_data, 
                    fs=self.sampling_rate
                )
            else:
                raise ValueError(f"Method {method} not supported. Use 'welch' or 'periodogram'")
            
            psd_list.append(psd_chan)
        
        # Convert to array
        psd = np.array(psd_list)
        
        # Plot PSD for each channel
        plt.figure(figsize=(12, 8))
        for chan_idx in range(self.filtered_data.shape[1]):
            plt.semilogy(f, psd[chan_idx], label=self.channel_names[chan_idx])
        
        plt.title(f"Power Spectral Density ({method.capitalize()})")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Power/Frequency (V²/Hz)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f"psd_{method}.png"))
        
        # Save PSD data
        psd_df = pd.DataFrame(
            psd,
            index=self.channel_names,
            columns=f
        )
        psd_df.to_csv(os.path.join(self.output_dir, f"psd_{method}.csv"))
        
        return f, psd
    
    def compute_coherence(self, channel1=0, channel2=1, nperseg=256):
        """
        Compute spectral coherence between two channels.
        
        Parameters:
        -----------
        channel1 : int or str
            First channel index or name
        channel2 : int or str
            Second channel index or name
        nperseg : int
            Length of each segment
            
        Returns:
        --------
        f : numpy.ndarray
            Frequency values
        Cxy : numpy.ndarray
            Coherence values
        """
        if self.filtered_data is None:
            raise ValueError("No filtered data available. Call preprocess first.")
        
        # Get the channel data
        if isinstance(channel1, str):
            if channel1 in self.channel_names:
                channel1_idx = self.channel_names.index(channel1)
            else:
                raise ValueError(f"Channel name {channel1} not found")
        else:
            channel1_idx = channel1
        
        if isinstance(channel2, str):
            if channel2 in self.channel_names:
                channel2_idx = self.channel_names.index(channel2)
            else:
                raise ValueError(f"Channel name {channel2} not found")
        else:
            channel2_idx = channel2
        
        channel1_data = self.filtered_data[:, channel1_idx]
        channel2_data = self.filtered_data[:, channel2_idx]
        
        channel1_name = self.channel_names[channel1_idx]
        channel2_name = self.channel_names[channel2_idx]
        
        # Compute coherence
        f, Cxy = signal.coherence(
            channel1_data,
            channel2_data,
            fs=self.sampling_rate,
            nperseg=nperseg
        )
        
        # Plot coherence
        plt.figure(figsize=(12, 6))
        plt.plot(f, Cxy)
        plt.title(f"Coherence between {channel1_name} and {channel2_name}")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Coherence")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f"coherence_{channel1_idx}_{channel2_idx}.png"))
        
        return f, Cxy
    
    def compute_coherence_matrix(self, freq_range=None, nperseg=256):
        """
        Compute coherence matrix for all channel pairs within a frequency range.
        
        Parameters:
        -----------
        freq_range : tuple, optional
            Frequency range (low, high) for averaging coherence
        nperseg : int
            Length of each segment
            
        Returns:
        --------
        coherence_matrix : numpy.ndarray
            Matrix of coherence values between all channel pairs
        """
        if self.filtered_data is None:
            raise ValueError("No filtered data available. Call preprocess first.")
        
        n_channels = self.filtered_data.shape[1]
        coherence_matrix = np.zeros((n_channels, n_channels))
        
        # Compute coherence for each channel pair
        for i in range(n_channels):
            for j in range(i, n_channels):
                if i == j:
                    coherence_matrix[i, j] = 1.0  # Coherence with self is 1
                else:
                    f, Cxy = signal.coherence(
                        self.filtered_data[:, i],
                        self.filtered_data[:, j],
                        fs=self.sampling_rate,
                        nperseg=nperseg
                    )
                    
                    # Average coherence within frequency range if specified
                    if freq_range is not None:
                        low, high = freq_range
                        mask = (f >= low) & (f <= high)
                        avg_coherence = np.mean(Cxy[mask])
                    else:
                        avg_coherence = np.mean(Cxy)
                    
                    # Symmetric matrix
                    coherence_matrix[i, j] = avg_coherence
                    coherence_matrix[j, i] = avg_coherence
        
        # Plot coherence matrix
        plt.figure(figsize=(10, 8))
        cax = plt.matshow(coherence_matrix, fignum=1, cmap='viridis', vmin=0, vmax=1)
        plt.colorbar(cax, label='Coherence')
        plt.title("Coherence Matrix")
        plt.xticks(range(n_channels), self.channel_names, rotation=90)
        plt.yticks(range(n_channels), self.channel_names)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "coherence_matrix.png"))
        
        # Save as CSV
        coherence_df = pd.DataFrame(
            coherence_matrix,
            index=self.channel_names,
            columns=self.channel_names
        )
        coherence_df.to_csv(os.path.join(self.output_dir, "coherence_matrix.csv"))
        
        return coherence_matrix
    
    def apply_ica(self, n_components=None):
        """
        Apply Independent Component Analysis for source separation.
        
        Parameters:
        -----------
        n_components : int, optional
            Number of ICA components to extract
            
        Returns:
        --------
        components : numpy.ndarray
            ICA components with shape (n_components, n_samples)
        mixing : numpy.ndarray
            Mixing matrix with shape (n_channels, n_components)
        """
        if self.filtered_data is None:
            raise ValueError("No filtered data available. Call preprocess first.")
        
        if n_components is None:
            n_components = min(self.filtered_data.shape[1], 10)  # Default to min(n_channels, 10)
        
        # Apply ICA
        ica = FastICA(n_components=n_components, random_state=42)
        components = ica.fit_transform(self.filtered_data)
        mixing = ica.mixing_
        
        # Plot ICA components
        plt.figure(figsize=(12, n_components * 2))
        time = np.arange(self.filtered_data.shape[0]) / self.sampling_rate
        
        for i in range(n_components):
            plt.subplot(n_components, 1, i + 1)
            plt.plot(time, components[:, i])
            plt.title(f"ICA Component {i+1}")
            plt.xlabel("Time (s)")
            
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "ica_components.png"))
        
        # Plot mixing matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            mixing, 
            cmap='coolwarm', 
            xticklabels=[f"IC{i+1}" for i in range(n_components)],
            yticklabels=self.channel_names
        )
        plt.title("ICA Mixing Matrix")
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "ica_mixing_matrix.png"))
        
        # Save components and mixing matrix
        component_df = pd.DataFrame(
            components, 
            columns=[f"IC{i+1}" for i in range(n_components)]
        )
        component_df.to_csv(os.path.join(self.output_dir, "ica_components.csv"), index=False)
        
        mixing_df = pd.DataFrame(
            mixing,
            index=self.channel_names,
            columns=[f"IC{i+1}" for i in range(n_components)]
        )
        mixing_df.to_csv(os.path.join(self.output_dir, "ica_mixing_matrix.csv"))
        
        return components, mixing
    
    def apply_pca(self, n_components=None):
        """
        Apply Principal Component Analysis for dimensionality reduction.
        
        Parameters:
        -----------
        n_components : int, optional
            Number of PCA components to extract
            
        Returns:
        --------
        components : numpy.ndarray
            Principal components with shape (n_samples, n_components)
        explained_variance : numpy.ndarray
            Explained variance ratio for each component
        """
        if self.filtered_data is None:
            raise ValueError("No filtered data available. Call preprocess first.")
        
        if n_components is None:
            n_components = min(self.filtered_data.shape[1], 10)  # Default to min(n_channels, 10)
        
        # Apply PCA
        pca = PCA(n_components=n_components)
        components = pca.fit_transform(self.filtered_data)
        explained_variance = pca.explained_variance_ratio_
        
        # Plot explained variance
        plt.figure(figsize=(10, 6))
        plt.bar(range(1, n_components + 1), explained_variance)
        plt.plot(range(1, n_components + 1), np.cumsum(explained_variance), 'r-')
        plt.xlabel("Principal Component")
        plt.ylabel("Explained Variance Ratio")
        plt.title("PCA Explained Variance")
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "pca_explained_variance.png"))
        
        # Plot PCA components
        plt.figure(figsize=(12, n_components * 2))
        time = np.arange(self.filtered_data.shape[0]) / self.sampling_rate
        
        for i in range(n_components):
            plt.subplot(n_components, 1, i + 1)
            plt.plot(time, components[:, i])
            plt.title(f"PC {i+1} ({explained_variance[i]*100:.2f}% explained variance)")
            plt.xlabel("Time (s)")
            
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "pca_components.png"))
        
        # Save components and explained variance
        component_df = pd.DataFrame(
            components, 
            columns=[f"PC{i+1}" for i in range(n_components)]
        )
        component_df.to_csv(os.path.join(self.output_dir, "pca_components.csv"), index=False)
        
        variance_df = pd.DataFrame({
            'component': [f"PC{i+1}" for i in range(n_components)],
            'explained_variance': explained_variance,
            'cumulative_variance': np.cumsum(explained_variance)
        })
        variance_df.to_csv(os.path.join(self.output_dir, "pca_explained_variance.csv"), index=False)
        
        return components, explained_variance
    
    def extract_features(self, window_size=None, overlap=0.5):
        """
        Extract time-domain and frequency-domain features from the signal.
        
        Parameters:
        -----------
        window_size : int, optional
            Size of the window for feature extraction (in samples)
        overlap : float
            Overlap between consecutive windows (0 to 1)
            
        Returns:
        --------
        features_df : pandas.DataFrame
            DataFrame with extracted features
        """
        if self.filtered_data is None:
            raise ValueError("No filtered data available. Call preprocess first.")
        
        # Set default window size if not provided
        if window_size is None:
            window_size = int(self.sampling_rate)  # 1-second window by default
        
        # Calculate step size based on overlap
        step_size = int(window_size * (1 - overlap))
        
        # Calculate number of windows
        n_samples = self.filtered_data.shape[0]
        n_windows = max(1, (n_samples - window_size) // step_size + 1)
        
        # Initialize feature lists
        features = []
        
        # Extract features for each window
        for win_idx in range(n_windows):
            start_idx = win_idx * step_size
            end_idx = start_idx + window_size
            
            if end_idx > n_samples:
                break
                
            window_data = self.filtered_data[start_idx:end_idx, :]
            
            # Time information
            time_start = start_idx / self.sampling_rate
            time_end = end_idx / self.sampling_rate
            
            # Extract features for each channel
            for chan_idx in range(self.filtered_data.shape[1]):
                chan_data = window_data[:, chan_idx]
                chan_name = self.channel_names[chan_idx]
                
                # Time-domain features
                mean = np.mean(chan_data)
                std = np.std(chan_data)
                rms = np.sqrt(np.mean(chan_data**2))
                kurtosis = np.mean((chan_data - mean)**4) / (std**4) if std > 0 else 0
                skewness = np.mean((chan_data - mean)**3) / (std**3) if std > 0 else 0
                
                # Calculate zero crossings
                zero_crossings = np.sum(np.diff(np.signbit(chan_data)))
                
                # Frequency-domain features
                f, psd = signal.welch(
                    chan_data, 
                    fs=self.sampling_rate, 
                    nperseg=min(256, len(chan_data))
                )
                
                # Calculate band powers
                band_ranges = {
                    'delta': (0.5, 4),
                    'theta': (4, 8),
                    'alpha': (8, 13),
                    'beta': (13, 30),
                    'gamma': (30, min(100, self.sampling_rate/2))
                }
                
                band_powers = {}
                for band_name, (low_freq, high_freq) in band_ranges.items():
                    # Find indices corresponding to the frequency band
                    idx = np.logical_and(f >= low_freq, f <= high_freq)
                    if np.any(idx):
                        band_powers[band_name] = np.mean(psd[idx])
                    else:
                        band_powers[band_name] = 0
                
                # Spectral edge frequency (95% of power)
                total_power = np.sum(psd)
                if total_power > 0:
                    cumsum_power = np.cumsum(psd) / total_power
                    sef_95_idx = np.argmax(cumsum_power >= 0.95)
                    sef_95 = f[sef_95_idx]
                else:
                    sef_95 = 0
                
                # Spectral entropy
                psd_norm = psd / (np.sum(psd) + 1e-10)
                spectral_entropy = -np.sum(psd_norm * np.log2(psd_norm + 1e-10))
                
                # Add to features list
                features.append({
                    'window_start': time_start,
                    'window_end': time_end,
                    'channel': chan_name,
                    'mean': mean,
                    'std': std,
                    'rms': rms,
                    'kurtosis': kurtosis,
                    'skewness': skewness,
                    'zero_crossings': zero_crossings,
                    'delta_power': band_powers['delta'],
                    'theta_power': band_powers['theta'],
                    'alpha_power': band_powers['alpha'],
                    'beta_power': band_powers['beta'],
                    'gamma_power': band_powers['gamma'],
                    'sef_95': sef_95,
                    'spectral_entropy': spectral_entropy
                })
        
        # Create a DataFrame
        features_df = pd.DataFrame(features)
        
        # Save features
        features_df.to_csv(os.path.join(self.output_dir, "signal_features.csv"), index=False)
        
        return features_df


def run_signal_analysis(data, sampling_rate, channel_names=None, output_dir="results/signal_analysis"):
    """
    Run the complete signal analysis workflow.
    
    Parameters:
    -----------
    data : numpy.ndarray or pandas.DataFrame
        Neural signal data with shape (n_samples, n_channels)
    sampling_rate : float
        Sampling rate in Hz
    channel_names : list, optional
        Names of channels/electrodes
    output_dir : str
        Directory to save results
        
    Returns:
    --------
    processor : NeuralSignalProcessor
        The signal processor object
    """
    # Initialize signal processor
    processor = NeuralSignalProcessor(output_dir=output_dir)
    
    # Load data
    processor.load_data(data, sampling_rate, channel_names)
    
    # Preprocess data
    processor.preprocess(
        notch_freq=60,
        bandpass_range=(0.5, 100),
        normalize=True,
        remove_artifacts=True
    )
    
    # Extract frequency bands
    processor.extract_frequency_bands()
    
    # Compute PSD
    processor.compute_power_spectral_density()
    
    # Compute coherence matrix
    processor.compute_coherence_matrix()
    
    # Apply ICA
    processor.apply_ica()
    
    # Apply PCA
    processor.apply_pca()
    
    # Extract features
    processor.extract_features()
    
    # Compute spectrogram for first channel
    processor.compute_spectrogram(channel=0)
    
    # Compute wavelet transform for first channel
    processor.compute_wavelet_transform(channel=0)
    
    print("Signal analysis completed successfully!")
    print(f"Results saved to {output_dir}")
    
    return processor


if __name__ == "__main__":
    # Example usage
    import sys
    sys.path.append("../..")  # Add project root to path
    
    from data.loaders import load_neural_data
    
    # Load example data
    data = load_neural_data("data/raw/neural_signals.csv")
    
    # Example channel names
    channel_names = [
        "Auditory_Cortex_L", "Auditory_Cortex_R",
        "Cochlear_Nucleus_L", "Cochlear_Nucleus_R", 
        "Superior_Olivary_Complex_L", "Superior_Olivary_Complex_R",
        "Inferior_Colliculus_L", "Inferior_Colliculus_R"
    ]
    
    # Run analysis
    processor = run_signal_analysis(
        data,
        sampling_rate=1000,  # 1000 Hz
        channel_names=channel_names,
        output_dir="results/neural_signal_analysis"
    )
