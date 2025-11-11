"""
EDF Reader for Seizure Prediction System
Reads EEG data from .edf files and performs seizure prediction
"""

import numpy as np
import pyedflib
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import sys
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.feature_extractor import EEGFeatureExtractor, FeatureConfig
from src.models.deep_learning_models import create_model, ModelConfig
import torch


class EDFReader:
    """
    Read and process EEG data from EDF files.
    Compatible with standard EEG recording formats.
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize EDF reader with optional trained model.
        
        Args:
            model_path: Path to trained model (.pth file)
        """
        self.feature_extractor = EEGFeatureExtractor(FeatureConfig())
        self.model_config = ModelConfig()
        self.model = None
        
        # Load trained model if provided
        if model_path:
            self.load_model(model_path)
        else:
            # Try to load default trained model
            default_path = Path(__file__).parent.parent.parent / 'data' / 'models' / 'trained_seizure_model.pth'
            if default_path.exists():
                self.load_model(str(default_path))
    
    def load_model(self, model_path: str):
        """Load trained seizure prediction model."""
        try:
            self.model = create_model('cnn', self.model_config)
            self.model.load_state_dict(torch.load(model_path, map_location=self.model_config.device, weights_only=True))
            self.model.eval()
            print(f"✓ Loaded trained model from {model_path}")
        except Exception as e:
            print(f"⚠️  Could not load model: {e}")
            self.model = None
    
    def read_edf(self, file_path: str) -> Dict:
        """
        Read EEG data from EDF file.
        
        Args:
            file_path: Path to .edf file
            
        Returns:
            Dictionary with EEG data and metadata
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"EDF file not found: {file_path}")
        
        if file_path.suffix.lower() != '.edf':
            raise ValueError(f"File must be .edf format, got: {file_path.suffix}")
        
        print(f"📂 Reading EDF file: {file_path.name}")
        
        # Open EDF file
        try:
            edf_file = pyedflib.EdfReader(str(file_path))
        except Exception as e:
            raise RuntimeError(f"Failed to open EDF file: {e}")
        
        # Extract metadata
        n_channels = edf_file.signals_in_file
        signal_labels = edf_file.getSignalLabels()
        sample_frequencies = edf_file.getSampleFrequencies()
        duration = edf_file.getFileDuration()
        
        print(f"   Channels: {n_channels}")
        print(f"   Duration: {duration:.2f} seconds")
        print(f"   Sampling rates: {set(sample_frequencies)} Hz")
        
        # Read all channels
        signals = []
        for i in range(n_channels):
            signal = edf_file.readSignal(i)
            signals.append(signal)
        
        # Get patient and recording info
        header = edf_file.getHeader()
        
        edf_file.close()
        
        # Organize data
        eeg_data = {
            'signals': np.array(signals),
            'labels': signal_labels,
            'sampling_rates': sample_frequencies,
            'n_channels': n_channels,
            'duration': duration,
            'header': header,
            'file_name': file_path.name
        }
        
        print(f"   Shape: {eeg_data['signals'].shape}")
        print(f"✓ EDF file loaded successfully")
        
        return eeg_data
    
    def select_eeg_channels(self, eeg_data: Dict, channel_names: Optional[List[str]] = None) -> np.ndarray:
        """
        Select specific EEG channels from the data.
        
        Args:
            eeg_data: EEG data dictionary from read_edf()
            channel_names: List of channel names to select (e.g., ['Fp1', 'Fp2', 'F3', 'F4'])
                          If None, selects first 18 channels or all available
        
        Returns:
            Selected EEG channels as numpy array
        """
        signals = eeg_data['signals']
        labels = eeg_data['labels']
        
        if channel_names is None:
            # Use first 18 channels or all if less
            n_select = min(18, signals.shape[0])
            selected = signals[:n_select]
            print(f"   Selected first {n_select} channels")
        else:
            # Find matching channels
            selected_indices = []
            for name in channel_names:
                # Case-insensitive matching
                matches = [i for i, label in enumerate(labels) 
                          if name.lower() in label.lower()]
                if matches:
                    selected_indices.append(matches[0])
                else:
                    print(f"   Warning: Channel '{name}' not found")
            
            if not selected_indices:
                raise ValueError(f"None of the requested channels found: {channel_names}")
            
            selected = signals[selected_indices]
            print(f"   Selected {len(selected_indices)} channels: {[labels[i] for i in selected_indices]}")
        
        return selected
    
    def resample_signal(self, signal: np.ndarray, original_fs: float, target_fs: float = 256.0) -> np.ndarray:
        """
        Resample signal to target sampling frequency.
        
        Args:
            signal: Input signal
            original_fs: Original sampling frequency
            target_fs: Target sampling frequency (default 256 Hz)
            
        Returns:
            Resampled signal
        """
        from scipy import signal as scipy_signal
        
        if original_fs == target_fs:
            return signal
        
        num_samples = int(len(signal) * target_fs / original_fs)
        resampled = scipy_signal.resample(signal, num_samples)
        
        return resampled
    
    def preprocess_eeg(self, eeg_data: Dict, target_fs: float = 256.0, 
                       channel_names: Optional[List[str]] = None) -> np.ndarray:
        """
        Preprocess EEG data for prediction.
        
        Args:
            eeg_data: EEG data from read_edf()
            target_fs: Target sampling frequency (default 256 Hz)
            channel_names: Optional list of channel names to use
            
        Returns:
            Preprocessed EEG data (channels × time_samples)
        """
        print("\n🔄 Preprocessing EEG data...")
        
        # Select channels
        selected_signals = self.select_eeg_channels(eeg_data, channel_names)
        
        # Resample each channel to target frequency
        sampling_rates = eeg_data['sampling_rates']
        resampled_signals = []
        
        for i, signal in enumerate(selected_signals):
            original_fs = sampling_rates[i] if i < len(sampling_rates) else sampling_rates[0]
            resampled = self.resample_signal(signal, original_fs, target_fs)
            resampled_signals.append(resampled)
        
        processed = np.array(resampled_signals)
        
        # Ensure we have exactly 18 channels (pad or truncate)
        if processed.shape[0] < 18:
            # Pad with zeros
            padding = np.zeros((18 - processed.shape[0], processed.shape[1]))
            processed = np.vstack([processed, padding])
            print(f"   Padded to 18 channels")
        elif processed.shape[0] > 18:
            # Use first 18
            processed = processed[:18]
            print(f"   Truncated to 18 channels")
        
        print(f"✓ Preprocessed shape: {processed.shape}")
        print(f"   Sampling rate: {target_fs} Hz")
        
        return processed
    
    def predict_seizures(self, eeg_data: np.ndarray, epoch_length: float = 2.0, 
                        sampling_rate: float = 256.0) -> Dict:
        """
        Predict seizures in EEG data.
        
        Args:
            eeg_data: Preprocessed EEG data (18 × time_samples)
            epoch_length: Length of each epoch in seconds
            sampling_rate: Sampling rate of the data
            
        Returns:
            Dictionary with predictions and analysis
        """
        if self.model is None:
            raise RuntimeError("No model loaded. Cannot perform predictions.")
        
        print("\n🧠 Predicting seizures...")
        
        epoch_samples = int(epoch_length * sampling_rate)
        n_epochs = eeg_data.shape[1] // epoch_samples
        
        if n_epochs == 0:
            raise ValueError(f"Recording too short. Need at least {epoch_length}s")
        
        predictions = []
        probabilities = []
        features_list = []
        abnormalities = []
        
        self.feature_extractor.reset_history()
        
        for epoch_idx in range(n_epochs):
            start_sample = epoch_idx * epoch_samples
            end_sample = start_sample + epoch_samples
            current_time = epoch_idx * epoch_length
            
            # Extract epoch
            eeg_epoch = eeg_data[:, start_sample:end_sample]
            
            # Skip if not enough samples
            if eeg_epoch.shape[1] < epoch_samples:
                continue
            
            # Extract features
            features = self.feature_extractor.extract_epoch_features(eeg_epoch)
            features_list.append(features)
            
            # Prepare for model
            eeg_tensor = torch.FloatTensor(eeg_epoch).unsqueeze(0)
            
            # Predict
            with torch.no_grad():
                output = self.model(eeg_tensor)
                probs = torch.softmax(output, dim=1)
                seizure_prob = probs[0, 1].item()
                prediction = torch.argmax(output, dim=1).item()
            
            predictions.append(prediction)
            probabilities.append(seizure_prob)
            
            # Detect abnormalities
            avg_amplitude = np.mean(np.abs(eeg_epoch))
            max_amplitude = np.max(np.abs(eeg_epoch))
            std_amplitude = np.std(eeg_epoch)
            
            if seizure_prob > 0.5 or max_amplitude > 100:
                abnormalities.append({
                    'time': current_time,
                    'epoch': epoch_idx,
                    'type': 'seizure' if prediction == 1 else 'suspicious',
                    'probability': seizure_prob,
                    'avg_amplitude': float(avg_amplitude),
                    'max_amplitude': float(max_amplitude),
                    'std_amplitude': float(std_amplitude),
                    'channels_affected': [int(ch) for ch in range(18) 
                                         if np.max(np.abs(eeg_epoch[ch])) > 80]
                })
        
        # Overall analysis
        seizure_detected = any(p == 1 for p in predictions)
        avg_seizure_prob = np.mean(probabilities)
        max_seizure_prob = np.max(probabilities)
        seizure_epochs = sum(predictions)
        
        result = {
            'seizure_detected': seizure_detected,
            'confidence': float(max_seizure_prob),
            'average_probability': float(avg_seizure_prob),
            'num_epochs': n_epochs,
            'num_seizure_epochs': seizure_epochs,
            'num_abnormal_epochs': len(abnormalities),
            'predictions': predictions,
            'probabilities': [float(p) for p in probabilities],
            'abnormalities': abnormalities,
            'duration': n_epochs * epoch_length,
            'seizure_percentage': (seizure_epochs / n_epochs * 100) if n_epochs > 0 else 0
        }
        
        print(f"✓ Analysis complete")
        print(f"   Epochs analyzed: {n_epochs}")
        print(f"   Seizure detected: {'YES' if seizure_detected else 'NO'}")
        print(f"   Max probability: {max_seizure_prob:.2%}")
        print(f"   Seizure epochs: {seizure_epochs}/{n_epochs}")
        
        return result
    
    def analyze_edf_file(self, file_path: str, channel_names: Optional[List[str]] = None) -> Dict:
        """
        Complete pipeline: read EDF, preprocess, and predict seizures.
        
        Args:
            file_path: Path to .edf file
            channel_names: Optional list of channel names to analyze
            
        Returns:
            Dictionary with full analysis results
        """
        print("=" * 80)
        print("🧠 SEIZURE PREDICTION FROM EDF FILE")
        print("=" * 80)
        
        # Read EDF file
        eeg_data = self.read_edf(file_path)
        
        # Preprocess
        processed_eeg = self.preprocess_eeg(eeg_data, channel_names=channel_names)
        
        # Predict
        predictions = self.predict_seizures(processed_eeg)
        
        # Add metadata
        predictions['file_info'] = {
            'file_name': eeg_data['file_name'],
            'channels': eeg_data['labels'],
            'original_duration': eeg_data['duration'],
            'recording_date': eeg_data['header'].get('startdate', 'Unknown')
        }
        
        print("=" * 80)
        
        return predictions
    
    def generate_report(self, analysis: Dict, output_path: Optional[str] = None) -> str:
        """
        Generate text report from analysis.
        
        Args:
            analysis: Analysis results from analyze_edf_file()
            output_path: Optional path to save report
            
        Returns:
            Report text
        """
        report = []
        report.append("=" * 80)
        report.append("SEIZURE PREDICTION REPORT")
        report.append("=" * 80)
        report.append("")
        
        # File info
        if 'file_info' in analysis:
            info = analysis['file_info']
            report.append("FILE INFORMATION:")
            report.append(f"  File: {info['file_name']}")
            report.append(f"  Duration: {info['original_duration']:.2f} seconds")
            report.append(f"  Recording Date: {info['recording_date']}")
            report.append("")
        
        # Overall results
        report.append("OVERALL RESULTS:")
        report.append(f"  Seizure Detected: {'YES ⚠️' if analysis['seizure_detected'] else 'NO ✓'}")
        report.append(f"  Confidence: {analysis['confidence']:.2%}")
        report.append(f"  Average Probability: {analysis['average_probability']:.2%}")
        report.append(f"  Seizure Percentage: {analysis['seizure_percentage']:.1f}%")
        report.append("")
        
        # Epoch analysis
        report.append("EPOCH ANALYSIS:")
        report.append(f"  Total Epochs: {analysis['num_epochs']}")
        report.append(f"  Seizure Epochs: {analysis['num_seizure_epochs']}")
        report.append(f"  Abnormal Epochs: {analysis['num_abnormal_epochs']}")
        report.append("")
        
        # Abnormalities
        if analysis['abnormalities']:
            report.append("DETECTED ABNORMALITIES:")
            for i, abn in enumerate(analysis['abnormalities'][:10]):  # First 10
                report.append(f"  {i+1}. Time: {abn['time']:.1f}s - "
                            f"Type: {abn['type'].upper()} - "
                            f"Probability: {abn['probability']:.2%}")
            if len(analysis['abnormalities']) > 10:
                report.append(f"  ... and {len(analysis['abnormalities']) - 10} more")
            report.append("")
        
        report.append("=" * 80)
        
        report_text = "\n".join(report)
        
        # Save if output path provided
        if output_path:
            with open(output_path, 'w') as f:
                f.write(report_text)
            print(f"✓ Report saved to: {output_path}")
        
        return report_text


def main():
    """Example usage of EDF Reader."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze EDF files for seizure prediction')
    parser.add_argument('file', help='Path to .edf file')
    parser.add_argument('--model', help='Path to trained model', default=None)
    parser.add_argument('--channels', nargs='+', help='Channel names to analyze', default=None)
    parser.add_argument('--output', help='Output report path', default=None)
    
    args = parser.parse_args()
    
    # Create reader
    reader = EDFReader(model_path=args.model)
    
    # Analyze file
    try:
        results = reader.analyze_edf_file(args.file, channel_names=args.channels)
        
        # Generate report
        report = reader.generate_report(results, output_path=args.output)
        print("\n" + report)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
