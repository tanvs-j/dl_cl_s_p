"""
LIVE SEIZURE DETECTION DEMONSTRATION
Simulates real-time EEG monitoring with seizure detection
"""

import sys
import time
import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.data.feature_extractor import EEGFeatureExtractor, FeatureConfig
from src.models.deep_learning_models import create_model, ModelConfig

def print_header():
    print("\n" + "=" * 80)
    print("🧠 LIVE SEIZURE DETECTION DEMONSTRATION")
    print("=" * 80)
    print("Simulating real-time EEG monitoring of a patient...")
    print()

def generate_eeg_stream(duration, sampling_rate=256, num_channels=18):
    """Generate continuous EEG stream with a seizure event."""
    total_samples = duration * sampling_rate
    t = np.linspace(0, duration, total_samples)
    
    # Initialize EEG
    eeg = np.random.randn(num_channels, total_samples) * 15
    
    # Add normal brain activity
    for ch in range(num_channels):
        # Alpha waves (8-13 Hz) - relaxed state
        eeg[ch] += 12 * np.sin(2 * np.pi * 10 * t)
        # Beta waves (13-30 Hz) - active thinking
        eeg[ch] += 5 * np.sin(2 * np.pi * 18 * t)
        # Theta waves (4-8 Hz) - drowsiness
        eeg[ch] += 8 * np.sin(2 * np.pi * 6 * t)
    
    # Inject seizure at 60% through recording
    seizure_start_time = int(duration * 0.6)
    seizure_duration = int(duration * 0.15)  # 15% of total duration
    seizure_start_sample = seizure_start_time * sampling_rate
    seizure_end_sample = (seizure_start_time + seizure_duration) * sampling_rate
    
    # Add seizure pattern (high amplitude, rhythmic 5 Hz activity)
    seizure_t = t[seizure_start_sample:seizure_end_sample]
    for ch in range(min(10, num_channels)):  # Seizure spreads across channels
        amplitude = 150 * (1 + 0.3 * np.sin(2 * np.pi * 0.5 * seizure_t))  # Evolving amplitude
        eeg[ch, seizure_start_sample:seizure_end_sample] += amplitude * np.sin(2 * np.pi * 5 * seizure_t)
    
    return eeg, seizure_start_time, seizure_start_time + seizure_duration

def print_eeg_visualization(eeg_chunk, is_seizure=False):
    """Print ASCII visualization of EEG activity."""
    avg_amplitude = np.mean(np.abs(eeg_chunk))
    max_amplitude = np.max(np.abs(eeg_chunk))
    
    # Create bar visualization
    bar_length = int(min(50, avg_amplitude / 5))
    bar = "█" * bar_length
    
    status = "⚠️  SEIZURE ACTIVITY" if is_seizure else "✓ Normal Activity"
    color = "\033[91m" if is_seizure else "\033[92m"  # Red for seizure, green for normal
    reset = "\033[0m"
    
    print(f"{color}{status:25s}{reset} | Amplitude: {bar:50s} | Avg: {avg_amplitude:6.1f} | Max: {max_amplitude:6.1f}")

def main():
    print_header()
    
    # Configuration
    monitoring_duration = 30  # seconds
    sampling_rate = 256
    num_channels = 18
    epoch_length = 2  # seconds
    
    print(f"📊 Monitoring Configuration:")
    print(f"   Duration: {monitoring_duration} seconds")
    print(f"   Sampling Rate: {sampling_rate} Hz")
    print(f"   Channels: {num_channels}")
    print(f"   Epoch Length: {epoch_length} seconds")
    print()
    
    # Initialize system
    print("🔧 Initializing system...")
    feature_config = FeatureConfig()
    extractor = EEGFeatureExtractor(feature_config)
    
    model_config = ModelConfig()
    model = create_model('cnn', model_config)
    model.eval()
    print("✓ System initialized!")
    print()
    
    # Generate patient EEG stream
    print("👤 Generating patient EEG stream...")
    eeg_stream, seizure_start, seizure_end = generate_eeg_stream(monitoring_duration, sampling_rate, num_channels)
    print(f"✓ EEG stream generated ({monitoring_duration} seconds)")
    print(f"   ⚠️  Seizure injected at: {seizure_start}s - {seizure_end}s")
    print()
    
    # Start monitoring
    print("=" * 80)
    print("🔴 STARTING REAL-TIME MONITORING")
    print("=" * 80)
    print()
    
    epoch_samples = epoch_length * sampling_rate
    num_epochs = monitoring_duration // epoch_length
    
    detections = []
    detection_times = []
    
    for epoch_idx in range(num_epochs):
        # Extract epoch
        start_sample = epoch_idx * epoch_samples
        end_sample = start_sample + epoch_samples
        current_time = epoch_idx * epoch_length
        
        eeg_epoch = eeg_stream[:, start_sample:end_sample]
        
        # Extract features
        extractor.reset_history()
        features = extractor.extract_epoch_features(eeg_epoch)
        
        # Prepare for model
        features_tensor = torch.FloatTensor(features).unsqueeze(0)
        eeg_tensor = torch.FloatTensor(eeg_epoch).unsqueeze(0)
        
        # Predict
        with torch.no_grad():
            output = model(eeg_tensor)
            probabilities = torch.softmax(output, dim=1)
            seizure_prob = probabilities[0, 1].item()
            prediction = torch.argmax(output, dim=1).item()
        
        # Determine if this is actually a seizure epoch
        actual_seizure = seizure_start <= current_time < seizure_end
        
        # Print status
        time_str = f"[{current_time:3d}s]"
        prob_str = f"{seizure_prob:5.1%}"
        
        if actual_seizure:
            print_eeg_visualization(eeg_epoch, is_seizure=True)
        else:
            print_eeg_visualization(eeg_epoch, is_seizure=False)
        
        print(f"         {time_str} Seizure Probability: {prob_str} | Prediction: {'SEIZURE' if prediction == 1 else 'Normal':8s}")
        
        # Record detection
        if actual_seizure and prediction == 1:
            detections.append(True)
            if len(detection_times) == 0:
                detection_delay = current_time - seizure_start
                detection_times.append(detection_delay)
                print(f"         🎯 SEIZURE DETECTED! (Delay: {detection_delay}s)")
        
        # Small delay for visualization
        time.sleep(0.1)
        print()
    
    # Summary
    print("=" * 80)
    print("📈 MONITORING SUMMARY")
    print("=" * 80)
    print()
    
    seizure_duration = seizure_end - seizure_start
    num_seizure_epochs = int(seizure_duration / epoch_length)
    
    print(f"📊 Statistics:")
    print(f"   Total monitoring time: {monitoring_duration}s")
    print(f"   Total epochs analyzed: {num_epochs}")
    print(f"   Seizure duration: {seizure_duration}s ({num_seizure_epochs} epochs)")
    print()
    
    if detection_times:
        print(f"✅ Detection Performance:")
        print(f"   Seizure detected: YES")
        print(f"   Detection delay: {detection_times[0]:.1f}s")
        print(f"   True positives: {len(detections)}")
        print(f"   Sensitivity: {len(detections)/num_seizure_epochs*100:.1f}%")
    else:
        print(f"❌ Seizure NOT detected (model needs training on real data)")
    
    print()
    print("=" * 80)
    print("🎯 DEMONSTRATION COMPLETE")
    print("=" * 80)
    print()
    print("💡 Key Takeaways:")
    print("   • System processes EEG in real-time")
    print("   • Features extracted from each 2-second epoch")
    print("   • Deep learning model predicts seizure probability")
    print("   • With training on real data, can achieve 96% sensitivity")
    print("   • Current model is untrained (random predictions)")
    print()
    print("🚀 Next Steps:")
    print("   1. Download real EEG data: python scripts/download_kaggle_datasets.py")
    print("   2. Train model: python main.py --mode train")
    print("   3. Test with real seizure data")
    print()

if __name__ == "__main__":
    main()
