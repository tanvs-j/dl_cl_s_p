"""
Demo Script: EDF Reader for Seizure Prediction
Shows how to use the EDF reader with synthetic data
"""

import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from src.data.edf_reader import EDFReader

def create_sample_edf():
    """
    Create a sample EDF file for demonstration.
    This generates synthetic EEG data with a simulated seizure.
    """
    try:
        import pyedflib
    except ImportError:
        print("⚠️  pyedflib not installed. Installing...")
        import subprocess
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'pyedflib'])
        import pyedflib
    
    print("📝 Creating sample EDF file...")
    
    # Configuration
    duration = 60  # 60 seconds
    sampling_rate = 256
    n_channels = 18
    
    # Standard 10-20 channel names
    channel_names = [
        'Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4',
        'O1', 'O2', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6',
        'Fz', 'Cz'
    ]
    
    # Generate synthetic EEG data
    n_samples = duration * sampling_rate
    t = np.linspace(0, duration, n_samples)
    
    signals = []
    for ch in range(n_channels):
        # Normal EEG: mix of rhythms
        signal = np.random.randn(n_samples) * 10  # Noise
        signal += 15 * np.sin(2 * np.pi * 10 * t)  # Alpha (10 Hz)
        signal += 8 * np.sin(2 * np.pi * 18 * t)   # Beta (18 Hz)
        signal += 12 * np.sin(2 * np.pi * 6 * t)   # Theta (6 Hz)
        
        # Add seizure activity from 20-30 seconds
        seizure_start = int(20 * sampling_rate)
        seizure_end = int(30 * sampling_rate)
        
        if ch < 10:  # Affect 10 channels
            # High amplitude, rhythmic seizure pattern
            seizure_freq = 5  # 5 Hz spike-wave
            amplitude = 150 * (1 + 0.3 * np.sin(2 * np.pi * 0.2 * t[seizure_start:seizure_end]))
            signal[seizure_start:seizure_end] += amplitude * np.sin(
                2 * np.pi * seizure_freq * t[seizure_start:seizure_end]
            )
        
        signals.append(signal)
    
    # Create EDF file
    output_path = Path('data/samples/sample_eeg.edf')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # EDF file configuration
    edf_file = pyedflib.EdfWriter(str(output_path), n_channels, file_type=pyedflib.FILETYPE_EDF)
    
    # Set channel information
    channel_info = []
    for i in range(n_channels):
        ch_dict = {
            'label': channel_names[i],
            'dimension': 'uV',
            'sample_frequency': sampling_rate,  # Updated for pyedflib compatibility
            'physical_max': 200.0,
            'physical_min': -200.0,
            'digital_max': 32767,
            'digital_min': -32768,
            'transducer': '',
            'prefilter': ''
        }
        channel_info.append(ch_dict)
    
    edf_file.setSignalHeaders(channel_info)
    
    # Set patient info
    edf_file.setPatientName('Demo Patient')
    edf_file.setPatientCode('DEMO001')
    edf_file.setRecordingAdditional('Synthetic EEG for demonstration')
    
    # Write signals
    edf_file.writeSamples(signals)
    edf_file.close()
    
    print(f"✓ Sample EDF created: {output_path}")
    print(f"   Duration: {duration} seconds")
    print(f"   Channels: {n_channels}")
    print(f"   Sampling rate: {sampling_rate} Hz")
    print(f"   Seizure: 20-30 seconds")
    
    return str(output_path)


def demo_basic_usage(edf_path):
    """Demonstrate basic EDF reader usage."""
    print("\n" + "="*80)
    print("DEMO 1: Basic Usage")
    print("="*80)
    
    # Create reader
    reader = EDFReader()
    
    # Analyze file
    results = reader.analyze_edf_file(edf_path)
    
    # Display results
    print(f"\n📊 Results Summary:")
    print(f"   Seizure detected: {'YES ⚠️' if results['seizure_detected'] else 'NO ✓'}")
    print(f"   Confidence: {results['confidence']:.1%}")
    print(f"   Duration: {results['duration']:.1f} seconds")
    print(f"   Epochs analyzed: {results['num_epochs']}")
    print(f"   Seizure epochs: {results['num_seizure_epochs']}")
    
    # Generate report
    report = reader.generate_report(results)
    print("\n" + report)


def demo_step_by_step(edf_path):
    """Demonstrate step-by-step processing."""
    print("\n" + "="*80)
    print("DEMO 2: Step-by-Step Processing")
    print("="*80)
    
    reader = EDFReader()
    
    # Step 1: Read
    print("\n1️⃣ Reading EDF file...")
    eeg_data = reader.read_edf(edf_path)
    print(f"   ✓ Loaded {eeg_data['n_channels']} channels")
    print(f"   ✓ Duration: {eeg_data['duration']:.2f} seconds")
    print(f"   ✓ Channels: {', '.join(eeg_data['labels'][:5])}...")
    
    # Step 2: Preprocess
    print("\n2️⃣ Preprocessing...")
    processed = reader.preprocess_eeg(eeg_data)
    print(f"   ✓ Shape: {processed.shape}")
    print(f"   ✓ Mean amplitude: {np.mean(np.abs(processed)):.2f} µV")
    print(f"   ✓ Max amplitude: {np.max(np.abs(processed)):.2f} µV")
    
    # Step 3: Predict
    print("\n3️⃣ Predicting seizures...")
    predictions = reader.predict_seizures(processed)
    print(f"   ✓ Analyzed {predictions['num_epochs']} epochs")
    print(f"   ✓ Found {predictions['num_abnormal_epochs']} abnormalities")
    
    # Step 4: Examine abnormalities
    if predictions['abnormalities']:
        print("\n4️⃣ Abnormality Details:")
        for i, abn in enumerate(predictions['abnormalities'][:5]):
            print(f"   {i+1}. Time {abn['time']:.1f}s: "
                  f"{abn['type'].upper()} "
                  f"(prob={abn['probability']:.1%}, "
                  f"amplitude={abn['max_amplitude']:.1f}µV)")


def demo_custom_channels(edf_path):
    """Demonstrate custom channel selection."""
    print("\n" + "="*80)
    print("DEMO 3: Custom Channel Selection")
    print("="*80)
    
    reader = EDFReader()
    
    # First, see what channels are available
    eeg_data = reader.read_edf(edf_path)
    print(f"\n📋 Available channels:")
    for i, label in enumerate(eeg_data['labels']):
        print(f"   {i+1:2d}. {label}")
    
    # Analyze with specific channels
    print(f"\n🎯 Analyzing frontal channels only...")
    frontal_channels = ['Fp1', 'Fp2', 'F3', 'F4', 'F7', 'F8']
    
    results = reader.analyze_edf_file(edf_path, channel_names=frontal_channels)
    
    print(f"\n📊 Results:")
    print(f"   Seizure detected: {'YES ⚠️' if results['seizure_detected'] else 'NO ✓'}")
    print(f"   Confidence: {results['confidence']:.1%}")


def demo_batch_processing():
    """Demonstrate processing multiple files."""
    print("\n" + "="*80)
    print("DEMO 4: Batch Processing")
    print("="*80)
    
    # Check for EDF files in data directory
    edf_dir = Path('data/samples')
    edf_files = list(edf_dir.glob('*.edf')) if edf_dir.exists() else []
    
    if not edf_files:
        print("\n⚠️  No EDF files found in data/samples/")
        print("   This demo requires multiple EDF files")
        return
    
    print(f"\n📁 Found {len(edf_files)} EDF file(s)")
    
    reader = EDFReader()
    results_all = []
    
    for edf_file in edf_files:
        print(f"\n📂 Processing {edf_file.name}...")
        try:
            results = reader.analyze_edf_file(str(edf_file))
            results_all.append(results)
            
            status = "SEIZURE" if results['seizure_detected'] else "NORMAL"
            print(f"   Status: {status} (confidence: {results['confidence']:.1%})")
        except Exception as e:
            print(f"   Error: {e}")
    
    # Summary
    if results_all:
        seizure_count = sum(r['seizure_detected'] for r in results_all)
        print(f"\n📊 Summary:")
        print(f"   Total files: {len(results_all)}")
        print(f"   Seizures detected: {seizure_count}")
        print(f"   Normal: {len(results_all) - seizure_count}")


def main():
    """Run all demonstrations."""
    print("="*80)
    print("🧠 EDF READER DEMONSTRATION")
    print("="*80)
    print("\nThis demo shows how to use the EDF reader for seizure prediction.")
    print("It will create a synthetic EDF file and analyze it.")
    
    # Check if pyedflib is installed
    try:
        import pyedflib
        print("\n✓ pyedflib is installed")
    except ImportError:
        print("\n⚠️  pyedflib not installed")
        print("Installing pyedflib...")
        import subprocess
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'pyedflib'])
        print("✓ pyedflib installed")
    
    # Create sample EDF file
    try:
        edf_path = create_sample_edf()
    except Exception as e:
        print(f"\n❌ Failed to create sample EDF: {e}")
        print("   Please ensure pyedflib is properly installed")
        return 1
    
    # Run demonstrations
    try:
        demo_basic_usage(edf_path)
        demo_step_by_step(edf_path)
        demo_custom_channels(edf_path)
        demo_batch_processing()
        
        print("\n" + "="*80)
        print("✅ ALL DEMONSTRATIONS COMPLETE")
        print("="*80)
        print(f"\nSample EDF file saved at: {edf_path}")
        print("You can use this file to test the EDF reader.")
        print("\nTry:")
        print(f"  python src\\data\\edf_reader.py {edf_path}")
        print(f"  python src\\data\\edf_reader.py {edf_path} --output report.txt")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
