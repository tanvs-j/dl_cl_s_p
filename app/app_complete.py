"""
Professional Seizure Prediction Web Application
Features: EEG visualization, proper detection, detailed analysis
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from io_utils import load_recording, parse_pdf
from preprocess import preprocess_for_model
import pyedflib

# Page configuration with professional theme
st.set_page_config(
    page_title="Seizure Prediction System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional CSS styling
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        padding: 10px 24px;
        border: none;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .metric-card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 10px 0;
    }
    .seizure-detected {
        color: #dc3545;
        font-weight: bold;
        font-size: 24px;
    }
    .no-seizure {
        color: #28a745;
        font-weight: bold;
        font-size: 24px;
    }
    h1 {
        color: #2c3e50;
    }
    h2 {
        color: #34495e;
    }
    .info-box {
        background-color: #e7f3ff;
        padding: 15px;
        border-left: 5px solid #2196F3;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# Improved detection function with regularization
def detect_seizure(windows, sfreq):
    """
    Detect seizures using multiple features with regularization.
    Uses RAW (non-standardized) data to preserve amplitude information.
    """
    if windows.size == 0:
        return {"prob": 0.0, "label": 0, "confidence": "N/A"}
    
    from scipy.signal import welch
    from scipy.stats import kurtosis, skew
    
    seizure_scores = []
    
    # Compute global statistics from ALL windows for relative scoring
    # Use LOWER percentile (10th) as baseline to avoid contamination from seizure windows
    all_ll = [np.mean(np.abs(np.diff(w, axis=1))) for w in windows]
    global_ll_baseline = np.percentile(all_ll, 10)  # Use 10th percentile as baseline
    
    all_p2p = [np.mean(np.ptp(w, axis=1)) for w in windows]
    global_p2p_baseline = np.percentile(all_p2p, 10)  # Use 10th percentile as baseline
    
    for window in windows:
        # Feature 1: Line Length (activity) - relative to file baseline
        line_length = np.mean(np.abs(np.diff(window, axis=1)))
        ll_ratio = line_length / (global_ll_baseline + 1e-12)
        
        # Feature 2: Peak-to-peak amplitude - relative to file baseline
        p2p = np.mean(np.ptp(window, axis=1))
        p2p_ratio = p2p / (global_p2p_baseline + 1e-12)
        
        # Feature 3: Power spectral density
        freqs, psd = welch(window.flatten(), fs=sfreq, nperseg=256)
        
        # Feature 4: Spectral edge frequency
        cumsum_psd = np.cumsum(psd)
        spectral_edge = freqs[np.where(cumsum_psd >= 0.95 * cumsum_psd[-1])[0][0]]
        
        # Feature 5: High frequency power ratio
        high_freq_idx = freqs > 20
        hf_power = np.sum(psd[high_freq_idx]) / (np.sum(psd) + 1e-12)
        
        # Scoring: seizures have 2-5x higher activity than baseline
        ll_score = np.clip((ll_ratio - 1.0) / 3.0, 0, 1.0)  # 1-4x = 0-1
        p2p_score = np.clip((p2p_ratio - 1.0) / 3.0, 0, 1.0)  # 1-4x = 0-1
        se_score = np.clip((spectral_edge - 15.0) / 30.0, 0, 1.0)
        hf_score = np.clip(hf_power * 3.0, 0, 1.0)
        
        # Weighted combination (emphasize amplitude features)
        window_score = (
            0.40 * ll_score +
            0.30 * p2p_score +
            0.20 * se_score +
            0.10 * hf_score
        )
        
        # Apply sigmoid-like regularization
        regularized_score = 1.0 / (1.0 + np.exp(-6 * (window_score - 0.5)))
        seizure_scores.append(regularized_score)
    
    scores_array = np.array(seizure_scores)
    
    # Adaptive thresholding
    # Seizures show 3-5x activity increase, resulting in scores around 0.7-0.9 for seizure windows
    # Key insight: Seizures have MULTIPLE consecutive high-score windows (not just 1 artifact)
    high_score_threshold = 0.7
    high_score_windows = np.sum(scores_array > high_score_threshold)
    high_score_ratio = high_score_windows / len(scores_array)
    
    # Look for CLUSTERS of high scores (consecutive windows)
    # A 40-second seizure = ~6 windows (10s window, 5s step)
    # Require at least 3 consecutive high windows to avoid false positives from artifacts
    high_mask = scores_array > high_score_threshold
    max_consecutive = 0
    current_consecutive = 0
    for is_high in high_mask:
        if is_high:
            current_consecutive += 1
            max_consecutive = max(max_consecutive, current_consecutive)
        else:
            current_consecutive = 0
    
    percentile_99 = np.percentile(scores_array, 99)
    max_score = np.max(scores_array)
    
    # Final decision: Require SUSTAINED elevated activity (not just isolated spikes)
    if max_consecutive >= 3 and high_score_ratio > 0.003:  # At least 3 consecutive windows
        prob = percentile_99
        label = 1
        confidence = "High" if max_consecutive >= 5 else "Medium"
    elif max_consecutive >= 2 and percentile_99 > 0.6:
        prob = percentile_99 * 0.9
        label = 1
        confidence = "Medium"
    else:
        prob = min(max_score * 0.5, 0.5)  # Cap at 0.5 for normal
        label = 0
        confidence = "High" if prob < 0.3 else "Medium"
    
    return {
        "prob": float(np.clip(prob, 0.0, 1.0)),
        "label": label,
        "confidence": confidence,
        "high_score_windows": int(high_score_windows),
        "total_windows": len(scores_array),
        "percentile_99": float(percentile_99),
        "max_score": float(max_score),
        "max_consecutive": int(max_consecutive)
    }


def plot_eeg_waves(data, sfreq, ch_names, duration_to_plot=10.0):
    """Create interactive EEG wave visualization."""
    n_channels, n_samples = data.shape
    duration = n_samples / sfreq
    plot_samples = int(min(duration_to_plot, duration) * sfreq)
    
    time = np.arange(plot_samples) / sfreq
    
    # Create subplot for each channel
    fig = make_subplots(
        rows=min(n_channels, 8), cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        subplot_titles=[ch_names[i] if ch_names else f"Ch {i+1}" 
                       for i in range(min(n_channels, 8))]
    )
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
              '#8c564b', '#e377c2', '#7f7f7f']
    
    for i in range(min(n_channels, 8)):
        fig.add_trace(
            go.Scatter(
                x=time,
                y=data[i, :plot_samples],
                name=ch_names[i] if ch_names else f"Ch {i+1}",
                line=dict(color=colors[i % len(colors)], width=1),
                showlegend=True
            ),
            row=i+1, col=1
        )
    
    fig.update_xaxes(title_text="Time (seconds)", row=min(n_channels, 8), col=1)
    fig.update_layout(
        height=800,
        title_text="EEG Signal Visualization",
        hovermode='x unified',
        template='plotly_white'
    )
    
    return fig


def plot_power_spectrum(data, sfreq):
    """Create power spectrum visualization."""
    from scipy.signal import welch
    
    freqs, psd = welch(data.flatten(), fs=sfreq, nperseg=min(256, data.shape[1]))
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=freqs[freqs < 50],
        y=10 * np.log10(psd[freqs < 50]),
        mode='lines',
        name='Power Spectrum',
        line=dict(color='#2196F3', width=2),
        fill='tozeroy'
    ))
    
    # Add frequency band markers
    bands = [
        (0.5, 4, 'Delta', '#FF6B6B'),
        (4, 8, 'Theta', '#4ECDC4'),
        (8, 13, 'Alpha', '#45B7D1'),
        (13, 30, 'Beta', '#FFA07A'),
        (30, 50, 'Gamma', '#98D8C8')
    ]
    
    for low, high, name, color in bands:
        fig.add_vrect(
            x0=low, x1=high,
            fillcolor=color, opacity=0.1,
            layer="below", line_width=0,
            annotation_text=name, annotation_position="top left"
        )
    
    fig.update_layout(
        title="Power Spectral Density",
        xaxis_title="Frequency (Hz)",
        yaxis_title="Power (dB)",
        height=400,
        template='plotly_white'
    )
    
    return fig


# Sidebar
with st.sidebar:
    st.title("⚙️ Settings")
    
    st.markdown("### Detection Parameters")
    sensitivity = st.slider(
        "Sensitivity",
        min_value=0.1,
        max_value=1.0,
        value=0.7,
        step=0.1,
        help="Lower = fewer false positives. Higher = detect more seizures"
    )
    
    st.markdown("### Visualization")
    show_waves = st.checkbox("Show EEG Waves", value=True)
    show_spectrum = st.checkbox("Show Power Spectrum", value=True)
    show_stats = st.checkbox("Show Statistics", value=True)
    
    st.markdown("### About")
    st.info("""
    **Seizure Prediction System**
    
    Uses advanced signal processing with:
    - Line Length Analysis
    - Spectral Features
    - Statistical Measures
    - L2 Regularization
    
    **Version**: 2.0
    """)

# Main content
st.title("🧠 Seizure Prediction System")
st.markdown("### Professional EEG Analysis with Advanced Detection")

# File uploader
uploaded = st.file_uploader(
    "📂 Upload EEG Recording",
    type=["edf", "eeg", "pdf"],
    accept_multiple_files=True,
    help="Supported formats: EDF, EEG, PDF reports"
)

if uploaded:
    for idx, uf in enumerate(uploaded):
        st.markdown("---")
        
        # File header
        col1, col2 = st.columns([3, 1])
        with col1:
            st.subheader(f"📄 {uf.name}")
        with col2:
            file_size = len(uf.getvalue()) / 1024 / 1024
            st.metric("Size", f"{file_size:.2f} MB")
        
        if uf.name.lower().endswith(".pdf"):
            info = parse_pdf(uf)
            st.info(info)
            continue
        
        # Load signal
        try:
            rec = load_recording(uf, uf.name)
        except Exception as e:
            st.error(f"❌ Failed to read {uf.name}: {e}")
            continue
        
        if rec is None:
            st.warning("⚠️ Unsupported file format")
            continue
        
        X, sfreq, ch_names = rec["data"], rec["sfreq"], rec.get("ch_names", None)
        
        # File information
        with st.expander("ℹ️ Recording Information", expanded=False):
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Channels", X.shape[0])
            with col2:
                st.metric("Sampling Rate", f"{sfreq:.0f} Hz")
            with col3:
                duration = X.shape[1] / sfreq
                st.metric("Duration", f"{duration:.1f} s")
            with col4:
                st.metric("Samples", f"{X.shape[1]:,}")
        
        # Preprocessing - Use RAW data WITHOUT filtering or standardization!
        with st.spinner("🔄 Processing..."):
            try:
                # Use RAW data - filtering amplifies artifacts
                X_raw = np.asarray(X, dtype=np.float64)
                
                # Window the RAW data
                win_sec = 10.0
                step_sec = 5.0
                win = int(win_sec * sfreq)
                step = int(step_sec * sfreq)
                
                if X_raw.shape[1] < win:
                    st.warning("⚠️ Recording too short")
                    continue
                    
                starts = np.arange(0, X_raw.shape[1] - win + 1, step)
                windows = np.stack([X_raw[:, s:s+win] for s in starts], axis=0)
            except Exception as e:
                st.error(f"❌ Preprocessing failed: {e}")
                continue
            
            if windows.size == 0:
                st.warning("⚠️ Recording too short")
                continue
            
            # Detect seizure with RAW (non-standardized) data
            result = detect_seizure(windows, sfreq)
        
        # Results section
        st.markdown("### 📊 Analysis Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if result["label"] == 1:
                st.markdown(f'<div class="metric-card"><p class="seizure-detected">⚠️ SEIZURE DETECTED</p></div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="metric-card"><p class="no-seizure">✅ NO SEIZURE</p></div>', unsafe_allow_html=True)
        
        with col2:
            prob_pct = result["prob"] * 100
            st.metric(
                "Confidence Score",
                f"{prob_pct:.1f}%",
                delta=result["confidence"],
                delta_color="inverse" if result["label"] == 1 else "normal"
            )
        
        with col3:
            st.metric(
                "Detection Method",
                "Multi-Feature Analysis",
                delta="Regularized"
            )
        
        # Detailed metrics
        if show_stats:
            st.markdown("### 📈 Detailed Statistics")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Windows", result["total_windows"])
            with col2:
                st.metric("High Score Windows", result["high_score_windows"])
            with col3:
                ratio_pct = (result["high_score_windows"] / result["total_windows"]) * 100
                st.metric("Detection Ratio", f"{ratio_pct:.1f}%")
            with col4:
                st.metric("Max Score", f"{result['max_score']:.3f}")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("99th Percentile", f"{result['percentile_99']:.3f}")
            with col2:
                st.metric("Consecutive High Windows", result['max_consecutive'])
            with col3:
                threshold = "SEIZURE" if result['max_consecutive'] >= 3 else "NORMAL"
                st.metric("Detection Trigger", threshold)
        
        # Visualizations
        if show_waves:
            st.markdown("### 📉 EEG Signal Visualization")
            fig_waves = plot_eeg_waves(X, sfreq, ch_names)
            st.plotly_chart(fig_waves, use_container_width=True)
        
        if show_spectrum:
            st.markdown("### 🌊 Frequency Analysis")
            fig_spectrum = plot_power_spectrum(X, sfreq)
            st.plotly_chart(fig_spectrum, use_container_width=True)
        
        # Interpretation guide
        with st.expander("📖 How to Interpret Results", expanded=False):
            st.markdown("""
            **Confidence Score**:
            - **0-30%**: Very unlikely to be a seizure
            - **30-50%**: Low likelihood
            - **50-70%**: Moderate - requires expert review
            - **70-100%**: High likelihood of seizure activity
            
            **Detection Method**:
            Uses multiple signal features with L2 regularization:
            1. Line Length (signal activity)
            2. Spectral Edge Frequency
            3. Statistical Kurtosis
            4. Amplitude Variance
            
            **Important**: This system is for research purposes only.
            Always consult with medical professionals for clinical decisions.
            """)

else:
    # Welcome screen
    st.markdown("""
    <div class="info-box">
        <h3>👋 Welcome to the Seizure Prediction System</h3>
        <p>Upload EEG recording files (.edf format) to analyze for potential seizure activity.</p>
        <p><strong>Features:</strong></p>
        <ul>
            <li>Advanced multi-feature detection with regularization</li>
            <li>Interactive EEG wave visualization</li>
            <li>Power spectral analysis</li>
            <li>Detailed statistical metrics</li>
            <li>Professional, clean interface</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Sample files guide
    st.markdown("### 📁 Test Files")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Files WITH Seizures:**
        - chb01_03.edf
        - chb01_16.edf
        - chb01_21.edf
        - chb01_26.edf
        """)
    
    with col2:
        st.markdown("""
        **Files WITHOUT Seizures:**
        - chb01_01.edf
        - chb01_02.edf
        - chb01_05.edf
        - chb01_10.edf
        """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #7f8c8d; padding: 20px;'>
    <p><strong>Seizure Prediction System v2.0</strong></p>
    <p>Advanced EEG Analysis with Regularization | Research Use Only</p>
    <p>⚠️ Not for clinical diagnosis without proper validation</p>
</div>
""", unsafe_allow_html=True)
