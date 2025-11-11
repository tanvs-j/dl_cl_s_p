import streamlit as st
from typing import List, Tuple
import numpy as np
from io_utils import load_recording, parse_pdf
from inference import InferenceEngine
from preprocess import preprocess_for_model

st.set_page_config(page_title="Seizure Predictor", layout="wide")
st.title("Seizure Predictor (Deep + Continual Learning)")

with st.sidebar:
    st.markdown("Upload .edf or supported .eeg for best results. PDFs are treated as reports.")
    st.markdown("Model: 1D-CNN + online continual learning")

engine = InferenceEngine()

uploaded = st.file_uploader(
    "Upload EEG files (.edf, .eeg) or reports (.pdf)",
    type=["edf", "eeg", "pdf"],
    accept_multiple_files=True,
)

if uploaded:
    for uf in uploaded:
        st.divider()
        st.subheader(f"File: {uf.name}")
        if uf.name.lower().endswith(".pdf"):
            info = parse_pdf(uf)
            st.info(info)
            continue
        # Load signal
        try:
            rec = load_recording(uf, uf.name)
        except Exception as e:
            st.error(f"Failed to read {uf.name}: {e}")
            continue
        if rec is None:
            st.warning("Unsupported or unreadable file. Please provide .edf or BrainVision .eeg with headers.")
            continue
        X, sfreq, ch_names = rec["data"], rec["sfreq"], rec.get("ch_names", None)
        st.caption(f"Shape: {X.shape} (channels x samples), sfreq={sfreq:.2f} Hz")

        # Preprocess and window for the model
        try:
            windows = preprocess_for_model(X, sfreq)
        except Exception as e:
            st.error(f"Preprocessing failed: {e}")
            continue
        if windows.size == 0:
            st.warning("Recording too short after preprocessing.")
            continue

        # Predict
        pred = engine.predict(windows, sfreq)
        label = "SEIZURE" if pred["label"] == 1 else "NO SEIZURE"
        prob = pred["prob"]
        st.markdown(f"### Prediction: {label} (p={prob:.2f})")
        if not pred.get("model_loaded", False):
            st.info("Deep model not trained yet. Using heuristic. Provide training data and run training for better accuracy.")

        # Optional: user correction and online update
        with st.expander("Correct label / improve model"):
            user_label = st.radio("True label:", ["No Seizure", "Seizure"], index=1 if pred["label"] == 1 else 0, horizontal=True, key=f"radio_{uf.name}")
            if st.button("Improve model with this file", key=f"button_{uf.name}"):
                y = 1 if user_label == "Seizure" else 0
                try:
                    updated = engine.online_update(windows, y)
                    if updated:
                        st.success("Model updated with online learning.")
                    else:
                        st.info("Model not yet trained; cannot update. Train offline first.")
                except Exception as e:
                    st.error(f"Online update failed: {e}")
