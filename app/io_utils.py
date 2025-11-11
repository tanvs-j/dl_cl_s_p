from __future__ import annotations
from typing import Optional, Dict, Any
import io
import os
from pathlib import Path
import numpy as np
import mne
import pyedflib
from tempfile import NamedTemporaryFile


def _save_temp_file(file_like, suffix: str) -> str:
    data = file_like.read() if hasattr(file_like, 'read') else file_like
    tf = NamedTemporaryFile(delete=False, suffix=suffix)
    tf.write(data)
    tf.flush()
    tf.close()
    return tf.name


def load_recording(file_like, filename: str) -> Optional[Dict[str, Any]]:
    name = filename.lower()
    if name.endswith('.edf'):
        tmp = _save_temp_file(file_like, '.edf')
        raw = mne.io.read_raw_edf(tmp, preload=True, verbose=False)
        X = raw.get_data()  # shape (ch, samples)
        return {"data": X, "sfreq": float(raw.info['sfreq']), "ch_names": raw.ch_names}
    if name.endswith('.eeg'):
        # Try BrainVision via MNE if headers exist (requires .vhdr)
        try:
            tmp = _save_temp_file(file_like, '.eeg')
            # MNE BrainVision usually needs .vhdr; fall back to EDF reader if content is EDF
            # Try pyedflib as a generic EDF-like reader
            try:
                f = pyedflib.EdfReader(tmp)
                n = f.signals_in_file
                sigbufs = [f.readSignal(i) for i in range(n)]
                sfreq = f.getSampleFrequency(0)
                f.close()
                X = np.vstack(sigbufs)
                return {"data": X, "sfreq": float(sfreq), "ch_names": [str(i) for i in range(n)]}
            except Exception:
                pass
            # As a last attempt, wrap as raw array not supported
            raise RuntimeError("Unknown .eeg format (need BrainVision with headers or EDF-compatible)")
        finally:
            try:
                if 'tmp' in locals() and os.path.exists(tmp):
                    os.remove(tmp)
            except Exception:
                pass
    return None


def parse_pdf(file_like) -> str:
    # We do not extract raw EEG from PDFs; treat as report text only
    try:
        import PyPDF2  # optional dependency not in requirements by default
        reader = PyPDF2.PdfReader(io.BytesIO(file_like.read()))
        text = "\n".join(p.extract_text() or "" for p in reader.pages)
        snippet = (text[:500] + "...") if len(text) > 500 else text
        return "Parsed PDF text (not signals). Please upload .edf/.eeg for signal-based prediction.\n" + snippet
    except Exception:
        return "PDF treated as report; unable to parse text. Please upload .edf/.eeg for prediction."
