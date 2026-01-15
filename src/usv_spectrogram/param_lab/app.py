"""Streamlit UI for exploring USV spectrogram parameters.

This is the main entry point for the Parameter Lab application.
The implementation is split across multiple modules for maintainability.
"""

from __future__ import annotations

import streamlit as st

from usv_spectrogram.config import SpectrogramConfig
from usv_spectrogram.io_wav import get_default_wav_dir
from usv_spectrogram.param_lab.ui.main_content import render_main_content
from usv_spectrogram.param_lab.ui.sidebar import render_sidebar


def run() -> None:
    """Run the Streamlit app for USV parameter exploration."""
    st.set_page_config(page_title="USV Parameter Lab", layout="wide")
    st.title("USV Parameter Lab")
    st.write(
        "Explore STFT parameter effects on a short WAV segment and compare baseline vs variant settings."
    )

    default_cfg = SpectrogramConfig()
    default_wav_dir = get_default_wav_dir()

    with st.sidebar:
        sidebar_state = render_sidebar(default_cfg, default_wav_dir)

    base_cfg = SpectrogramConfig(
        expected_sample_rate_hz=sidebar_state["expected_sample_rate"],
        enforce_sample_rate=sidebar_state["enforce_sample_rate"],
    )

    render_main_content(sidebar_state, base_cfg)
