"""Test that Parameter Lab modules import correctly."""

from __future__ import annotations


def test_app_imports():
    """Verify app.py imports without errors."""
    from usv_spectrogram.param_lab.app import run
    assert callable(run)


def test_plotting_imports():
    """Verify plotting module imports."""
    from usv_spectrogram.param_lab.plotting import plot_difference, plot_spectrogram
    assert callable(plot_spectrogram)
    assert callable(plot_difference)


def test_state_imports():
    """Verify state module imports."""
    from usv_spectrogram.param_lab.state import (
        compute_segment_spectrogram,
        config_from_controls,
        parse_sweep_values,
        read_wav_info,
        stft_cache_key,
    )
    assert callable(read_wav_info)
    assert callable(compute_segment_spectrogram)
    assert callable(config_from_controls)
    assert callable(stft_cache_key)
    assert callable(parse_sweep_values)


def test_ui_components_imports():
    """Verify UI components module imports."""
    from usv_spectrogram.param_lab.ui.components import (
        WINDOW_OPTIONS,
        render_parameter_controls,
    )
    assert callable(render_parameter_controls)
    assert isinstance(WINDOW_OPTIONS, tuple)


def test_ui_sidebar_imports():
    """Verify sidebar module imports."""
    from usv_spectrogram.param_lab.ui.sidebar import render_sidebar
    assert callable(render_sidebar)


def test_ui_main_content_imports():
    """Verify main_content module imports."""
    from usv_spectrogram.param_lab.ui.main_content import (
        render_diff_view,
        render_main_content,
        render_parameter_explanations,
        render_spectrogram_panel,
        render_sweep_export,
    )
    assert callable(render_spectrogram_panel)
    assert callable(render_diff_view)
    assert callable(render_parameter_explanations)
    assert callable(render_sweep_export)
    assert callable(render_main_content)
