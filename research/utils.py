"""Utility functions for UI formatting, adaptive currency display, and style injection."""

from typing import Optional, Union
import streamlit as st
import numpy as np


METRIC_CSS = """
<style>
[data-testid="stMetricValue"] {
    font-size: clamp(1.1rem, 1.8vw, 1.5rem) !important;
    white-space: normal !important;
    word-break: break-word !important;
    overflow: visible !important;
    text-overflow: unset !important;
    line-height: 1.2 !important;
}
[data-testid="stMetricLabel"] {
    font-size: 0.82rem !important;
    color: #8b949e !important;
}
[data-testid="metric-container"] {
    padding: 0.75rem !important;
    background-color: #161b22;
    border: 1px solid #30363d;
    border-radius: 6px;
}
</style>
"""


def inject_metric_css() -> None:
    """Inject institutional dark terminal metric styling to prevent ellipsis truncation."""
    st.markdown(METRIC_CSS, unsafe_allow_html=True)


def format_money(
    val: Union[float, int, None],
    currency: str = "USD",
    compact: bool = False,
) -> str:
    """Adaptive humanized currency formatter handling varying scale dynamics.
    
    - JPY: No fractional cents; values scale to tens of thousands (万) or hundreds of millions (B/億).
    - USD, MYR, EUR, GBP: Two-decimal standard currency or compact scaling (K, M, B).
    - Handles negative values, NaN, and None gracefully.
    """
    if val is None or (isinstance(val, (float, np.floating)) and np.isnan(val)):
        return "N/A"

    symbols = {
        "USD": "$",
        "MYR": "RM ",
        "EUR": "€",
        "GBP": "£",
        "JPY": "¥",
        "LOCAL": "",
    }
    curr_upper = currency.upper().strip() if currency else "USD"
    sym = symbols.get(curr_upper, f"{curr_upper} ")

    is_neg = val < 0
    abs_val = abs(float(val))
    prefix = "-" if is_neg else ""

    if curr_upper == "JPY":
        if compact:
            if abs_val >= 1e8:
                return f"{prefix}{sym}{abs_val / 1e8:.2f}B"
            if abs_val >= 1e4:
                return f"{prefix}{sym}{abs_val / 1e4:.1f}万"
        return f"{prefix}{sym}{abs_val:,.0f}"

    if compact:
        if abs_val >= 1e9:
            return f"{prefix}{sym}{abs_val / 1e9:.2f}B"
        if abs_val >= 1e6:
            return f"{prefix}{sym}{abs_val / 1e6:.2f}M"
        if abs_val >= 1e3:
            return f"{prefix}{sym}{abs_val / 1e3:.1f}K"

    return f"{prefix}{sym}{abs_val:,.2f}"
