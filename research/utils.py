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


def render_data_freshness_badge(meta: Optional[dict] = None) -> None:
    """Render a compact status badge for data freshness, cache policy, and lookback horizon."""
    if meta is None:
        try:
            from data.loader import load_cache_metadata
            meta = load_cache_metadata()
        except Exception:
            meta = {}

    last_sync = meta.get("last_refresh_timestamp", st.session_state.get("last_sync_time", "N/A"))
    ttl_label = st.session_state.get("cache_ttl_label", meta.get("active_ttl_label", "4 Hours"))
    horizon = st.session_state.get("lookback_horizon", "5Y")
    start_d = st.session_state.get("start_date", "")
    end_d = st.session_state.get("end_date", "")
    status = meta.get("status", "Cached")

    st.markdown(
        f"""
        <div style="background-color: #131722; border: 1px solid #2a2e39; border-radius: 4px;
                    padding: 8px 14px; margin-top: 4px; margin-bottom: 16px; display: flex;
                    justify-content: space-between; align-items: center; font-size: 12px; font-family: monospace;">
            <div>
                <span style="color: #00c805; font-size: 14px;">●</span>
                <span style="color: #848e9c;"> Data Freshness:</span>
                <span style="color: #ffffff; font-weight: 600;">{last_sync}</span>
                <span style="color: #4a5568; margin: 0 8px;">|</span>
                <span style="color: #848e9c;">Policy:</span> <span style="color: #60a5fa; font-weight: 600;">{ttl_label}</span>
                <span style="color: #4a5568; margin: 0 8px;">|</span>
                <span style="color: #848e9c;">Horizon:</span> <span style="color: #fbbf24; font-weight: 700;">{horizon}</span>
                <span style="color: #64748b;">({start_d} to {end_d})</span>
            </div>
            <div style="color: #00c805; font-weight: 700; letter-spacing: 0.5px;">
                STATUS: {status}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_backfill_warning_badge(meta: Optional[dict] = None) -> None:
    """Surfaces an informational warning badge if instruments have fewer than 10 years and are proxy-backfilled."""
    if meta is None:
        try:
            from data.loader import load_cache_metadata
            meta = load_cache_metadata()
        except Exception:
            meta = {}

    backfilled = meta.get("backfilled_tickers", {})
    if not backfilled:
        return

    items_desc = ", ".join([f"<b>{t}</b> (IPO: {info.get('first_listing_date', 'N/A')} via {info.get('proxy')})" for t, info in backfilled.items()])
    st.markdown(
        f"""
        <div style="background-color: #1a1607; border: 1px solid #78350f; border-left: 4px solid #f59e0b;
                    padding: 8px 12px; border-radius: 4px; margin-bottom: 14px; font-size: 11px; color: #fef3c7;">
            <span style="color: #f59e0b; font-weight: 700; text-transform: uppercase;">ℹ️ History Proxy Backfill:</span>
            The following newer listings have fewer than 10 years of trading history and were backfilled using their regional benchmark proxy to preserve multi-cycle covariance stability:
            {items_desc}.
        </div>
        """,
        unsafe_allow_html=True,
    )

