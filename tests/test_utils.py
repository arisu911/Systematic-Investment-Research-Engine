"""Unit tests for UI utilities and adaptive currency formatter."""

import numpy as np
import pytest
from research.utils import format_money


def test_format_money_usd():
    assert format_money(1000.0, "USD") == "$1,000.00"
    assert format_money(1250000.50, "USD", compact=True) == "$1.25M"
    assert format_money(2500000000.0, "USD", compact=True) == "$2.50B"
    assert format_money(4500.0, "USD", compact=True) == "$4.5K"


def test_format_money_myr():
    assert format_money(500.25, "MYR") == "RM 500.25"
    assert format_money(100000.0, "MYR") == "RM 100,000.00"
    assert format_money(5000000.0, "MYR", compact=True) == "RM 5.00M"


def test_format_money_jpy():
    # JPY has no fractional cents
    assert format_money(15000.0, "JPY") == "¥15,000"
    assert format_money(250000.0, "JPY", compact=True) == "¥25.0万"
    assert format_money(500000000.0, "JPY", compact=True) == "¥5.00B"


def test_format_money_negative():
    assert format_money(-3420.50, "USD") == "-$3,420.50"
    assert format_money(-150000.0, "MYR") == "-RM 150,000.00"
    assert format_money(-25000.0, "JPY") == "-¥25,000"


def test_format_money_nan_and_none():
    assert format_money(None, "USD") == "N/A"
    assert format_money(np.nan, "USD") == "N/A"
