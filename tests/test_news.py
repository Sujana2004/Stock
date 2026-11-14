"""Offline tests for news.py, using payloads in the shapes Yahoo has returned."""
import sys
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import news  # noqa: E402
import stock_utils as su  # noqa: E402

NESTED = {"id": "1", "content": {
    "title": "Chipmaker falls after CEO resigns", "pubDate": "2026-11-13T11:00:00Z",
    "provider": {"displayName": "Yahoo Finance"},
    "canonicalUrl": {"url": "https://finance.yahoo.com/a"}}}
FLAT_OLD = {"title": "Older format item", "publisher": "Reuters", "link": "https://example.com/b",
            "providerPublishTime": 1794400000}


def test_parse_headlines_handles_both_formats_and_sorts_newest_first():
    out = news.parse_headlines([FLAT_OLD, NESTED])
    assert [h.title for h in out] == ["Chipmaker falls after CEO resigns", "Older format item"]
    assert out[0].publisher == "Yahoo Finance" and out[0].url == "https://finance.yahoo.com/a"
    assert out[0].published == datetime(2026, 11, 13, 11, tzinfo=timezone.utc)
    assert out[1].publisher == "Reuters" and out[1].published.tzinfo is not None


def test_parse_headlines_drops_unusable_items():
    bad = [
        None, "text", {},
        {"content": {"title": "", "canonicalUrl": {"url": "https://x.com"}}},            # no title
        {"content": {"title": "No link"}},                                               # no url
        {"content": {"title": "Script", "canonicalUrl": {"url": "javascript:alert(1)"}}},
    ]
    assert news.parse_headlines(bad) == []
    assert news.parse_headlines(None) == []


def test_parse_headlines_tolerates_missing_fields_and_respects_limit():
    item = {"content": {"title": "T", "pubDate": "not a date", "clickThroughUrl": {"url": "https://x.com"}}}
    out = news.parse_headlines([item] * 5, limit=3)
    assert len(out) == 3
    assert out[0].published is None and out[0].publisher == "Unknown source"


@pytest.mark.parametrize("calendar, expected", [
    ({"Earnings Date": [date(2026, 11, 17)]}, date(2026, 11, 17)),
    ({"Earnings Date": [date(2026, 10, 8)]}, None),                                   # already past
    ({"Earnings Date": [date(2026, 10, 8), date(2027, 1, 9), date(2026, 12, 1)]}, date(2026, 12, 1)),
    ({"Earnings Date": datetime(2026, 11, 14, 21, 0)}, date(2026, 11, 14)),           # today counts
    ({"Earnings Date": []}, None),
    ({}, None),
    (None, None),
])
def test_parse_next_earnings(calendar, expected):
    assert news.parse_next_earnings(calendar, today=date(2026, 11, 14)) == expected


def test_fetch_functions_return_empty_on_failure(monkeypatch):
    def broken(_):
        raise ConnectionError("offline")

    monkeypatch.setattr(news.yf, "Ticker", broken)
    assert news.fetch_headlines("NVDA") == []
    assert news.fetch_next_earnings("NVDA") is None
    assert news.fetch_headlines("not a ticker!") == []       # invalid symbol is not an error here


# ---------------------------------------------------------------- recent_shock
def _arrays(returns):
    close = 100 * np.exp(np.cumsum(returns))
    frame = pd.DataFrame({"Open": close, "High": close, "Low": close, "Close": close, "Volume": 1},
                         index=pd.bdate_range("2022-01-03", periods=len(close)))
    return su.prepare(frame)


def test_recent_shock_ignores_ordinary_days():
    rng = np.random.default_rng(1)
    returns = rng.normal(0, 0.01, 600)
    returns[-10:] = 0.004                       # calm final stretch
    assert su.recent_shock(_arrays(returns)) is None


def test_recent_shock_flags_a_single_crash_day():
    returns = np.random.default_rng(1).normal(0, 0.01, 600)
    returns[-10:] = 0.0
    returns[-3] = np.log(0.90)                  # -10% in one day for a 1%-a-day stock
    shock = su.recent_shock(_arrays(returns))
    assert shock["largest_daily_move"] == pytest.approx(-0.10)
    assert shock["total_move"] == pytest.approx(-0.10)
    assert shock["largest_move_date"] == pd.bdate_range("2022-01-03", periods=600)[-3]


def test_recent_shock_flags_a_steady_slide():
    returns = np.random.default_rng(1).normal(0, 0.01, 600)
    returns[-10:] = -0.012                      # no single big day, but -11% in total
    shock = su.recent_shock(_arrays(returns))
    assert shock is not None and shock["total_move"] < -0.10


def test_recent_shock_needs_enough_data():
    assert su.recent_shock(_arrays(np.zeros(5))) is None
