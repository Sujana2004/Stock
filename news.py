"""Headlines and earnings dates from Yahoo Finance (via yfinance).

The forecast is built from prices only, so it cannot know about news. This module
gives the user the context the model lacks. It is deliberately isolated from the
forecasting code: every fetch function returns an empty result on any failure, so
a news outage can never break the forecast.

Parsing is kept in pure functions (:func:`parse_headlines`, :func:`parse_next_earnings`)
so it can be tested without network access.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime, timezone

import yfinance as yf

from stock_utils import normalize_ticker

logger = logging.getLogger(__name__)

MAX_HEADLINES = 10


@dataclass(frozen=True)
class Headline:
    title: str
    publisher: str
    published: datetime | None    # timezone-aware UTC, or None if Yahoo gave no time
    url: str


def _parse_time(value) -> datetime | None:
    """Yahoo has used both ISO strings ('2026-11-14T11:00:00Z') and Unix seconds."""
    try:
        if isinstance(value, (int, float)):
            return datetime.fromtimestamp(value, tz=timezone.utc)
        if isinstance(value, str) and value:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    except (ValueError, OverflowError, OSError):
        pass
    return None


def _url_of(content: dict) -> str:
    for key in ("canonicalUrl", "clickThroughUrl"):
        link = content.get(key)
        if isinstance(link, dict) and link.get("url"):
            return str(link["url"])
    return str(content.get("link") or "")


def parse_headlines(raw, limit: int = MAX_HEADLINES) -> list[Headline]:
    """Convert yfinance's ``Ticker.news`` payload into headlines, newest first.

    Handles the current nested format (``item["content"]``) and the older flat one.
    Items without a title or without an http(s) link are dropped.
    """
    headlines = []
    for item in raw or []:
        if not isinstance(item, dict):
            continue
        content = item.get("content") if isinstance(item.get("content"), dict) else item
        title = str(content.get("title") or "").strip()
        url = _url_of(content)
        if not title or not url.startswith(("https://", "http://")):
            continue
        provider = content.get("provider")
        publisher = (provider.get("displayName") if isinstance(provider, dict)
                     else content.get("publisher")) or "Unknown source"
        published = _parse_time(content.get("pubDate") or content.get("displayTime")
                                or content.get("providerPublishTime"))
        headlines.append(Headline(title=title, publisher=str(publisher), published=published, url=url))
    oldest = datetime.min.replace(tzinfo=timezone.utc)
    headlines.sort(key=lambda h: h.published or oldest, reverse=True)
    return headlines[:limit]


def parse_next_earnings(calendar, today: date) -> date | None:
    """Earliest earnings date on or after ``today`` in yfinance's ``Ticker.calendar``."""
    if not isinstance(calendar, dict):
        return None
    value = calendar.get("Earnings Date")
    candidates = value if isinstance(value, (list, tuple)) else [value]
    upcoming = []
    for candidate in candidates:
        if isinstance(candidate, datetime):
            candidate = candidate.date()
        if isinstance(candidate, date) and candidate >= today:
            upcoming.append(candidate)
    return min(upcoming) if upcoming else None


def fetch_headlines(ticker: str, limit: int = MAX_HEADLINES) -> list[Headline]:
    """Recent headlines for ``ticker``; an empty list if Yahoo has none or the call fails."""
    try:
        return parse_headlines(yf.Ticker(normalize_ticker(ticker)).news, limit)
    except Exception as exc:  # optional feature: never let it break the app
        logger.warning("Headlines unavailable for %s: %s", ticker, exc)
        return []


def fetch_next_earnings(ticker: str, today: date | None = None) -> date | None:
    """Next earnings date for ``ticker``; ``None`` if unknown, past, or the call fails.

    Indices and many non-US symbols have no earnings calendar on Yahoo.
    """
    try:
        calendar = yf.Ticker(normalize_ticker(ticker)).calendar
        return parse_next_earnings(calendar, today or date.today())
    except Exception as exc:  # optional feature: never let it break the app
        logger.warning("Earnings date unavailable for %s: %s", ticker, exc)
        return None
