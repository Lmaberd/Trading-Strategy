import pandas as pd
from collections import defaultdict


def get_quarter_key(date_str):
    """Convert a date string to a quarter key like '2015-Q3'."""
    dt = pd.to_datetime(date_str)
    return f"{dt.year}-Q{(dt.month - 1) // 3 + 1}"


def build_sentiment_lookup(sentiment_cache):
    """
    Organise cached sentiments by ticker for fast chronological search.

    Args:
        sentiment_cache: dict keyed by (ticker, date_str) -> float score

    Returns:
        sentiment_by_ticker: defaultdict(list) of {ticker: [(date_str, score), ...]} sorted by date
    """
    sentiment_by_ticker = defaultdict(list)
    for (ticker, ds), score in sentiment_cache.items():
        sentiment_by_ticker[ticker].append((ds, score))
    for t in sentiment_by_ticker:
        sentiment_by_ticker[t].sort()
    return sentiment_by_ticker


def get_latest_sentiment(sentiment_by_ticker, leader, date_str):
    """Most recent sentiment score for a leader on or before date_str."""
    best = None
    for ds, sc in sentiment_by_ticker.get(leader, []):
        if ds <= date_str:
            best = sc
        else:
            break
    return best
