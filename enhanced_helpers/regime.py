def compute_market_breadth(analytics_df):
    """
    Compute daily market breadth: % of stocks with Close > 200-SMA.
    Bull regime: breadth > 50%. Bear regime: breadth <= 50%.

    Returns:
        market_breadth: dict of {date_str: breadth_pct}
    """
    breadth_df = analytics_df.dropna(subset=['sma_200']).groupby('date').agg(
        above_count=('above_sma_200', 'sum'),
        total_count=('above_sma_200', 'count')
    ).reset_index()
    breadth_df['breadth_pct'] = breadth_df['above_count'] / breadth_df['total_count']

    market_breadth = dict(zip(breadth_df['date'], breadth_df['breadth_pct']))

    bull_days = sum(1 for v in market_breadth.values() if v > 0.50)
    total_days = len(market_breadth)
    print(f"Market breadth computed: {total_days} days, {bull_days} bull ({bull_days/max(total_days,1)*100:.1f}%)")

    return market_breadth


def get_regime(date, market_breadth):
    """
    Determine market regime from breadth data.
    Uses the most recent available breadth reading on or before the given date.

    Returns:
        True for Bull (breadth > 50%), False for Bear.
    """
    breadth = market_breadth.get(date)
    if breadth is not None:
        return breadth > 0.50

    # Fallback: find most recent breadth reading before this date
    sorted_dates = sorted(market_breadth.keys())
    for d in reversed(sorted_dates):
        if d <= date:
            return market_breadth[d] > 0.50

    return True  # default to bull if no data yet
