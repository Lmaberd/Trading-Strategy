"""
EnhancedStrategy: Multi-Sleeve Trading Strategy (Polars Edition)
================================================================
Same logic as V7 (GPU-Optimised) with one key upgrade:
  - calculate_analytics() rewritten in Polars for multi-threaded,
    zero-copy analytics. Replaces pandas groupby+rolling+reset_index
    chains with Polars window expressions (.over('ticker')).

Why faster:
  - Polars runs all rolling calculations in parallel across CPU cores
  - No reset_index() overhead — .over() aligns results automatically
  - Lazy expression engine avoids intermediate allocations
  - Written in Rust under the hood

Everything else (FinBERT batching, make_decision, universe selection)
is identical to V7.
"""
# Depends on notebook globals: BaseStrategy, TradingSimulation, STARTING_CASH

import re
import numpy as np
import pandas as pd
import polars as pl
from collections import defaultdict

from enhanced_helpers.sentiment import get_quarter_key
from enhanced_helpers.lookups import build_analytics_lookup_vectorised


class EnhancedStrategy(BaseStrategy):
    """
    Multi-Sleeve Trading Strategy — Polars Analytics Edition.
    """

    def __init__(self, finbert_pipeline=None, rsi_threshold=40, vol_mult=2,
                 s1_atr_mult=2, s2_atr_mult=2, sleeve2_exit_weeks=5):
        super().__init__(finbert_pipeline)
        self.rsi_threshold = rsi_threshold
        self.vol_mult = vol_mult
        self.s1_atr_mult = s1_atr_mult
        self.s2_atr_mult = s2_atr_mult
        self.sleeve2_exit_weeks = sleeve2_exit_weeks

        self.universe = set()
        self.universe_last_updated_month = None
        self.universe_analytics_df = None

        self.sleeve1_positions = {}  # {ticker: {'entry_date': str, 'peak_price': float}}
        self.sleeve2_positions = {}  # {ticker: {'entry_date': str, 'weeks_held': int, 'peak_price': float}}
        self.sentiment_cache = {}

    # =====================================================================
    # DATA PREPARATION
    # =====================================================================

    def clean_data(self, prices_df, earnings_df):
        prices = prices_df.copy().drop_duplicates()
        earnings = earnings_df.copy().drop_duplicates()
        prices['date'] = pd.to_datetime(prices['date'])
        earnings['date'] = pd.to_datetime(earnings['date'])
        earnings = earnings[earnings['transcript'].str.len() > 100]
        prices = prices.sort_values(['ticker', 'date']).reset_index(drop=True)
        return prices, earnings

    # =====================================================================
    # ANALYTICS CALCULATION — Polars multi-threaded window functions
    # =====================================================================

    def calculate_analytics(self, prices_df):
        print("Computing advanced technicals using Polars (multi-threaded)...")

        has_hl = 'high' in prices_df.columns and 'low' in prices_df.columns
        cols = ['ticker', 'date', 'open', 'close', 'volume']
        if has_hl:
            cols += ['high', 'low']

        # Convert to Polars and sort
        df = pl.from_pandas(prices_df[cols].copy()).sort(['ticker', 'date'])

        annualisation = float(np.sqrt(252))

        # --- Daily return and MA200 ---
        df = df.with_columns([
            pl.col('close').pct_change().over('ticker').alias('daily_return'),
            pl.col('close').rolling_mean(window_size=200, min_periods=50).over('ticker').alias('ma_200'),
        ])

        # --- Volatility (annualised) ---
        df = df.with_columns([
            (pl.col('daily_return').rolling_std(window_size=60, min_periods=30).over('ticker') * annualisation).alias('vol_60'),
            (pl.col('daily_return').rolling_std(window_size=10, min_periods=5).over('ticker') * annualisation).alias('vol_10'),
        ])

        # --- Amihud illiquidity ---
        df = df.with_columns([
            pl.when(pl.col('close') * pl.col('volume') == 0)
              .then(None)
              .otherwise(pl.col('daily_return').abs() / (pl.col('close') * pl.col('volume')))
              .alias('_amihud_daily')
        ])
        df = df.with_columns([
            pl.col('_amihud_daily').rolling_mean(window_size=60, min_periods=30).over('ticker').alias('amihud_60'),
        ])

        # --- ATR (True Range) ---
        prev_close = pl.col('close').shift(1).over('ticker')
        if has_hl:
            df = df.with_columns([
                pl.max_horizontal(
                    pl.col('high') - pl.col('low'),
                    (pl.col('high') - prev_close).abs(),
                    (pl.col('low') - prev_close).abs(),
                ).alias('_true_range')
            ])
        else:
            df = df.with_columns([
                (pl.col('close') - prev_close).abs().alias('_true_range')
            ])
        df = df.with_columns([
            pl.col('_true_range').rolling_mean(window_size=14, min_periods=1).over('ticker').alias('atr_14'),
        ])

        # --- RSI (14) ---
        df = df.with_columns([
            (pl.col('close') - pl.col('close').shift(1).over('ticker')).alias('_delta'),
        ])
        df = df.with_columns([
            pl.when(pl.col('_delta') > 0).then(pl.col('_delta')).otherwise(0.0).alias('_gain'),
            pl.when(pl.col('_delta') < 0).then(-pl.col('_delta')).otherwise(0.0).alias('_loss'),
        ])
        df = df.with_columns([
            pl.col('_gain').rolling_mean(window_size=14, min_periods=14).over('ticker').alias('_avg_gain'),
            pl.col('_loss').rolling_mean(window_size=14, min_periods=14).over('ticker').alias('_avg_loss'),
        ])
        df = df.with_columns([
            pl.when(pl.col('_avg_loss') == 0)
              .then(100.0)
              .otherwise(100.0 - (100.0 / (1.0 + pl.col('_avg_gain') / pl.col('_avg_loss'))))
              .alias('rsi_14'),
        ])

        # --- Volume MA (20) ---
        df = df.with_columns([
            pl.col('volume').cast(pl.Float64).rolling_mean(window_size=20, min_periods=10).over('ticker').alias('volume_ma_20'),
        ])

        # --- Convert back to pandas ---
        result_df = df.select([
            'ticker', 'date', 'open', 'close', 'volume', 'daily_return',
            'vol_60', 'vol_10', 'amihud_60', 'rsi_14', 'volume_ma_20', 'ma_200', 'atr_14'
        ]).to_pandas()

        result_df['date'] = pd.to_datetime(result_df['date']).dt.strftime('%Y-%m-%d')

        self.universe_analytics_df = result_df
        print(f"  Analytics computed: {len(result_df):,} rows for {result_df['ticker'].nunique()} tickers")
        return result_df

    # =====================================================================
    # UNIVERSE SELECTION
    # =====================================================================

    def _update_universe(self, current_date):
        if self.universe_analytics_df is None:
            return

        df = self.universe_analytics_df[self.universe_analytics_df['date'] <= current_date]
        if df.empty:
            return

        latest = df.groupby('ticker').last().reset_index()
        scored = latest.dropna(subset=['vol_60', 'amihud_60']).copy()
        if scored.empty:
            return

        scored['vol_rank'] = scored['vol_60'].rank(ascending=True, method='average')
        scored['amihud_rank'] = scored['amihud_60'].rank(ascending=False, method='average')
        scored['composite_score'] = scored['vol_rank'] + scored['amihud_rank']

        threshold = scored['composite_score'].quantile(0.80)
        self.universe = set(scored[scored['composite_score'] >= threshold]['ticker'].tolist())

        month_key = current_date[:7]
        self.universe_last_updated_month = month_key
        print(f"  Universe updated ({month_key}): {len(self.universe)} stocks selected.")

    # =====================================================================
    # LLM / FINBERT ANALYSIS — GPU-BATCHED INFERENCE (unchanged from V7)
    # =====================================================================

    def llm_analysis(self, ticker, transcript, date):
        if transcript is None or self.finbert_pipeline is None:
            return None
        cache_key = f"{ticker}_{date}"
        if cache_key in self.llm_cache:
            self.llm_cache_hits += 1
            return self.llm_cache[cache_key]

        self.llm_cache_misses += 1
        try:
            sentences = re.split(r'(?<=[.!?])\s+', transcript.strip())
            chunks, current_chunk = [], ""
            for sentence in sentences:
                if len(current_chunk) + len(sentence) > 400:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = sentence
                else:
                    current_chunk += " " + sentence
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
            if not chunks:
                return None

            valid_chunks = [c[:512] for c in chunks if c and len(c) >= 10]
            if not valid_chunks:
                return None

            pos, neg, neu = 0, 0, 0
            try:
                batch_results = self.finbert_pipeline(
                    valid_chunks,
                    batch_size=64,
                    truncation=True,
                    max_length=512
                )
                for result in batch_results:
                    label = result['label'].lower() if isinstance(result, dict) else 'neutral'
                    if 'positive' in label:
                        pos += 1
                    elif 'negative' in label:
                        neg += 1
                    else:
                        neu += 1
            except Exception:
                neu = len(valid_chunks)

            total = pos + neg + neu
            if total == 0:
                return None
            net_sentiment = (pos - neg) / total
            dominant = 'positive' if pos >= neg and pos >= neu else (
                'negative' if neg >= pos and neg >= neu else 'neutral')

            quarter_key = get_quarter_key(date)
            self.sentiment_cache[(ticker, quarter_key)] = net_sentiment
            dt = pd.to_datetime(date)
            prev_dt = dt - pd.DateOffset(months=3)
            prev_key = f"{prev_dt.year}-Q{(prev_dt.month - 1) // 3 + 1}"
            prev_sentiment = self.sentiment_cache.get((ticker, prev_key))
            accel = (net_sentiment > prev_sentiment) if prev_sentiment is not None else False

            result = {
                'sentiment': dominant, 'net_sentiment': net_sentiment,
                'sentiment_acceleration': accel,
                'positive_ratio': pos / total if total > 0 else 0,
                'chunks_processed': total
            }
            self.llm_cache[cache_key] = result
            return result
        except Exception:
            return None

    # =====================================================================
    # DECISION LOGIC (unchanged from V7)
    # =====================================================================

    def make_decision(self, ticker, date, transcript, portfolio_state, analytics):
        price = analytics.get('close', 0)
        if price <= 0:
            return 'HOLD'
        has_pos = ticker in portfolio_state.get('positions', {})

        if self.universe_last_updated_month != date[:7]:
            self._update_universe(date)

        atr_14 = analytics.get('atr_14', price * 0.05)

        if ticker in self.sleeve1_positions:
            pos_info = self.sleeve1_positions[ticker]
            pos_info['peak_price'] = max(pos_info['peak_price'], price)
            if price <= pos_info['peak_price'] - (self.s1_atr_mult * atr_14) or date > pos_info['entry_date']:
                del self.sleeve1_positions[ticker]
                return 'SELL'

        if ticker in self.sleeve2_positions:
            pos_info = self.sleeve2_positions[ticker]
            pos_info['weeks_held'] += 1
            pos_info['peak_price'] = max(pos_info['peak_price'], price)
            if price <= pos_info['peak_price'] - (self.s2_atr_mult * atr_14) or pos_info['weeks_held'] >= self.sleeve2_exit_weeks:
                del self.sleeve2_positions[ticker]
                return 'SELL'
            return 'HOLD'

        in_univ = ticker in self.universe
        if has_pos and not in_univ and ticker not in self.sleeve1_positions and ticker not in self.sleeve2_positions:
            return 'SELL'
        if not in_univ or has_pos:
            return 'HOLD'

        vol_10, vol_60 = analytics.get('vol_10'), analytics.get('vol_60')
        ma_200 = analytics.get('ma_200')

        vol_veto = (vol_10 is not None and vol_60 is not None
                    and not np.isnan(vol_10) and not np.isnan(vol_60)
                    and vol_10 > 1.5 * vol_60)
        is_downtrend = (ma_200 is not None and price < ma_200)

        rsi = analytics.get('rsi_14')
        daily_ret = analytics.get('daily_return')
        vol = analytics.get('volume')
        vol_ma = analytics.get('volume_ma_20')

        s1_trigger = False

        if rsi is not None and not np.isnan(rsi) and rsi < self.rsi_threshold and not vol_veto:
            s1_trigger = True

        if not s1_trigger and not is_downtrend and not vol_veto:
            if all(v is not None and not np.isnan(v) for v in (daily_ret, vol, vol_ma)):
                if daily_ret > 0 and vol_ma > 0 and vol > self.vol_mult * vol_ma:
                    s1_trigger = True

        if s1_trigger:
            self.sleeve1_positions[ticker] = {'entry_date': date, 'peak_price': price}
            return 'BUY'

        if transcript is not None and not is_downtrend and not vol_veto:
            sent_res = self.llm_analysis(ticker, transcript, date)
            if sent_res is not None and sent_res.get('sentiment_acceleration', False):
                self.sleeve2_positions[ticker] = {'entry_date': date, 'weeks_held': 0, 'peak_price': price}
                return 'BUY'

        return 'HOLD'

    # =====================================================================
    # EVALUATION ORCHESTRATION
    # =====================================================================

    def evaluate(self, verbose=False):
        if self.prices is None or self.earnings is None:
            raise ValueError("Must call set_data() before evaluate()")
        self.sleeve1_positions, self.sleeve2_positions, self.universe = {}, {}, set()
        self.universe_last_updated_month, self.universe_analytics_df = None, None

        print("Running evaluation...")
        analytics = self.calculate_analytics(self.prices)
        analytics_lookup = build_analytics_lookup_vectorised(analytics)

        print("Running backtest simulation...")
        sim = TradingSimulation(self.prices, self.earnings, STARTING_CASH)
        return sim.run(lambda t, d, tr, ps, a: self.make_decision(t, d, tr, ps, a), analytics_lookup, verbose)
