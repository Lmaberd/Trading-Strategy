"""
EnhancedStrategy: Multi-Sleeve Trading Strategy (GPU-Optimised)
================================================================
Extends BaseStrategy with:
  - Step 1: Universe Selection via Volatility + Amihud Illiquidity composite scoring (monthly)
  - Step 2: FinBERT Sentiment Acceleration (Earnings NLP) — GPU-batched inference
  - Step 3: Dual-sleeve entry triggers (RSI Mean Reversion + Volume Momentum / Sentiment Acceleration)
  - Step 4: Time-based and ATR-based trailing-stop exits

GPU Optimisations applied:
  - Vectorised Pandas groupby for all analytics (no per-ticker Python loop)
  - Batched FinBERT inference via HuggingFace pipeline batch_size (GPU parallelism)
  - Vectorised _build_analytics_lookup (no iterrows)
"""
# Depends on notebook globals: BaseStrategy, TradingSimulation, STARTING_CASH

import re
import numpy as np
import pandas as pd
from collections import defaultdict

from enhanced_helpers.sentiment import get_quarter_key
from enhanced_helpers.lookups import build_analytics_lookup_vectorised


class EnhancedStrategy(BaseStrategy):
    """
    Multi-Sleeve Trading Strategy with Minimalist Drawdown Defense.
    (GPU-Optimised Edition — batched FinBERT + vectorised analytics)
    """

    def __init__(self, finbert_pipeline=None, rsi_threshold=40, vol_mult=2,
                 s1_atr_mult=2, s2_atr_mult=2, sleeve2_exit_weeks=5):
        super().__init__(finbert_pipeline)
        self.rsi_threshold = rsi_threshold
        self.vol_mult = vol_mult
        self.s1_atr_mult = s1_atr_mult
        self.s2_atr_mult = s2_atr_mult
        self.sleeve2_exit_weeks = sleeve2_exit_weeks

        # --- Universe selection state ---
        self.universe = set()
        self.universe_last_updated_month = None
        self.universe_analytics_df = None

        # --- Sleeve tracking (Updated for Peak Price Tracking) ---
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
    # ANALYTICS CALCULATION (Vectorised via Pandas GroupBy — no per-ticker loop)
    # =====================================================================

    def calculate_analytics(self, prices_df):
        print("Computing advanced technicals using fully vectorised Pandas groupby...")
        df = prices_df.copy().sort_values(['ticker', 'date'])

        ticker_index = df['ticker']
        grouped = df.groupby('ticker', sort=False)
        annualisation = np.sqrt(252)

        daily_return = grouped['close'].pct_change()
        close_rolling = df['close'].groupby(ticker_index, sort=False)
        volume_rolling = df['volume'].groupby(ticker_index, sort=False)

        df['daily_return'] = daily_return
        df['ma_200'] = close_rolling.rolling(200, min_periods=50).mean().reset_index(level=0, drop=True)

        daily_return_rolling = daily_return.groupby(ticker_index, sort=False)
        df['vol_60'] = daily_return_rolling.rolling(60, min_periods=30).std().mul(annualisation).reset_index(level=0, drop=True)
        df['vol_10'] = daily_return_rolling.rolling(10, min_periods=5).std().mul(annualisation).reset_index(level=0, drop=True)

        dollar_volume = df['close'] * df['volume']
        amihud_daily = daily_return.abs().div(dollar_volume.replace(0, np.nan))
        df['amihud_60'] = amihud_daily.groupby(ticker_index, sort=False).rolling(60, min_periods=30).mean().reset_index(level=0, drop=True)

        prev_close = grouped['close'].shift()
        if 'high' in df.columns and 'low' in df.columns:
            true_range = pd.concat([
                df['high'] - df['low'],
                (df['high'] - prev_close).abs(),
                (df['low'] - prev_close).abs(),
            ], axis=1).max(axis=1)
        else:
            true_range = (df['close'] - prev_close).abs()
        df['atr_14'] = true_range.groupby(ticker_index, sort=False).rolling(14, min_periods=1).mean().reset_index(level=0, drop=True)

        delta = grouped['close'].diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)
        avg_gain = gain.groupby(ticker_index, sort=False).rolling(14, min_periods=14).mean().reset_index(level=0, drop=True)
        avg_loss = loss.groupby(ticker_index, sort=False).rolling(14, min_periods=14).mean().reset_index(level=0, drop=True)
        rs = avg_gain.div(avg_loss.replace(0, np.nan))
        df['rsi_14'] = 100.0 - (100.0 / (1.0 + rs))

        df['volume_ma_20'] = volume_rolling.rolling(20, min_periods=10).mean().reset_index(level=0, drop=True)

        result_df = df[[
            'ticker', 'date', 'open', 'close', 'volume', 'daily_return',
            'vol_60', 'vol_10', 'amihud_60', 'rsi_14', 'volume_ma_20',
            'ma_200', 'atr_14'
        ]].copy()
        result_df['date'] = result_df['date'].dt.strftime('%Y-%m-%d')

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

        # Expanded to Top 40%
        threshold = scored['composite_score'].quantile(0.80)
        self.universe = set(scored[scored['composite_score'] >= threshold]['ticker'].tolist())

        month_key = current_date[:7]
        self.universe_last_updated_month = month_key
        print(f"  Universe updated ({month_key}): {len(self.universe)} stocks selected.")

    # =====================================================================
    # LLM / FINBERT ANALYSIS — GPU-BATCHED INFERENCE
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
            # --- Chunking: split into sentences ---
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

            # --- Filter valid chunks ---
            valid_chunks = [c[:512] for c in chunks if c and len(c) >= 10]
            if not valid_chunks:
                return None

            # --- GPU-BATCHED INFERENCE ---
            # Instead of processing one chunk at a time, batch them all
            # through the pipeline. The HuggingFace pipeline with batch_size
            # processes multiple inputs in parallel on the GPU.
            pos, neg, neu = 0, 0, 0
            try:
                batch_results = self.finbert_pipeline(
                    valid_chunks,
                    batch_size=64,       # Process 64 chunks at once on GPU
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
                # Fallback: if batching fails, count all as neutral
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
    # DECISION LOGIC (Minimalist Defense Integrated)
    # =====================================================================

    def make_decision(self, ticker, date, transcript, portfolio_state, analytics):
        price = analytics.get('close', 0)
        if price <= 0:
            return 'HOLD'
        has_pos = ticker in portfolio_state.get('positions', {})

        if self.universe_last_updated_month != date[:7]:
            self._update_universe(date)

        atr_14 = analytics.get('atr_14', price * 0.05)

        # --- EXIT LOGIC ---
        if ticker in self.sleeve1_positions:
            pos_info = self.sleeve1_positions[ticker]
            pos_info['peak_price'] = max(pos_info['peak_price'], price)

            # Sleeve 1 Exit: ATR Trailing Stop OR time-based
            if price <= pos_info['peak_price'] - (self.s1_atr_mult * atr_14) or date > pos_info['entry_date']:
                del self.sleeve1_positions[ticker]
                return 'SELL'

        if ticker in self.sleeve2_positions:
            pos_info = self.sleeve2_positions[ticker]
            pos_info['weeks_held'] += 1
            pos_info['peak_price'] = max(pos_info['peak_price'], price)

            # Sleeve 2 Exit: ATR Trailing Stop OR time-based
            if price <= pos_info['peak_price'] - (self.s2_atr_mult * atr_14) or pos_info['weeks_held'] >= self.sleeve2_exit_weeks:
                del self.sleeve2_positions[ticker]
                return 'SELL'
            return 'HOLD'

        in_univ = ticker in self.universe
        if has_pos and not in_univ and ticker not in self.sleeve1_positions and ticker not in self.sleeve2_positions:
            return 'SELL'
        if not in_univ or has_pos:
            return 'HOLD'

        # --- DEFENSIVE VETO LOGIC ---
        vol_10, vol_60 = analytics.get('vol_10'), analytics.get('vol_60')
        ma_200 = analytics.get('ma_200')

        # H3: Volatility Acceleration Veto
        vol_veto = (vol_10 is not None and vol_60 is not None
                    and not np.isnan(vol_10) and not np.isnan(vol_60)
                    and vol_10 > 1.5 * vol_60)

        # H7: Trend Filter Veto
        is_downtrend = (ma_200 is not None and price < ma_200)

        # --- ENTRY LOGIC ---
        rsi = analytics.get('rsi_14')
        daily_ret = analytics.get('daily_return')
        vol = analytics.get('volume')
        vol_ma = analytics.get('volume_ma_20')

        s1_trigger = False

        # Trigger A: Mean Reversion
        if rsi is not None and not np.isnan(rsi) and rsi < self.rsi_threshold and not vol_veto:
            s1_trigger = True

        # Trigger B: Momentum
        if not s1_trigger and not is_downtrend and not vol_veto:
            if all(v is not None and not np.isnan(v) for v in (daily_ret, vol, vol_ma)):
                if daily_ret > 0 and vol_ma > 0 and vol > self.vol_mult * vol_ma:
                    s1_trigger = True

        if s1_trigger:
            self.sleeve1_positions[ticker] = {'entry_date': date, 'peak_price': price}
            return 'BUY'

        # Sleeve 2: Earnings Momentum
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
        # Assuming TradingSimulation and STARTING_CASH are imported in your master script
        sim = TradingSimulation(self.prices, self.earnings, STARTING_CASH)
        return sim.run(lambda t, d, tr, ps, a: self.make_decision(t, d, tr, ps, a), analytics_lookup, verbose)
