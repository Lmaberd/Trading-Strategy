"""
EnhancedStrategy: Multi-Sleeve Trading Strategy
=================================================
Extends BaseStrategy with:
  - Step 1: Universe Selection via Volatility + Amihud Illiquidity composite scoring (monthly)
  - Step 2: FinBERT Sentiment Acceleration (Earnings NLP)
  - Step 3: Dual-sleeve entry triggers (RSI Mean Reversion + Volume Momentum / Sentiment Acceleration)
  - Step 4: Time-based and ATR-based trailing-stop exits
  - Step 5: Dynamic capital allocation with Leverage Effect sizing
  - Market Breadth Regime Filter (200-SMA based)
"""
# Depends on notebook globals: TradingSimulation, STARTING_CASH
# Note: this version does NOT extend BaseStrategy (standalone class)

import re
import numpy as np
import pandas as pd
from collections import defaultdict

from enhanced_helpers.sentiment import get_quarter_key
from enhanced_helpers.lookups import build_analytics_lookup
from enhanced_helpers.regime import compute_market_breadth, get_regime


class EnhancedStrategy:
    """
    Multi-Sleeve Trading Strategy.

    Sleeve 1 (Tactical Weekly): RSI mean reversion + Volume-confirmed momentum.
        - Entry: Friday close — RSI<30 OR (positive return AND volume > 1.5x 20d MA)
        - Exit: Sell the following week (approximating Monday Close exit)
        - Sizing: 60% of capital, further attenuated 0.6x on red Fridays (leverage effect)

    Sleeve 2 (Fundamental Momentum): FinBERT Earnings Sentiment Acceleration.
        - Entry: When sentiment acceleration flag is True (Q0 net sentiment > Q-1)
        - Exit: 20-30 trading days (~4-6 weekly evaluations) OR 3×ATR(14) trailing stop below highest high
        - Sizing: 40% of capital, equally weighted across triggered stocks

    Market Breadth Regime Filter:
        - 200-day SMA breadth: % of stocks with Close > 200-SMA
        - Bull (>50%): Full allocation for both sleeves
        - Bear (<50%): Exit all positions, no new entries
    """

    def __init__(self, finbert_pipeline=None):
        self.finbert_pipeline = finbert_pipeline
        self.llm_cache = {}
        self.llm_cache_hits = 0
        self.llm_cache_misses = 0
        self.prices = None
        self.earnings = None

        # --- Universe selection state ---
        self.universe = set()
        self.universe_last_updated_month = None
        self.universe_analytics_df = None  # store full analytics for universe updates

        # --- Sleeve 1: Tactical Weekly tracking ---
        self.sleeve1_positions = {}  # {ticker: entry_week_date}

        # --- Sleeve 2: Fundamental Momentum tracking ---
        # {ticker: {'entry_date': str, 'weeks_held': int, 'highest_high': float, 'current_atr': float}}
        self.sleeve2_positions = {}

        # --- Sentiment cache for acceleration detection ---
        self.sentiment_cache = {}  # {(ticker, quarter_key): net_sentiment}

        # --- Capital allocation ---
        self.sleeve1_allocation = 0.60  # 60% to Sleeve 1
        self.sleeve2_allocation = 0.40  # 40% to Sleeve 2

        # --- Market Breadth Regime ---
        self.market_breadth = {}  # {date_str: breadth_pct}
        self.is_bull_regime = True  # default to bull until calculated

    # =====================================================================
    # DATA PREPARATION
    # =====================================================================

    def set_data(self, prices_df, earnings_df):
        """Set and preprocess data for evaluation. Call this before evaluate()."""
        print("Cleaning and preprocessing data...")
        self.prices, self.earnings = self.clean_data(prices_df, earnings_df)
        print(f"Data ready: {len(self.prices):,} price records")

    def clean_data(self, prices_df, earnings_df):
        """Clean and validate data with enhanced preprocessing."""
        prices = prices_df.copy().drop_duplicates()
        earnings = earnings_df.copy().drop_duplicates()
        prices['date'] = pd.to_datetime(prices['date'])
        earnings['date'] = pd.to_datetime(earnings['date'])
        earnings = earnings[earnings['transcript'].str.len() > 100]

        # Sort prices for proper rolling calculations
        prices = prices.sort_values(['ticker', 'date']).reset_index(drop=True)

        return prices, earnings

    # =====================================================================
    # ANALYTICS CALCULATION (Step 1 metrics + technical indicators)
    # =====================================================================

    def calculate_analytics(self, prices_df):
        """
        Calculate all technical indicators needed for the strategy:
          - daily_return, vol_60 (annualized), amihud_60
          - rsi_14, volume_ma_20, ma_50, atr_14
          - sma_200 (for market breadth regime filter)
          - open, high, close, volume (passed through for decision logic)
        """
        print("Computing enhanced technical indicators...")
        results = []

        for ticker in prices_df['ticker'].unique():
            df = prices_df[prices_df['ticker'] == ticker].copy().sort_values('date')

            # --- Basic returns ---
            df['daily_return'] = df['close'].pct_change()
            df['abs_return'] = df['daily_return'].abs()

            # --- Step 1 metrics: Universe Selection ---
            # Volatility: annualized 60-day rolling std of daily returns
            df['vol_60'] = df['daily_return'].rolling(60, min_periods=30).std() * np.sqrt(252)

            # Amihud Illiquidity: avg(|return| / dollar_volume) over 60 days
            df['dollar_volume'] = df['close'] * df['volume']
            # Avoid division by zero: replace 0 dollar_volume with NaN
            df['amihud_daily'] = df['abs_return'] / df['dollar_volume'].replace(0, np.nan)
            df['amihud_60'] = df['amihud_daily'].rolling(60, min_periods=30).mean()

            # --- Step 3 metrics: Entry Triggers ---
            # RSI-14
            delta = df['close'].diff()
            gain = delta.where(delta > 0, 0.0)
            loss = (-delta).where(delta < 0, 0.0)
            avg_gain = gain.rolling(14, min_periods=14).mean()
            avg_loss = loss.rolling(14, min_periods=14).mean()
            rs = avg_gain / avg_loss.replace(0, np.nan)
            df['rsi_14'] = 100.0 - (100.0 / (1.0 + rs))

            # Volume 20-day MA
            df['volume_ma_20'] = df['volume'].rolling(20, min_periods=10).mean()

            # MA-50 (from base strategy)
            df['ma_50'] = df['close'].rolling(50, min_periods=1).mean()

            # --- ATR-14 (for Sleeve 2 trailing stop) ---
            high = df['high'] if 'high' in df.columns else df['close']
            low = df['low'] if 'low' in df.columns else df['close']
            tr1 = high - low
            tr2 = (high - df['close'].shift(1)).abs()
            tr3 = (low - df['close'].shift(1)).abs()
            true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            df['atr_14'] = true_range.rolling(14, min_periods=14).mean()

            # --- 200-day SMA (for market breadth regime) ---
            df['sma_200'] = df['close'].rolling(200, min_periods=100).mean()
            df['above_sma_200'] = (df['close'] > df['sma_200']).astype(int)

            results.append(df[['ticker', 'date', 'open', 'close', 'volume',
                               'daily_return', 'vol_60', 'amihud_60',
                               'rsi_14', 'volume_ma_20', 'ma_50',
                               'atr_14', 'sma_200', 'above_sma_200']])

        result_df = pd.concat(results, ignore_index=True)
        result_df['date'] = result_df['date'].dt.strftime('%Y-%m-%d')
        print(f"Enhanced indicators computed: {len(result_df):,} rows")

        # Store for universe updates
        self.universe_analytics_df = result_df

        # --- Pre-compute market breadth for regime filter ---
        self.market_breadth = compute_market_breadth(result_df)

        return result_df

    # =====================================================================
    # UNIVERSE SELECTION (Step 1: Monthly filter — top 20% composite score)
    # =====================================================================

    def _update_universe(self, current_date):
        """
        Recalculate the tradable universe monthly.
        Composite Score = Volatility Rank (ascending) + Amihud Rank (descending).
        Select top 20% by composite score.
        """
        if self.universe_analytics_df is None:
            return

        # Get latest analytics per ticker up to current_date
        df = self.universe_analytics_df[self.universe_analytics_df['date'] <= current_date]
        if df.empty:
            return

        latest = df.groupby('ticker').last().reset_index()

        # Drop tickers with missing metrics
        scored = latest.dropna(subset=['vol_60', 'amihud_60'])
        if scored.empty:
            return

        # Rank volatility ascending (low vol = rank 1 = good for Low Volatility Anomaly)
        scored = scored.copy()
        scored['vol_rank'] = scored['vol_60'].rank(ascending=True, method='average')

        # Rank Amihud descending (high illiquidity = rank 1 = good for Illiquidity Premium)
        scored['amihud_rank'] = scored['amihud_60'].rank(ascending=False, method='average')

        # Composite score (higher is better)
        scored['composite_score'] = scored['vol_rank'] + scored['amihud_rank']

        # Top 20%
        threshold = scored['composite_score'].quantile(0.80)
        self.universe = set(scored[scored['composite_score'] >= threshold]['ticker'].tolist())

        month_key = current_date[:7]  # YYYY-MM
        self.universe_last_updated_month = month_key
        print(f"  Universe updated ({month_key}): {len(self.universe)} stocks selected from {len(scored)} scored")

    # =====================================================================
    # LLM / FINBERT ANALYSIS (Step 2: Sentiment Acceleration)
    # =====================================================================

    def llm_analysis(self, ticker, transcript, date):
        """
        FinBERT sentiment analysis with chunking, aggregation, and acceleration detection.

        Returns:
            dict with 'sentiment', 'net_sentiment', 'sentiment_acceleration'
            or None if no transcript/pipeline
        """
        if transcript is None or self.finbert_pipeline is None:
            return None

        # --- Cache check ---
        cache_key = f"{ticker}_{date}"
        if cache_key in self.llm_cache:
            self.llm_cache_hits += 1
            return self.llm_cache[cache_key]
        self.llm_cache_misses += 1

        try:
            # --- Chunking: split into sentences ---
            sentences = re.split(r'(?<=[.!?])\s+', transcript.strip())
            # Group sentences into chunks of ~400 chars to stay within 512 token limit
            chunks = []
            current_chunk = ""
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

            # --- Inference + Labeling ---
            positive_count = 0
            negative_count = 0
            neutral_count = 0

            for chunk in chunks:
                if not chunk or len(chunk) < 10:
                    continue
                try:
                    result = self.finbert_pipeline(chunk[:512])  # truncate to token limit
                    label = result[0]['label'].lower() if isinstance(result, list) and len(result) > 0 else 'neutral'
                    if isinstance(label, str):
                        if 'positive' in label:
                            positive_count += 1
                        elif 'negative' in label:
                            negative_count += 1
                        else:
                            neutral_count += 1
                except Exception:
                    neutral_count += 1

            total_valid = positive_count + negative_count + neutral_count
            if total_valid == 0:
                return None

            # --- Aggregation: Net Sentiment ---
            net_sentiment = (positive_count - negative_count) / total_valid

            # Dominant label
            if positive_count >= negative_count and positive_count >= neutral_count:
                dominant = 'positive'
            elif negative_count >= positive_count and negative_count >= neutral_count:
                dominant = 'negative'
            else:
                dominant = 'neutral'

            # --- Sentiment Acceleration ---
            quarter_key = get_quarter_key(date)
            self.sentiment_cache[(ticker, quarter_key)] = net_sentiment

            # Find previous quarter
            dt = pd.to_datetime(date)
            prev_quarter_dt = dt - pd.DateOffset(months=3)
            prev_quarter_key = f"{prev_quarter_dt.year}-Q{(prev_quarter_dt.month - 1) // 3 + 1}"

            prev_sentiment = self.sentiment_cache.get((ticker, prev_quarter_key))
            sentiment_acceleration = False
            if prev_sentiment is not None:
                sentiment_acceleration = (net_sentiment > prev_sentiment)

            result = {
                'sentiment': dominant,
                'net_sentiment': net_sentiment,
                'sentiment_acceleration': sentiment_acceleration,
                'positive_ratio': positive_count / total_valid if total_valid > 0 else 0,
                'chunks_processed': total_valid
            }

            self.llm_cache[cache_key] = result
            return result

        except Exception:
            return None

    # =====================================================================
    # DECISION LOGIC (Steps 3-5)
    # =====================================================================

    def make_decision(self, ticker, date, transcript, portfolio_state, analytics):
        """
        Multi-sleeve trading decision with regime filter.

        Regime Filter (200-SMA Market Breadth):
            Bull (>50%): Both sleeves fully active.
            Bear (<=50%): Exit all positions, no new entries.

        Sleeve 1 (Tactical Weekly):
            Entry: RSI < 30 OR (positive return AND volume > 1.5x 20d MA) on Friday.
            Exit: Sell the following week.

        Sleeve 2 (Fundamental Momentum):
            Entry: Sentiment Acceleration flag is True.
            Exit: ~5 weekly evaluations OR 3×ATR(14) trailing stop below highest high.
        """
        price = analytics.get('close', 0)
        if price <= 0:
            return 'HOLD'

        has_position = ticker in portfolio_state.get('positions', {})

        # --- Step 0: Monthly universe update ---
        current_month = date[:7]
        if self.universe_last_updated_month != current_month:
            self._update_universe(date)

        # --- Regime Filter: Check market breadth ---
        self.is_bull_regime = get_regime(date, self.market_breadth)

        # --- BEAR REGIME: Exit all positions, no new entries ---
        if not self.is_bull_regime:
            # Force exit Sleeve 1 positions
            if ticker in self.sleeve1_positions:
                del self.sleeve1_positions[ticker]
                return 'SELL'
            # Force exit Sleeve 2 positions
            if ticker in self.sleeve2_positions:
                del self.sleeve2_positions[ticker]
                return 'SELL'
            # Exit any other positions held outside of sleeve tracking
            if has_position:
                return 'SELL'
            return 'HOLD'  # No new entries in bear regime

        # === BULL REGIME: Normal sleeve logic below ===

        # --- Sleeve 1 EXIT — sell after 1 week holding ---
        if ticker in self.sleeve1_positions:
            entry_date = self.sleeve1_positions[ticker]
            if date > entry_date:
                # Time to exit: sell on this week's Friday (approximating Monday Close)
                del self.sleeve1_positions[ticker]
                return 'SELL'

        # --- Sleeve 2 EXIT — time-based or 3×ATR trailing stop ---
        if ticker in self.sleeve2_positions:
            pos_info = self.sleeve2_positions[ticker]
            pos_info['weeks_held'] += 1

            # Update highest high since entry
            pos_info['highest_high'] = max(pos_info['highest_high'], price)

            # Update ATR if available
            current_atr = analytics.get('atr_14')
            if current_atr is not None and not np.isnan(current_atr):
                pos_info['current_atr'] = current_atr

            # 3×ATR trailing stop below highest high
            atr = pos_info.get('current_atr', 0)
            if atr > 0:
                stop_level = pos_info['highest_high'] - (3.0 * atr)
                if price <= stop_level:
                    del self.sleeve2_positions[ticker]
                    return 'SELL'

            # Time-based exit: 5 weekly evaluations ≈ 25 trading days
            if pos_info['weeks_held'] >= 5:
                del self.sleeve2_positions[ticker]
                return 'SELL'

            return 'HOLD'

        # --- If ticker not in universe, don't open new positions ---
        in_universe = ticker in self.universe

        # If we have a position but ticker left the universe, exit it
        if has_position and not in_universe:
            if ticker not in self.sleeve1_positions and ticker not in self.sleeve2_positions:
                return 'SELL'

        if not in_universe:
            return 'HOLD'

        # --- Don't enter if already holding this ticker ---
        if has_position:
            return 'HOLD'

        # --- Sleeve 1 ENTRY (Tactical Weekly) ---
        rsi = analytics.get('rsi_14')
        daily_ret = analytics.get('daily_return')
        volume = analytics.get('volume')
        volume_ma = analytics.get('volume_ma_20')

        sleeve1_trigger = False
        if rsi is not None and not np.isnan(rsi):
            # Trigger A: RSI Mean Reversion
            if rsi < 30:
                sleeve1_trigger = True

        if not sleeve1_trigger and daily_ret is not None and volume is not None and volume_ma is not None:
            if not np.isnan(daily_ret) and not np.isnan(volume) and not np.isnan(volume_ma):
                # Trigger B: Volume-Confirmed Momentum
                if daily_ret > 0 and volume_ma > 0 and volume > 1.5 * volume_ma:
                    sleeve1_trigger = True

        if sleeve1_trigger:
            self.sleeve1_positions[ticker] = date
            return 'BUY'

        # --- Sleeve 2 ENTRY (Fundamental Momentum / Sentiment Acceleration) ---
        if transcript is not None:
            sentiment_result = self.llm_analysis(ticker, transcript, date)
            if sentiment_result is not None and sentiment_result.get('sentiment_acceleration', False):
                current_atr = analytics.get('atr_14', 0)
                if current_atr is None or np.isnan(current_atr):
                    current_atr = 0
                self.sleeve2_positions[ticker] = {
                    'entry_date': date,
                    'weeks_held': 0,
                    'highest_high': price,
                    'current_atr': current_atr
                }
                return 'BUY'

        return 'HOLD'

    # =====================================================================
    # EVALUATION ORCHESTRATION
    # =====================================================================

    def evaluate(self, verbose=False):
        """Evaluate strategy, resetting sleeve state for clean run."""
        if self.prices is None or self.earnings is None:
            raise ValueError("Must call set_data() before evaluate()")

        # Reset sleeve tracking state for fresh evaluation
        self.sleeve1_positions = {}
        self.sleeve2_positions = {}
        self.universe = set()
        self.universe_last_updated_month = None
        self.universe_analytics_df = None
        self.market_breadth = {}
        self.is_bull_regime = True
        # Note: We keep sentiment_cache — it accumulates knowledge across the backtest

        print("Running evaluation...")

        # Calculate analytics
        analytics = self.calculate_analytics(self.prices)
        analytics_lookup = build_analytics_lookup(analytics)

        # Run backtest
        print("Running backtest simulation...")
        sim = TradingSimulation(self.prices, self.earnings, STARTING_CASH)
        results = sim.run(
            lambda t, d, tr, ps, a: self.make_decision(t, d, tr, ps, a),
            analytics_lookup, verbose
        )

        return results
