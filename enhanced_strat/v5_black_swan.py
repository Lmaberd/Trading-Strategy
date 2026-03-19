"""
EnhancedStrategy: Multi-Sleeve Trading Strategy (Black Swan Defensive Edition)
=============================================================================
Extends BaseStrategy with:
  - Step 1: Universe Selection via Volatility + Amihud composite scoring (monthly)
  - Step 2: FinBERT Sentiment Acceleration (Earnings NLP)
  - Step 3: Dual-sleeve entry triggers (RSI < 35 + Volume > 1.25x MA)
  - Step 4: Chandelier Exits (3x ATR) and time-based exits
  - Step 5: Master Regime Filter, Volatility Acceleration Vetoes, and Systemic Circuit Breakers
"""
# Depends on notebook globals: BaseStrategy, TradingSimulation, STARTING_CASH

import re
import numpy as np
import pandas as pd
from collections import defaultdict

from enhanced_helpers.sentiment import get_quarter_key
from enhanced_helpers.lookups import build_analytics_lookup


class EnhancedStrategy(BaseStrategy):
    """
    Multi-Sleeve Trading Strategy with robust drawdown defense.
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
        self.universe_analytics_df = None
        self.momentum_winners = set()  # H1: Track Top Decile Winners to avoid skewness

        # --- Sleeve tracking ---
        self.sleeve1_positions = {}
        self.sleeve2_positions = {}

        # --- Sentiment cache ---
        self.sentiment_cache = {}

        # --- Capital allocation ---
        self.sleeve1_allocation = 0.60
        self.sleeve2_allocation = 0.40

    # =====================================================================
    # DATA PREPARATION
    # =====================================================================

    def set_data(self, prices_df, earnings_df):
        print("Cleaning and preprocessing data...")
        self.prices, self.earnings = self.clean_data(prices_df, earnings_df)
        print(f"Data ready: {len(self.prices):,} price records")

    def clean_data(self, prices_df, earnings_df):
        prices = prices_df.copy().drop_duplicates()
        earnings = earnings_df.copy().drop_duplicates()
        prices['date'] = pd.to_datetime(prices['date'])
        earnings['date'] = pd.to_datetime(earnings['date'])
        earnings = earnings[earnings['transcript'].str.len() > 100]

        prices = prices.sort_values(['ticker', 'date']).reset_index(drop=True)
        return prices, earnings

    # =====================================================================
    # ANALYTICS CALCULATION (Including Market Regimes & ATR)
    # =====================================================================

    def calculate_analytics(self, prices_df):
        print("Computing advanced technicals and market regimes...")
        results = []

        for ticker in prices_df['ticker'].unique():
            df = prices_df[prices_df['ticker'] == ticker].copy().sort_values('date')

            # --- Basic returns & Moving Averages ---
            df['daily_return'] = df['close'].pct_change()
            df['abs_return'] = df['daily_return'].abs()
            df['ma_200'] = df['close'].rolling(200, min_periods=50).mean()  # H7: Trend Filter
            df['ma_50'] = df['close'].rolling(50, min_periods=1).mean()
            df['ret_12m'] = df['close'].pct_change(252)  # H1: 12-Month Momentum

            # --- Volatility metrics (H3 & Universe) ---
            df['vol_60'] = df['daily_return'].rolling(60, min_periods=30).std() * np.sqrt(252)
            df['vol_10'] = df['daily_return'].rolling(10, min_periods=5).std() * np.sqrt(252)  # H3: Vol Acceleration

            # --- Amihud Illiquidity (H4 & H5) ---
            df['dollar_volume'] = df['close'] * df['volume']
            df['amihud_daily'] = df['abs_return'] / df['dollar_volume'].replace(0, np.nan)
            df['amihud_60'] = df['amihud_daily'].rolling(60, min_periods=30).mean()

            # --- ATR Calculation (Chandelier Exit) ---
            df['prev_close'] = df['close'].shift()
            if 'high' in df.columns and 'low' in df.columns:
                tr1 = df['high'] - df['low']
                tr2 = (df['high'] - df['prev_close']).abs()
                tr3 = (df['low'] - df['prev_close']).abs()
                df['tr'] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            else:
                df['tr'] = (df['close'] - df['prev_close']).abs()
            df['atr_14'] = df['tr'].rolling(14, min_periods=1).mean()

            # --- Entry Triggers ---
            delta = df['close'].diff()
            gain = delta.where(delta > 0, 0.0)
            loss = (-delta).where(delta < 0, 0.0)
            avg_gain = gain.rolling(14, min_periods=14).mean()
            avg_loss = loss.rolling(14, min_periods=14).mean()
            rs = avg_gain / avg_loss.replace(0, np.nan)
            df['rsi_14'] = 100.0 - (100.0 / (1.0 + rs))
            df['volume_ma_20'] = df['volume'].rolling(20, min_periods=10).mean()

            results.append(df[['ticker', 'date', 'open', 'close', 'volume',
                               'daily_return', 'vol_60', 'vol_10', 'amihud_60',
                               'rsi_14', 'volume_ma_20', 'ma_200', 'ret_12m', 'atr_14']])

        result_df = pd.concat(results, ignore_index=True)

        # --- MARKET REGIME CALCULATIONS (H2, H6) ---
        print("Calculating Systemic Market Breadth and Realized Volatility...")
        # 1. Market Breadth (% above 200 SMA)
        result_df['above_200'] = (result_df['close'] > result_df['ma_200']).astype(int)
        daily_breadth = result_df.groupby('date')['above_200'].mean().rename('market_breadth_pct')

        # 2. Market Realized Volatility (Equal weight synthetic index)
        daily_ret_market = result_df.groupby('date')['daily_return'].mean()
        market_vol = daily_ret_market.rolling(20, min_periods=10).std() * np.sqrt(252)
        market_vol = market_vol.rename('market_vol_20')

        # Merge Systemic indicators back to the main dataframe
        result_df = result_df.merge(daily_breadth, on='date', how='left')
        result_df = result_df.merge(market_vol, on='date', how='left')
        result_df.drop(columns=['above_200'], inplace=True)

        result_df['date'] = result_df['date'].dt.strftime('%Y-%m-%d')
        self.universe_analytics_df = result_df
        return result_df

    # =====================================================================
    # UNIVERSE SELECTION (Dynamic Regime Filters)
    # =====================================================================

    def _update_universe(self, current_date):
        if self.universe_analytics_df is None:
            return

        df = self.universe_analytics_df[self.universe_analytics_df['date'] <= current_date]
        if df.empty:
            return

        latest = df.groupby('ticker').last().reset_index()
        scored = latest.dropna(subset=['vol_60', 'amihud_60', 'ma_200']).copy()
        if scored.empty:
            return

        # Track Top Decile Momentum Winners (H1)
        scored['ret_12m_rank'] = scored['ret_12m'].rank(pct=True, na_option='bottom')
        self.momentum_winners = set(scored[scored['ret_12m_rank'] >= 0.90]['ticker'])

        # Get Current Market Breadth
        regime_breadth = latest['market_breadth_pct'].mean()

        # --- CORRECTION REGIME PENALTIES ---
        if regime_breadth < 0.50:
            # H7: Trend Filter (Drop stocks below 200 SMA)
            scored = scored[scored['close'] >= scored['ma_200']]

            # H5: Exclude High IVOL (Top 20% vol penalty)
            vol_thresh = scored['vol_60'].quantile(0.80)
            scored = scored[scored['vol_60'] <= vol_thresh]

            # H4: Exclude Severe Illiquidity (Top 20% Amihud penalty)
            amihud_thresh = scored['amihud_60'].quantile(0.80)
            scored = scored[scored['amihud_60'] <= amihud_thresh]

        # Composite score
        scored['vol_rank'] = scored['vol_60'].rank(ascending=True, method='average')
        scored['amihud_rank'] = scored['amihud_60'].rank(ascending=False, method='average')
        scored['composite_score'] = scored['vol_rank'] + scored['amihud_rank']

        # Widen Universe funnel to Top 40% (60th percentile)
        threshold = scored['composite_score'].quantile(0.60)
        self.universe = set(scored[scored['composite_score'] >= threshold]['ticker'].tolist())

        month_key = current_date[:7]
        self.universe_last_updated_month = month_key
        print(f"  Universe updated ({month_key}): {len(self.universe)} stocks selected. Breadth: {regime_breadth:.1%}")

    # =====================================================================
    # LLM / FINBERT ANALYSIS (Sentiment Acceleration)
    # =====================================================================

    def llm_analysis(self, ticker, transcript, date):
        # ... (Existing FinBERT logic remains entirely unchanged) ...
        if transcript is None or self.finbert_pipeline is None:
            return None
        cache_key = f"{ticker}_{date}"
        if cache_key in self.llm_cache:
            self.llm_cache_hits += 1
            return self.llm_cache[cache_key]
        self.llm_cache_misses += 1

        try:
            sentences = re.split(r'(?<=[.!?])\s+', transcript.strip())
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

            positive_count, negative_count, neutral_count = 0, 0, 0
            for chunk in chunks:
                if not chunk or len(chunk) < 10:
                    continue
                try:
                    result = self.finbert_pipeline(chunk[:512])
                    label = result[0]['label'].lower() if isinstance(result, list) and len(result) > 0 else 'neutral'
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
            net_sentiment = (positive_count - negative_count) / total_valid

            if positive_count >= negative_count and positive_count >= neutral_count:
                dominant = 'positive'
            elif negative_count >= positive_count and negative_count >= neutral_count:
                dominant = 'negative'
            else:
                dominant = 'neutral'

            quarter_key = get_quarter_key(date)
            self.sentiment_cache[(ticker, quarter_key)] = net_sentiment
            dt = pd.to_datetime(date)
            prev_quarter_dt = dt - pd.DateOffset(months=3)
            prev_quarter_key = f"{prev_quarter_dt.year}-Q{(prev_quarter_dt.month - 1) // 3 + 1}"
            prev_sentiment = self.sentiment_cache.get((ticker, prev_quarter_key))
            sentiment_acceleration = (net_sentiment > prev_sentiment) if prev_sentiment is not None else False

            result = {
                'sentiment': dominant, 'net_sentiment': net_sentiment,
                'sentiment_acceleration': sentiment_acceleration,
                'positive_ratio': positive_count / total_valid if total_valid > 0 else 0,
                'chunks_processed': total_valid
            }
            self.llm_cache[cache_key] = result
            return result
        except Exception:
            return None

    # =====================================================================
    # DECISION LOGIC (Drawdown Vetoes & Entry/Exit)
    # =====================================================================

    def make_decision(self, ticker, date, transcript, portfolio_state, analytics):
        price = analytics.get('close', 0)
        if price <= 0:
            return 'HOLD'
        has_position = ticker in portfolio_state.get('positions', {})

        # Monthly update
        current_month = date[:7]
        if self.universe_last_updated_month != current_month:
            self._update_universe(date)

        # Extraction of key risk metrics
        market_vol = analytics.get('market_vol_20', 0)
        market_breadth = analytics.get('market_breadth_pct', 1.0)
        vol_10 = analytics.get('vol_10')
        vol_60 = analytics.get('vol_60')
        atr_14 = analytics.get('atr_14', price * 0.05)  # fallback to 5% if missing

        # --- H2 & H6: BLACK SWAN CIRCUIT BREAKER ---
        # Correlation Breakdown & Vol Target Defense
        if not np.isnan(market_vol) and not np.isnan(market_breadth):
            if market_vol > 0.35 or market_breadth < 0.15:
                # Systemic Shock: Liquidate everything immediately
                if ticker in self.sleeve1_positions:
                    del self.sleeve1_positions[ticker]
                    return 'SELL'
                if ticker in self.sleeve2_positions:
                    del self.sleeve2_positions[ticker]
                    return 'SELL'
                return 'HOLD'  # Block all new entries

        # --- Sleeve 1 EXIT ---
        if ticker in self.sleeve1_positions:
            if date > self.sleeve1_positions[ticker]:
                del self.sleeve1_positions[ticker]
                return 'SELL'

        # --- Sleeve 2 EXIT (Chandelier) ---
        if ticker in self.sleeve2_positions:
            pos_info = self.sleeve2_positions[ticker]
            pos_info['weeks_held'] += 1
            pos_info['peak_price'] = max(pos_info['peak_price'], price)

            # Volatility-adjusted trailing stop (3 * ATR)
            if price <= pos_info['peak_price'] - (3 * atr_14):
                del self.sleeve2_positions[ticker]
                return 'SELL'

            if pos_info['weeks_held'] >= 5:
                del self.sleeve2_positions[ticker]
                return 'SELL'
            return 'HOLD'

        in_universe = ticker in self.universe
        if has_position and not in_universe:
            if ticker not in self.sleeve1_positions and ticker not in self.sleeve2_positions:
                return 'SELL'
        if not in_universe or has_position:
            return 'HOLD'

        # --- DEFENSIVE VETO LOGIC FOR NEW ENTRIES ---

        # H3: Volatility Acceleration Veto (Blocks 24% probability of >10% crash)
        vol_acceleration_veto = False
        if vol_10 is not None and vol_60 is not None and not np.isnan(vol_10) and not np.isnan(vol_60):
            if vol_10 > 1.5 * vol_60:
                vol_acceleration_veto = True

        # H1: Momentum Skewness Veto
        momentum_skewness_veto = False
        if market_vol > 0.20 and ticker in self.momentum_winners:
            momentum_skewness_veto = True

        # --- ENTRY: Tactical Weekly ---
        rsi = analytics.get('rsi_14')
        daily_ret = analytics.get('daily_return')
        volume = analytics.get('volume')
        volume_ma = analytics.get('volume_ma_20')

        sleeve1_trigger = False
        if rsi is not None and not np.isnan(rsi):
            # Relaxed RSI to 35, enforced by Vetoes
            if rsi < 35 and not vol_acceleration_veto and not momentum_skewness_veto:
                sleeve1_trigger = True

        if not sleeve1_trigger and daily_ret is not None and volume is not None and volume_ma is not None:
            if not np.isnan(daily_ret) and not np.isnan(volume) and not np.isnan(volume_ma):
                # Relaxed Momentum volume to 1.25x
                if daily_ret > 0 and volume_ma > 0 and volume > 1.25 * volume_ma and not vol_acceleration_veto:
                    sleeve1_trigger = True

        if sleeve1_trigger:
            self.sleeve1_positions[ticker] = date
            return 'BUY'

        # --- ENTRY: Fundamental Momentum ---
        if transcript is not None and not vol_acceleration_veto:
            sentiment_result = self.llm_analysis(ticker, transcript, date)
            if sentiment_result is not None and sentiment_result.get('sentiment_acceleration', False):
                self.sleeve2_positions[ticker] = {
                    'entry_date': date,
                    'weeks_held': 0,
                    'peak_price': price
                }
                return 'BUY'

        return 'HOLD'

    # =====================================================================
    # EVALUATION ORCHESTRATION
    # =====================================================================

    def evaluate(self, verbose=False):
        if self.prices is None or self.earnings is None:
            raise ValueError("Must call set_data() before evaluate()")

        self.sleeve1_positions = {}
        self.sleeve2_positions = {}
        self.universe = set()
        self.universe_last_updated_month = None
        self.universe_analytics_df = None
        self.momentum_winners = set()

        print("Running evaluation...")
        analytics = self.calculate_analytics(self.prices)
        analytics_lookup = build_analytics_lookup(analytics)

        print("Running backtest simulation...")
        sim = TradingSimulation(self.prices, self.earnings, STARTING_CASH)
        results = sim.run(
            lambda t, d, tr, ps, a: self.make_decision(t, d, tr, ps, a),
            analytics_lookup, verbose
        )
        return results
