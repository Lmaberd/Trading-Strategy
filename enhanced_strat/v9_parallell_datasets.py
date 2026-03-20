"""
EnhancedStrategy: Multi-Sleeve Trading Strategy (GPU-Optimized)
================================================================
Extends BaseStrategy with:
  - Step 1: Universe selection via volatility + Amihud illiquidity composite scoring
  - Step 2: FinBERT sentiment acceleration with bulk batched inference
  - Step 3: Dual-sleeve entry triggers (RSI mean reversion + volume momentum / sentiment acceleration)
  - Step 4: Time-based and ATR-based trailing-stop exits

GPU optimizations applied:
  - Vectorized Pandas groupby for analytics
  - Bulk FinBERT transcript precomputation so inference can run in large GPU batches
  - Weekly-aligned analytics lookup to reduce repeated daily-history scans during the backtest
"""
# Depends on notebook globals: BaseStrategy, TradingSimulation, STARTING_CASH

import re
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict

import numpy as np
import pandas as pd
from datasets import Dataset
from transformers.pipelines.pt_utils import KeyDataset

from enhanced_helpers.lookups import build_analytics_lookup_vectorised
from enhanced_helpers.sentiment import get_quarter_key


SENTENCE_SPLIT_RE = re.compile(r'(?<=[.!?])\s+')


class EnhancedStrategy(BaseStrategy):
    """
    Multi-Sleeve Trading Strategy with Minimalist Drawdown Defense.
    GPU work is pushed into batched FinBERT inference while portfolio execution
    stays sequential to preserve the original trading behavior.
    """

    def __init__(self, finbert_pipeline=None, rsi_threshold=40, vol_mult=2,
                 s1_atr_mult=2, s2_atr_mult=2, sleeve2_exit_weeks=5,
                 max_transcript_chars=10000):
        super().__init__(finbert_pipeline)
        self.rsi_threshold = rsi_threshold
        self.vol_mult = vol_mult
        self.s1_atr_mult = s1_atr_mult
        self.s2_atr_mult = s2_atr_mult
        self.sleeve2_exit_weeks = sleeve2_exit_weeks
        self.max_transcript_chars = max_transcript_chars

        # Universe selection state
        self.universe = set()
        self.universe_last_updated_month = None
        self.universe_analytics_df = None
        self.precomputed_universes = {}

        # Sleeve tracking
        self.sleeve1_positions = {}  # {ticker: {'entry_date': str, 'peak_price': float}}
        self.sleeve2_positions = {}  # {ticker: {'entry_date': str, 'weeks_held': int, 'peak_price': float}}
        self.sentiment_cache = {}

        # Bulk sentiment preprocessing state
        self.sentiment_batch_size = 64
        self.sentiment_super_batch_size = 4096
        self.preprocessing_workers = 3
        self.sentiment_chunk_workers = 4
        self.precomputed_llm_results = {}
        self.weekly_schedule = []

    # =====================================================================
    # DATA PREPARATION
    # =====================================================================

    def clean_data(self, prices_df, earnings_df):
        prices = prices_df.copy().drop_duplicates()
        earnings = earnings_df.copy().drop_duplicates()
        prices['date'] = pd.to_datetime(prices['date'])
        earnings['date'] = pd.to_datetime(earnings['date'])
        earnings = earnings[earnings['transcript'].fillna('').str.len() > 100]

        prices = prices.sort_values(['ticker', 'date']).reset_index(drop=True)
        earnings = earnings.sort_values(['ticker', 'date']).reset_index(drop=True)
        return prices, earnings

    # =====================================================================
    # ANALYTICS CALCULATION
    # =====================================================================

    def calculate_analytics(self, prices_df):
        print("Computing advanced technicals using fully vectorized Pandas groupby...")
        df = prices_df.copy().sort_values(['ticker', 'date'])

        ticker_index = df['ticker']
        grouped = df.groupby('ticker', sort=False)
        annualisation = np.sqrt(252.0)

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

    def _build_weekly_schedule(self, prices_df):
        min_date = pd.to_datetime(prices_df['date']).min()
        max_date = pd.to_datetime(prices_df['date']).max()
        return pd.date_range(start=min_date, end=max_date, freq='W-FRI').strftime('%Y-%m-%d').tolist()

    def _build_weekly_analytics_lookup(self, analytics_df, weekly_schedule):
        """
        Compress daily analytics down to the simulator's weekly schedule.
        The simulation still executes sequentially, but it no longer walks
        daily analytics rows to find the latest value for each week.
        """
        print("Aligning analytics to weekly schedule...")
        if analytics_df.empty or not weekly_schedule:
            return {}

        sorted_df = analytics_df.copy().sort_values(['ticker', 'date'])
        sorted_df['date'] = pd.to_datetime(sorted_df['date'])
        week_dates = pd.to_datetime(pd.Index(weekly_schedule))
        week_values = week_dates.to_numpy()

        aligned_frames = []
        for ticker, group in sorted_df.groupby('ticker', sort=False):
            group = group.reset_index(drop=True)
            group_dates = group['date'].to_numpy()
            positions = group_dates.searchsorted(week_values, side='right') - 1
            valid_mask = positions >= 0
            if not valid_mask.any():
                continue

            aligned = group.iloc[positions[valid_mask]].copy()
            aligned['date'] = week_dates[valid_mask].strftime('%Y-%m-%d')
            aligned['ticker'] = ticker
            aligned_frames.append(aligned)

        if not aligned_frames:
            return {}

        weekly_df = pd.concat(aligned_frames, ignore_index=True)
        print(f"  Weekly analytics aligned: {len(weekly_df):,} rows")
        return build_analytics_lookup_vectorised(weekly_df)

    def _build_weekly_market_views(self, analytics_lookup):
        """
        Transform weekly analytics into week-indexed views to avoid repeated
        nested lookups during the backtest.
        """
        weekly_records = defaultdict(list)
        weekly_prices = defaultdict(dict)

        for ticker, records in analytics_lookup.items():
            for week_date, analytics in records:
                weekly_records[week_date].append((ticker, analytics))
                price = analytics.get('close')
                if price is not None and not np.isnan(price):
                    weekly_prices[week_date][ticker] = price

        return weekly_records, weekly_prices

    def _build_weekly_earnings_lookup(self):
        if self.earnings is None or self.earnings.empty:
            return {}

        earnings_df = self.earnings[['ticker', 'date', 'transcript']].copy()
        earnings_df['date'] = pd.to_datetime(earnings_df['date'])
        earnings_df['week_end'] = (
            earnings_df['date']
            + pd.to_timedelta((4 - earnings_df['date'].dt.weekday) % 7, unit='D')
        ).dt.strftime('%Y-%m-%d')

        latest_earnings = (
            earnings_df.sort_values(['ticker', 'date'])
            .drop_duplicates(subset=['ticker', 'week_end'], keep='last')
        )

        return {
            (row.ticker, row.week_end): row.transcript
            for row in latest_earnings.itertuples(index=False)
        }

    def _buy_target(self, portfolio, ticker, price, date, target_value=5000):
        if ticker in portfolio['positions']:
            return 0
        max_shares = int(target_value // price)
        if max_shares <= 0:
            return 0

        cost = min(max_shares * price, portfolio['cash'])
        shares = int(cost // price)
        if shares <= 0:
            return 0

        actual_cost = shares * price
        portfolio['cash'] -= actual_cost
        portfolio['positions'][ticker] = {'shares': shares, 'buy_price': price}
        portfolio['trades'].append({
            'date': date,
            'ticker': ticker,
            'action': 'BUY',
            'shares': shares,
            'price': price,
            'value': actual_cost
        })
        return shares

    def _sell_position(self, portfolio, ticker, price, date):
        if ticker not in portfolio['positions']:
            return 0

        position = portfolio['positions'][ticker]
        shares = position['shares']
        proceeds = shares * price
        del portfolio['positions'][ticker]
        portfolio['cash'] += proceeds
        portfolio['trades'].append({
            'date': date,
            'ticker': ticker,
            'action': 'SELL',
            'shares': shares,
            'price': price,
            'value': proceeds
        })
        return shares

    def _get_portfolio_value(self, portfolio, current_prices):
        total = portfolio['cash']
        for ticker, pos in portfolio['positions'].items():
            price = current_prices.get(ticker)
            if price is not None:
                total += pos['shares'] * price
        return total

    def _get_portfolio_state(self, portfolio, current_prices):
        return {
            'cash': portfolio['cash'],
            'positions': {
                ticker: {'shares': pos['shares'], 'buy_price': pos['buy_price']}
                for ticker, pos in portfolio['positions'].items()
            },
            'total_value': self._get_portfolio_value(portfolio, current_prices)
        }

    def _run_fast_backtest(self, weekly_records, weekly_prices, weekly_earnings, verbose=False):
        """
        Faster replacement for TradingSimulation.run().
        It preserves the sequential weekly decision process, but removes the
        repeated per-ticker analytics scans and price-history fallbacks.
        """
        print("Running fast backtest loop...")
        portfolio = {
            'cash': STARTING_CASH,
            'positions': {},
            'trades': []
        }
        portfolio_history = []

        for i, week_date in enumerate(self.weekly_schedule):
            if verbose and i % 10 == 0:
                print(f"  Week {i+1}/{len(self.weekly_schedule)}: {week_date}")

            current_prices = weekly_prices.get(week_date, {})
            portfolio_state = self._get_portfolio_state(portfolio, current_prices)

            for ticker, analytics in weekly_records.get(week_date, []):
                transcript = weekly_earnings.get((ticker, week_date))
                decision = self.make_decision(ticker, week_date, transcript, portfolio_state, analytics)
                price = analytics.get('close')
                if price is None or np.isnan(price) or price <= 0:
                    continue

                if decision == 'BUY':
                    self._buy_target(portfolio, ticker, price, week_date, target_value=5000)
                elif decision == 'SELL':
                    self._sell_position(portfolio, ticker, price, week_date)

            portfolio_history.append({
                'date': week_date,
                'portfolio_value': self._get_portfolio_value(portfolio, current_prices),
                'cash': portfolio['cash'],
                'positions': len(portfolio['positions'])
            })

        final_date = self.weekly_schedule[-1]
        final_prices = weekly_prices.get(final_date, {})
        return {
            'trades': portfolio['trades'],
            'portfolio_history': portfolio_history,
            'final_portfolio': self._get_portfolio_state(portfolio, final_prices),
            'final_prices': final_prices
        }

    # =====================================================================
    # UNIVERSE SELECTION
    # =====================================================================

    def _select_universe(self, analytics_slice):
        latest = analytics_slice.groupby('ticker').last().reset_index()
        scored = latest.dropna(subset=['vol_60', 'amihud_60']).copy()
        if scored.empty:
            return set()

        scored['vol_rank'] = scored['vol_60'].rank(ascending=True, method='average')
        scored['amihud_rank'] = scored['amihud_60'].rank(ascending=False, method='average')
        scored['composite_score'] = scored['vol_rank'] + scored['amihud_rank']

        threshold = scored['composite_score'].quantile(0.80)
        return set(scored[scored['composite_score'] >= threshold]['ticker'].tolist())

    def _precompute_monthly_universes(self, weekly_schedule):
        self.precomputed_universes = {}
        if self.universe_analytics_df is None or not weekly_schedule:
            return

        print("Precomputing monthly universe snapshots...")
        first_week_by_month = {}
        for week_date in weekly_schedule:
            month_key = week_date[:7]
            if month_key not in first_week_by_month:
                first_week_by_month[month_key] = week_date

        for month_key, cutoff_date in first_week_by_month.items():
            analytics_slice = self.universe_analytics_df[self.universe_analytics_df['date'] <= cutoff_date]
            if analytics_slice.empty:
                continue
            self.precomputed_universes[month_key] = self._select_universe(analytics_slice)

        print(f"  Universe snapshots ready for {len(self.precomputed_universes)} months")

    def _update_universe(self, current_date):
        month_key = current_date[:7]
        if month_key in self.precomputed_universes:
            self.universe = self.precomputed_universes[month_key]
            self.universe_last_updated_month = month_key
            print(f"  Universe updated ({month_key}): {len(self.universe)} stocks selected.")
            return

        if self.universe_analytics_df is None:
            return

        analytics_slice = self.universe_analytics_df[self.universe_analytics_df['date'] <= current_date]
        if analytics_slice.empty:
            return

        self.universe = self._select_universe(analytics_slice)
        self.universe_last_updated_month = month_key
        print(f"  Universe updated ({month_key}): {len(self.universe)} stocks selected.")

    # =====================================================================
    # LLM / FINBERT ANALYSIS
    # =====================================================================

    def _trim_transcript(self, transcript):
        transcript = str(transcript).strip()
        if not transcript:
            return ""

        if len(transcript) <= self.max_transcript_chars:
            return transcript

        head_chars = self.max_transcript_chars // 2
        tail_chars = self.max_transcript_chars - head_chars
        return transcript[:head_chars] + "\n" + transcript[-tail_chars:]

    def _chunk_transcript(self, transcript):
        if transcript is None:
            return []

        transcript = self._trim_transcript(transcript)
        if not transcript:
            return []

        sentences = SENTENCE_SPLIT_RE.split(transcript)
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

        return [chunk[:512] for chunk in chunks if chunk and len(chunk) >= 10]

    def _label_bucket(self, result):
        label = result['label'].lower() if isinstance(result, dict) and 'label' in result else 'neutral'
        if 'positive' in label:
            return 0
        if 'negative' in label:
            return 1
        return 2

    def _build_sentiment_payload(self, pos, neg, neu):
        total = pos + neg + neu
        if total == 0:
            return None

        net_sentiment = (pos - neg) / total
        dominant = 'positive' if pos >= neg and pos >= neu else (
            'negative' if neg >= pos and neg >= neu else 'neutral'
        )
        return {
            'sentiment': dominant,
            'net_sentiment': net_sentiment,
            'positive_ratio': pos / total if total > 0 else 0.0,
            'chunks_processed': total
        }

    def _infer_sentiment_counts(self, valid_chunks):
        pos, neg, neu = 0, 0, 0
        try:
            ds = Dataset.from_dict({"text": valid_chunks})
            batch_results = list(self.finbert_pipeline(
                KeyDataset(ds, "text"),
                batch_size=self.sentiment_batch_size,
                truncation=True,
                max_length=512
            ))
            if isinstance(batch_results, dict):
                batch_results = [batch_results]

            for result in batch_results:
                bucket = self._label_bucket(result)
                if bucket == 0:
                    pos += 1
                elif bucket == 1:
                    neg += 1
                else:
                    neu += 1
        except Exception:
            neu = len(valid_chunks)
        return pos, neg, neu

    def _prepare_transcript_payload(self, row):
        cache_key = f"{row.ticker}_{row.date}"
        valid_chunks = self._chunk_transcript(row.transcript)
        if not valid_chunks:
            return None
        return cache_key, valid_chunks

    def _precompute_llm_analysis(self, allowed_tickers=None):
        self.precomputed_llm_results = {}
        if self.finbert_pipeline is None or self.earnings is None or self.earnings.empty:
            return

        print("Precomputing transcript sentiment in batched mode...")
        earnings_df = self.earnings[['ticker', 'date', 'transcript']].copy()
        if allowed_tickers:
            earnings_df = earnings_df[earnings_df['ticker'].isin(allowed_tickers)]
        if earnings_df.empty:
            print("  No eligible earnings transcripts for precomputation")
            return
        earnings_df['date'] = pd.to_datetime(earnings_df['date']).dt.strftime('%Y-%m-%d')

        chunk_texts = []
        chunk_owners = []
        seen_cache_keys = set()
        rows = list(earnings_df.itertuples(index=False))

        if self.sentiment_chunk_workers > 1 and len(rows) > 1:
            with ThreadPoolExecutor(max_workers=self.sentiment_chunk_workers) as executor:
                payloads = executor.map(self._prepare_transcript_payload, rows)
                for payload in payloads:
                    if payload is None:
                        continue
                    cache_key, valid_chunks = payload
                    if cache_key in seen_cache_keys:
                        continue
                    seen_cache_keys.add(cache_key)
                    chunk_texts.extend(valid_chunks)
                    chunk_owners.extend([cache_key] * len(valid_chunks))
        else:
            for row in rows:
                payload = self._prepare_transcript_payload(row)
                if payload is None:
                    continue
                cache_key, valid_chunks = payload
                if cache_key in seen_cache_keys:
                    continue
                seen_cache_keys.add(cache_key)
                chunk_texts.extend(valid_chunks)
                chunk_owners.extend([cache_key] * len(valid_chunks))

        if not chunk_texts:
            print("  No valid transcript chunks found for precomputation")
            return

        sentiment_counts = defaultdict(lambda: [0, 0, 0])
        for start in range(0, len(chunk_texts), self.sentiment_super_batch_size):
            batch_chunks = chunk_texts[start:start + self.sentiment_super_batch_size]
            batch_owners = chunk_owners[start:start + self.sentiment_super_batch_size]
            try:
                ds = Dataset.from_dict({"text": batch_chunks})
                batch_results = list(self.finbert_pipeline(
                    KeyDataset(ds, "text"),
                    batch_size=self.sentiment_batch_size,
                    truncation=True,
                    max_length=512
                ))
                if isinstance(batch_results, dict):
                    batch_results = [batch_results]
            except Exception:
                batch_results = None

            if batch_results is None or len(batch_results) != len(batch_chunks):
                for owner in batch_owners:
                    sentiment_counts[owner][2] += 1
                continue

            for owner, result in zip(batch_owners, batch_results):
                bucket = self._label_bucket(result)
                sentiment_counts[owner][bucket] += 1

        for cache_key, counts in sentiment_counts.items():
            payload = self._build_sentiment_payload(*counts)
            if payload is not None:
                self.precomputed_llm_results[cache_key] = payload

        print(
            f"  Precomputed {len(self.precomputed_llm_results):,} transcript sentiments "
            f"from {len(chunk_texts):,} chunks"
        )

    def llm_analysis(self, ticker, transcript, date):
        if transcript is None or self.finbert_pipeline is None:
            return None

        cache_key = f"{ticker}_{date}"
        if cache_key in self.llm_cache:
            self.llm_cache_hits += 1
            return self.llm_cache[cache_key]

        self.llm_cache_misses += 1
        try:
            base_result = self.precomputed_llm_results.get(cache_key)
            if base_result is None:
                valid_chunks = self._chunk_transcript(transcript)
                if not valid_chunks:
                    return None

                pos, neg, neu = self._infer_sentiment_counts(valid_chunks)
                base_result = self._build_sentiment_payload(pos, neg, neu)
                if base_result is None:
                    return None
                self.precomputed_llm_results[cache_key] = base_result

            quarter_key = get_quarter_key(date)
            net_sentiment = base_result['net_sentiment']
            self.sentiment_cache[(ticker, quarter_key)] = net_sentiment

            dt = pd.to_datetime(date)
            prev_dt = dt - pd.DateOffset(months=3)
            prev_key = f"{prev_dt.year}-Q{(prev_dt.month - 1) // 3 + 1}"
            prev_sentiment = self.sentiment_cache.get((ticker, prev_key))
            accel = (net_sentiment > prev_sentiment) if prev_sentiment is not None else False

            result = dict(base_result)
            result['sentiment_acceleration'] = accel
            self.llm_cache[cache_key] = result
            return result
        except Exception:
            return None

    # =====================================================================
    # DECISION LOGIC
    # =====================================================================

    def make_decision(self, ticker, date, transcript, portfolio_state, analytics):
        price = analytics.get('close', 0)
        if price <= 0:
            return 'HOLD'
        has_pos = ticker in portfolio_state.get('positions', {})

        if self.universe_last_updated_month != date[:7]:
            self._update_universe(date)

        atr_14 = analytics.get('atr_14', price * 0.05)

        # Exit logic
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

        # Defensive veto logic
        vol_10 = analytics.get('vol_10')
        vol_60 = analytics.get('vol_60')
        ma_200 = analytics.get('ma_200')

        vol_veto = (
            vol_10 is not None and vol_60 is not None
            and not np.isnan(vol_10) and not np.isnan(vol_60)
            and vol_10 > 1.5 * vol_60
        )
        is_downtrend = ma_200 is not None and price < ma_200

        # Entry logic
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
        self.precomputed_universes = {}
        self.precomputed_llm_results = {}
        self.weekly_schedule = self._build_weekly_schedule(self.prices)

        print("Running evaluation...")
        with ThreadPoolExecutor(max_workers=self.preprocessing_workers) as executor:
            analytics_future = executor.submit(self.calculate_analytics, self.prices)
            earnings_future = executor.submit(self._build_weekly_earnings_lookup)
            sentiment_future = executor.submit(self._precompute_llm_analysis)

            analytics = analytics_future.result()
            weekly_earnings = earnings_future.result()
            sentiment_future.result()

        with ThreadPoolExecutor(max_workers=2) as executor:
            universe_future = executor.submit(self._precompute_monthly_universes, self.weekly_schedule)
            analytics_lookup_future = executor.submit(
                self._build_weekly_analytics_lookup,
                analytics,
                self.weekly_schedule
            )

            universe_future.result()
            analytics_lookup = analytics_lookup_future.result()

        weekly_records, weekly_prices = self._build_weekly_market_views(analytics_lookup)

        return self._run_fast_backtest(weekly_records, weekly_prices, weekly_earnings, verbose)
