"""
EnhancedStrategy V1: Low Volatility, High Liquidity Universe
============================================================
Based on V14 (Optimised Parkinson Volatility).
Universe selection changed to target the LEAST volatile and MOST liquid stocks.
Both park_vol_60 and amihud_60 are ranked ascending so that low values
(= low volatility, high liquidity) receive low ranks. The bottom 25th
percentile of the composite score is then selected.

Optimisations applied on top of V13:

  1. return_all_scores=True bug (sleeve 2 was completely broken)
     - Notebook loads FinBERT with return_all_scores=True, so each pipeline result
       is a LIST of {label,score} dicts, not a single dict.
     - Old _label_bucket did isinstance(result, dict) → always returned 'neutral'
       → net_sentiment always 0 → sentiment_acceleration never True → sleeve 2 dead.
     - Fix: _extract_scores() handles both list and dict output formats.
     - Bonus: we already have all 3 probability scores per chunk, so we sum them
       directly instead of counting winners — more accurate aggregation.

  2. FP16 inference (~2x FinBERT throughput)
     - Model converted to float16 in-place before the GPU inference call.
     - Only applied when model is on CUDA. No notebook change required.

  3. Monthly universe: per-ticker precomputation (eliminates 216× groupby)
     - Old: 216 × groupby('ticker').last() on slices up to 1.3M rows.
     - New: precompute per-ticker sorted {dates, park60, amihud60} arrays once,
       then do one np.searchsorted per ticker per month (216 × 340 = 73k lookups).

  4. Analytics alignment: pd.merge_asof replaces Python for-loop over 340 tickers
     - Old: Python loop → groupby → searchsorted → iloc → concat (340 iterations).
     - New: cross-join weeks × tickers (93k rows), single vectorised merge_asof call.

Notebook globals required: BaseStrategy, STARTING_CASH
"""

import re
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict

import numpy as np
import pandas as pd


SENTENCE_SPLIT_RE = re.compile(r'(?<=[.!?])\s+')


class EnhancedStrategy(BaseStrategy):

    def __init__(self, finbert_pipeline=None, rsi_threshold=40, vol_mult=2,
                 s1_atr_mult=2, s2_atr_mult=2, sleeve2_exit_weeks=5,
                 max_transcript_chars=4000, park_veto_mult=1.75):
        super().__init__(finbert_pipeline)
        self.rsi_threshold = rsi_threshold
        self.vol_mult = vol_mult
        self.s1_atr_mult = s1_atr_mult
        self.s2_atr_mult = s2_atr_mult
        self.sleeve2_exit_weeks = sleeve2_exit_weeks
        self.max_transcript_chars = max_transcript_chars
        self.park_veto_mult = park_veto_mult

        self.universe = set()
        self.universe_last_updated_month = None
        self.universe_analytics_df = None
        self.precomputed_universes = {}

        self.sleeve1_positions = {}
        self.sleeve2_positions = {}
        self.sentiment_cache = {}

        self.sentiment_batch_size = 128
        self.sentiment_chunk_workers = 4
        self.precomputed_llm_results = {}
        self.weekly_schedule = []

    def _reset_run_state(self):
        self.universe = set()
        self.universe_last_updated_month = None
        self.universe_analytics_df = None
        self.precomputed_universes = {}
        self.sleeve1_positions = {}
        self.sleeve2_positions = {}
        self.sentiment_cache = {}
        self.precomputed_llm_results = {}
        self.weekly_schedule = []

        if hasattr(self, 'llm_cache'):
            self.llm_cache = {}
        if hasattr(self, 'llm_cache_hits'):
            self.llm_cache_hits = 0
        if hasattr(self, 'llm_cache_misses'):
            self.llm_cache_misses = 0

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
        daily_return = grouped['close'].pct_change()
        close_rolling = df['close'].groupby(ticker_index, sort=False)
        volume_rolling = df['volume'].groupby(ticker_index, sort=False)

        df['daily_return'] = daily_return
        df['ma_200'] = close_rolling.rolling(200, min_periods=50).mean().reset_index(level=0, drop=True)

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

            safe_high = df['high'].replace(0, np.nan)
            safe_low = df['low'].replace(0, np.nan)
            log_hl = np.log(safe_high.div(safe_low))
            park_daily_var = log_hl.pow(2).div(4.0 * np.log(2.0))
            park_daily_var = park_daily_var.replace([np.inf, -np.inf], np.nan)
            park_var_10 = park_daily_var.groupby(ticker_index, sort=False).rolling(10, min_periods=5).mean().reset_index(level=0, drop=True)
            park_var_60 = park_daily_var.groupby(ticker_index, sort=False).rolling(60, min_periods=30).mean().reset_index(level=0, drop=True)
            df['park_vol_10'] = np.sqrt(252.0 * park_var_10)
            df['park_vol_60'] = np.sqrt(252.0 * park_var_60)
        else:
            true_range = (df['close'] - prev_close).abs()
            daily_return_rolling = daily_return.groupby(ticker_index, sort=False)
            df['park_vol_10'] = daily_return_rolling.rolling(10, min_periods=5).std().mul(np.sqrt(252.0)).reset_index(level=0, drop=True)
            df['park_vol_60'] = daily_return_rolling.rolling(60, min_periods=30).std().mul(np.sqrt(252.0)).reset_index(level=0, drop=True)
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
            'park_vol_10', 'park_vol_60', 'amihud_60', 'rsi_14', 'volume_ma_20',
            'ma_200', 'atr_14'
        ]].copy()
        result_df['date'] = result_df['date'].dt.strftime('%Y-%m-%d')

        self.universe_analytics_df = result_df
        print(f"  Analytics computed: {len(result_df):,} rows for {result_df['ticker'].nunique()} tickers")
        return result_df

    @staticmethod
    def _get_quarter_key(date_str):
        dt = pd.to_datetime(date_str)
        return f"{dt.year}-Q{(dt.month - 1) // 3 + 1}"

    @staticmethod
    def _build_analytics_lookup_vectorised(analytics_df):
        print("Building analytics lookup (vectorised)...")
        lookup = {}
        sorted_df = analytics_df.sort_values(['ticker', 'date'])
        for ticker, group in sorted_df.groupby('ticker', sort=False):
            dates = group['date'].tolist()
            records = group.to_dict('records')
            lookup[ticker] = list(zip(dates, records))
        print(f"  Lookup built for {len(lookup)} tickers")
        return lookup

    def _build_weekly_schedule(self, prices_df):
        min_date = pd.to_datetime(prices_df['date']).min()
        max_date = pd.to_datetime(prices_df['date']).max()
        return pd.date_range(start=min_date, end=max_date, freq='W-FRI').strftime('%Y-%m-%d').tolist()

    def _build_weekly_analytics_frame(self, analytics_df, weekly_schedule):
        # fix #4: pd.merge_asof replaces Python for-loop over 340 tickers
        print("Aligning analytics to weekly schedule...")
        if analytics_df.empty or not weekly_schedule:
            return analytics_df.iloc[0:0].copy()

        sorted_df = analytics_df.copy()
        sorted_df['date'] = pd.to_datetime(sorted_df['date'])
        # merge_asof requires the join key (date) to be globally monotone —
        # sort by date only, NOT ['ticker','date'] which resets dates per ticker
        sorted_df = sorted_df.sort_values(['date', 'ticker']).reset_index(drop=True)

        week_dates = pd.to_datetime(pd.Index(weekly_schedule))
        all_tickers = sorted_df['ticker'].unique()

        # Cross-join: 276 weeks × 340 tickers = ~93k rows
        # Sort by date globally (same reason — merge_asof requires monotone left key)
        weeks_expanded = (
            pd.DataFrame({'date': week_dates})
            .merge(pd.DataFrame({'ticker': all_tickers}), how='cross')
            .sort_values(['date', 'ticker'])
            .reset_index(drop=True)
        )

        # Single vectorised asof join — finds most recent analytics row per (ticker, week)
        weekly_df = pd.merge_asof(
            weeks_expanded,
            sorted_df,
            on='date',
            by='ticker',
            direction='backward'
        ).dropna(subset=['close'])

        weekly_df['date'] = weekly_df['date'].dt.strftime('%Y-%m-%d')
        print(f"  Weekly analytics aligned: {len(weekly_df):,} rows")
        return weekly_df

    def _build_weekly_market_views(self, weekly_df, allowed_tickers=None):
        if weekly_df.empty:
            return {}, {}

        if allowed_tickers:
            weekly_df = weekly_df[weekly_df['ticker'].isin(allowed_tickers)]

        weekly_records = {}
        weekly_prices = {}

        for week_date, group in weekly_df.groupby('date', sort=False):
            records = group.to_dict('records')
            weekly_records[week_date] = [(record['ticker'], record) for record in records]
            weekly_prices[week_date] = {
                record['ticker']: record['close']
                for record in records
                if record.get('close') is not None and not np.isnan(record['close'])
            }

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
            'date': date, 'ticker': ticker, 'action': 'BUY',
            'shares': shares, 'price': price, 'value': actual_cost
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
            'date': date, 'ticker': ticker, 'action': 'SELL',
            'shares': shares, 'price': price, 'value': proceeds
        })
        return shares

    def _get_portfolio_value(self, portfolio, current_prices):
        total = portfolio['cash']
        for ticker, pos in portfolio['positions'].items():
            price = current_prices.get(ticker)
            if price is not None:
                total += pos['shares'] * price
        return total

    def _get_portfolio_state(self, portfolio):
        return {
            'cash': portfolio['cash'],
            'positions': portfolio['positions']
        }

    def _run_fast_backtest(self, weekly_records, weekly_prices, weekly_earnings, verbose=False):
        print("Running fast backtest loop...")
        portfolio = {'cash': STARTING_CASH, 'positions': {}, 'trades': []}
        portfolio_history = []
        get_week_prices = weekly_prices.get
        get_week_records = weekly_records.get
        get_weekly_earnings = weekly_earnings.get
        make_decision = self.make_decision
        buy_target = self._buy_target
        sell_position = self._sell_position
        get_portfolio_value = self._get_portfolio_value

        for i, week_date in enumerate(self.weekly_schedule):
            if verbose and i % 10 == 0:
                print(f"  Week {i+1}/{len(self.weekly_schedule)}: {week_date}")

            current_prices = get_week_prices(week_date, {})
            portfolio_state = self._get_portfolio_state(portfolio)

            for ticker, analytics in get_week_records(week_date, ()):
                transcript = get_weekly_earnings((ticker, week_date))
                decision = make_decision(ticker, week_date, transcript, portfolio_state, analytics)
                price = analytics.get('close')
                if price is None or np.isnan(price) or price <= 0:
                    continue
                if decision == 'BUY':
                    buy_target(portfolio, ticker, price, week_date, target_value=5000)
                elif decision == 'SELL':
                    sell_position(portfolio, ticker, price, week_date)

            portfolio_value = get_portfolio_value(portfolio, current_prices)

            portfolio_history.append({
                'date': week_date,
                'portfolio_value': portfolio_value,
                'cash': portfolio['cash'],
                'positions': len(portfolio['positions'])
            })

        final_date = self.weekly_schedule[-1]
        final_prices = get_week_prices(final_date, {})
        return {
            'trades': portfolio['trades'],
            'portfolio_history': portfolio_history,
            'final_portfolio': {
                'cash': portfolio['cash'],
                'positions': {
                    ticker: {'shares': pos['shares'], 'buy_price': pos['buy_price']}
                    for ticker, pos in portfolio['positions'].items()
                },
                'total_value': get_portfolio_value(portfolio, final_prices)
            },
            'final_prices': final_prices
        }

    # =====================================================================
    # UNIVERSE SELECTION
    # =====================================================================

    def _select_universe(self, analytics_slice):
        """Used only by _update_universe fallback. Normal path uses _select_universe_from_latest."""
        latest = analytics_slice.groupby('ticker').last().reset_index()
        scored = latest.dropna(subset=['park_vol_60', 'amihud_60']).copy()
        if scored.empty:
            return set()
        scored['vol_rank'] = scored['park_vol_60'].rank(ascending=True, method='average')
        scored['amihud_rank'] = scored['amihud_60'].rank(ascending=True, method='average')
        scored['composite_score'] = scored['vol_rank'] + scored['amihud_rank']
        threshold = scored['composite_score'].quantile(0.25)
        return set(scored[scored['composite_score'] <= threshold]['ticker'].tolist())

    def _select_universe_from_latest(self, latest_dict):
        """
        Select universe from {ticker: (park_vol_60, amihud_60)} dict.
        Avoids groupby — input is already the latest value per ticker.
        """
        if not latest_dict:
            return set()

        tickers = list(latest_dict.keys())
        park60s = np.array([v[0] for v in latest_dict.values()], dtype=float)
        amihud60s = np.array([v[1] for v in latest_dict.values()], dtype=float)

        valid = ~(np.isnan(park60s) | np.isnan(amihud60s))
        if not valid.any():
            return set()

        valid_tickers = [t for t, v in zip(tickers, valid) if v]
        park_valid = park60s[valid]
        amihud_valid = amihud60s[valid]

        vol_rank    = pd.Series(park_valid).rank(ascending=True,  method='average').values
        amihud_rank = pd.Series(amihud_valid).rank(ascending=True,  method='average').values
        composite   = vol_rank + amihud_rank
        threshold   = np.quantile(composite, 0.25)

        return {t for t, s in zip(valid_tickers, composite) if s <= threshold}

    def _precompute_monthly_universes(self, weekly_df):
        # fix #3: precompute per-ticker sorted arrays, then binary search per month
        # replaces 216 × groupby('ticker').last() on slices up to 1.3M rows
        self.precomputed_universes = {}
        if weekly_df is None or weekly_df.empty:
            return

        print("Precomputing monthly universe snapshots...")

        first_week_by_month = {}
        for week_date in self.weekly_schedule:
            month_key = week_date[:7]
            if month_key not in first_week_by_month:
                first_week_by_month[month_key] = week_date

        weekly_groups = {
            week_date: group[['ticker', 'park_vol_60', 'amihud_60']]
            for week_date, group in weekly_df.groupby('date', sort=False)
        }

        for month_key, cutoff_date in sorted(first_week_by_month.items()):
            group = weekly_groups.get(cutoff_date)
            if group is None or group.empty:
                continue

            latest = {
                ticker: (park_vol_60, amihud_60)
                for ticker, park_vol_60, amihud_60 in group.itertuples(index=False, name=None)
            }
            universe = self._select_universe_from_latest(latest)
            if universe:
                self.precomputed_universes[month_key] = universe

        print(f"  Universe snapshots ready for {len(self.precomputed_universes)} months")

    def _update_universe(self, current_date):
        month_key = current_date[:7]
        if month_key in self.precomputed_universes:
            self.universe = self.precomputed_universes[month_key]
            self.universe_last_updated_month = month_key
            return

        if self.universe_analytics_df is None:
            return

        analytics_slice = self.universe_analytics_df[self.universe_analytics_df['date'] <= current_date]
        if analytics_slice.empty:
            return

        self.universe = self._select_universe(analytics_slice)
        self.universe_last_updated_month = month_key

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

    def _extract_scores(self, result):
        """
        fix #1: handle both pipeline output formats.
          return_all_scores=True  → result is a list of {label, score} dicts
          return_all_scores=False → result is a single {label, score} dict
        Returns (positive_score, negative_score, neutral_score).
        """
        if isinstance(result, list):
            # return_all_scores=True: we have exact probabilities for all 3 classes
            scores = {r['label'].lower(): r.get('score', 0.0) for r in result}
            return (scores.get('positive', 0.0),
                    scores.get('negative', 0.0),
                    scores.get('neutral',  0.0))
        if isinstance(result, dict):
            label = result.get('label', 'neutral').lower()
            score = result.get('score', 1.0)
            if 'positive' in label:
                return score, 0.0, 1.0 - score
            if 'negative' in label:
                return 0.0, score, 1.0 - score
            return 0.0, 0.0, 1.0
        return 0.0, 0.0, 1.0

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
            'positive_ratio': pos / total,
            'chunks_processed': total
        }

    def _infer_sentiment_counts(self, valid_chunks):
        """Fallback: run inference on a single transcript's chunks (cache miss path)."""
        pos = neg = neu = 0.0
        try:
            results = list(self.finbert_pipeline(
                valid_chunks,
                batch_size=self.sentiment_batch_size,
                truncation=True,
                max_length=512
            ))
            if isinstance(results, dict):
                results = [results]
            for r in results:
                p, n, u = self._extract_scores(r)
                pos += p; neg += n; neu += u
        except Exception:
            neu = float(len(valid_chunks))
        return pos, neg, neu

    def _prepare_transcript_payload(self, row):
        cache_key = f"{row.ticker}_{row.date}"
        valid_chunks = self._chunk_transcript(row.transcript)
        if not valid_chunks:
            return None
        return cache_key, valid_chunks

    @staticmethod
    def _get_previous_quarter_key(date_str):
        dt = pd.to_datetime(date_str) - pd.DateOffset(months=3)
        return f"{dt.year}-Q{(dt.month - 1) // 3 + 1}"

    def _precompute_llm_analysis(self, allowed_tickers=None):
        self.precomputed_llm_results = {}
        self.sentiment_cache = {}
        if self.finbert_pipeline is None or self.earnings is None or self.earnings.empty:
            return

        print("Precomputing transcript sentiment in batched mode...")
        earnings_df = self.earnings[['ticker', 'date', 'transcript']].copy()
        if allowed_tickers:
            earnings_df = earnings_df[earnings_df['ticker'].isin(allowed_tickers)]
            print(f"  Filtering to {len(allowed_tickers)} universe tickers "
                  f"({len(earnings_df):,} transcripts)")
        if earnings_df.empty:
            print("  No eligible earnings transcripts")
            return
        earnings_df['date'] = pd.to_datetime(earnings_df['date']).dt.strftime('%Y-%m-%d')
        earnings_df = earnings_df.sort_values(['ticker', 'date']).reset_index(drop=True)

        # Chunk all transcripts in parallel (CPU text processing)
        rows = list(earnings_df.itertuples(index=False))
        if self.sentiment_chunk_workers > 1 and len(rows) > 1:
            with ThreadPoolExecutor(max_workers=self.sentiment_chunk_workers) as executor:
                payloads = list(executor.map(self._prepare_transcript_payload, rows))
        else:
            payloads = [self._prepare_transcript_payload(row) for row in rows]

        chunk_texts = []
        chunk_owners = []
        seen_cache_keys = set()
        for payload in payloads:
            if payload is None:
                continue
            cache_key, valid_chunks = payload
            if cache_key in seen_cache_keys:
                continue
            seen_cache_keys.add(cache_key)
            chunk_texts.extend(valid_chunks)
            chunk_owners.extend([cache_key] * len(valid_chunks))

        if not chunk_texts:
            print("  No valid transcript chunks found")
            return

        # fix #2: convert model to FP16 on GPU for ~2x inference throughput
        try:
            model = self.finbert_pipeline.model
            if next(model.parameters()).is_cuda:
                model.half()
                print("  Model converted to FP16 for inference")
        except Exception:
            pass

        print(f"  Running FinBERT on {len(chunk_texts):,} chunks (batch_size={self.sentiment_batch_size})...")

        # fix #1: pass list directly — single pipeline call, no super-batch loop
        sentiment_sums = defaultdict(lambda: [0.0, 0.0, 0.0])
        try:
            batch_results = list(self.finbert_pipeline(
                chunk_texts,
                batch_size=self.sentiment_batch_size,
                truncation=True,
                max_length=512
            ))
            if isinstance(batch_results, dict):
                batch_results = [batch_results]
        except Exception as e:
            print(f"  FinBERT inference failed: {e} — marking all as neutral")
            batch_results = None

        if batch_results is not None and len(batch_results) == len(chunk_texts):
            for owner, result in zip(chunk_owners, batch_results):
                p, n, u = self._extract_scores(result)
                s = sentiment_sums[owner]
                s[0] += p; s[1] += n; s[2] += u
        else:
            for owner in chunk_owners:
                sentiment_sums[owner][2] += 1.0  # fallback: neutral

        for cache_key, (p, n, u) in sentiment_sums.items():
            payload = self._build_sentiment_payload(p, n, u)
            if payload is not None:
                self.precomputed_llm_results[cache_key] = payload

        quarter_sentiments = {}
        transcript_rows = []
        for ticker, date in earnings_df[['ticker', 'date']].itertuples(index=False, name=None):
            cache_key = f"{ticker}_{date}"
            payload = self.precomputed_llm_results.get(cache_key)
            if payload is None:
                continue
            quarter_key = self._get_quarter_key(date)
            quarter_sentiments[(ticker, quarter_key)] = payload['net_sentiment']
            transcript_rows.append((ticker, date, cache_key))

        self.sentiment_cache.update(quarter_sentiments)
        for ticker, date, cache_key in transcript_rows:
            payload = self.precomputed_llm_results[cache_key]
            prev_key = self._get_previous_quarter_key(date)
            prev_sentiment = quarter_sentiments.get((ticker, prev_key))
            full_result = dict(payload)
            full_result['sentiment_acceleration'] = (
                prev_sentiment is not None
                and payload['net_sentiment'] > prev_sentiment
            )
            self.precomputed_llm_results[cache_key] = full_result

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
            result = self.precomputed_llm_results.get(cache_key)
            if result is None:
                valid_chunks = self._chunk_transcript(transcript)
                if not valid_chunks:
                    return None
                pos, neg, neu = self._infer_sentiment_counts(valid_chunks)
                payload = self._build_sentiment_payload(pos, neg, neu)
                if payload is None:
                    return None
                quarter_key = self._get_quarter_key(date)
                prev_key = self._get_previous_quarter_key(date)
                prev_sentiment = self.sentiment_cache.get((ticker, prev_key))
                result = dict(payload)
                result['sentiment_acceleration'] = (
                    prev_sentiment is not None
                    and payload['net_sentiment'] > prev_sentiment
                )
                self.sentiment_cache[(ticker, quarter_key)] = payload['net_sentiment']
                self.precomputed_llm_results[cache_key] = result

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

        park_vol_10 = analytics.get('park_vol_10')
        park_vol_60 = analytics.get('park_vol_60')
        ma_200 = analytics.get('ma_200')

        vol_veto = (
            park_vol_10 is not None and park_vol_60 is not None
            and not np.isnan(park_vol_10) and not np.isnan(park_vol_60)
            and park_vol_10 > self.park_veto_mult * park_vol_60
        )
        is_downtrend = ma_200 is not None and price < ma_200

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

        self._reset_run_state()
        self.weekly_schedule = self._build_weekly_schedule(self.prices)

        print("Running evaluation...")

        # Step 1: CPU preprocessing in parallel
        with ThreadPoolExecutor(max_workers=2) as executor:
            analytics_future = executor.submit(self.calculate_analytics, self.prices)
            earnings_future = executor.submit(self._build_weekly_earnings_lookup)
            analytics = analytics_future.result()
            weekly_earnings = earnings_future.result()

        # Step 2: Align daily analytics to the weekly schedule once
        weekly_analytics = self._build_weekly_analytics_frame(analytics, self.weekly_schedule)

        # Step 3: Derive monthly universes from the same weekly snapshot
        self._precompute_monthly_universes(weekly_analytics)

        # Step 4: Collect all tickers that ever qualify for the universe
        universe_tickers = set()
        for tickers in self.precomputed_universes.values():
            universe_tickers.update(tickers)
        print(f"  Universe covers {len(universe_tickers)} unique tickers across all months "
              f"(out of {self.prices['ticker'].nunique()} total)")

        # Step 5: FinBERT only on universe tickers
        if universe_tickers:
            weekly_earnings = {
                key: transcript
                for key, transcript in weekly_earnings.items()
                if key[0] in universe_tickers
            }

        self._precompute_llm_analysis(allowed_tickers=universe_tickers)

        # Step 6: Build week-indexed views and run backtest
        weekly_records, weekly_prices = self._build_weekly_market_views(
            weekly_analytics,
            allowed_tickers=universe_tickers
        )
        return self._run_fast_backtest(weekly_records, weekly_prices, weekly_earnings, verbose)
