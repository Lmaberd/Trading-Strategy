"""
EnhancedStrategy V18.3: Combined Universe With Sentiment Sleeve
===============================================================
Based on V19's composite-score universe and bucket-specific technical sleeve,
with the sentiment sleeve from V16 reintroduced.

Universe construction:
  - `top`: top 25% of `park_vol_60.rank(ascending=True) + amihud_60.rank(ascending=False)`
  - `bottom`: bottom 25% of `park_vol_60.rank(ascending=False) + amihud_60.rank(ascending=False)`

Trading sleeves:
  - sleeve 1: technical entries with bucket-specific `vol_mult`, `park_veto_mult`,
    and ATR exits
  - sleeve 2: V16-style FinBERT sentiment acceleration entries with shared
    ATR and max-hold controls

Notebook globals required: BaseStrategy, STARTING_CASH
"""

import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd


SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


class EnhancedStrategy(BaseStrategy):

    def __init__(
        self,
        finbert_pipeline=None,
        rsi_threshold=40,
        vol_mult=2,
        s1_atr_mult=2,
        s2_atr_mult=2,
        sleeve2_exit_weeks=5,
        max_transcript_chars=4000,
        park_veto_mult=1.5,
        bottom_vol_mult=1.75,
        top_vol_mult=2,
        bottom_park_veto_mult=1.25,
        top_park_veto_mult=None,
        bottom_atr_mult=None,
        top_atr_mult=None,
    ):
        super().__init__(finbert_pipeline)
        self.rsi_threshold = rsi_threshold
        self.vol_mult = vol_mult
        self.s1_atr_mult = s1_atr_mult
        self.s2_atr_mult = s2_atr_mult
        self.sleeve2_exit_weeks = sleeve2_exit_weeks
        self.max_transcript_chars = max_transcript_chars
        self.park_veto_mult = park_veto_mult

        self.bottom_vol_mult = vol_mult if bottom_vol_mult is None else bottom_vol_mult
        self.top_vol_mult = vol_mult if top_vol_mult is None else top_vol_mult
        self.bottom_park_veto_mult = (
            park_veto_mult if bottom_park_veto_mult is None else bottom_park_veto_mult
        )
        self.top_park_veto_mult = (
            park_veto_mult if top_park_veto_mult is None else top_park_veto_mult
        )
        self.bottom_atr_mult = s1_atr_mult if bottom_atr_mult is None else bottom_atr_mult
        self.top_atr_mult = s1_atr_mult if top_atr_mult is None else top_atr_mult

        self.universe = set()
        self.universe_buckets = {}
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

        if not hasattr(self, "llm_cache"):
            self.llm_cache = {}
        if not hasattr(self, "llm_cache_hits"):
            self.llm_cache_hits = 0
        if not hasattr(self, "llm_cache_misses"):
            self.llm_cache_misses = 0

    def _reset_run_state(self):
        self.universe = set()
        self.universe_buckets = {}
        self.universe_last_updated_month = None
        self.universe_analytics_df = None
        self.precomputed_universes = {}
        self.sleeve1_positions = {}
        self.sleeve2_positions = {}
        self.sentiment_cache = {}
        self.precomputed_llm_results = {}
        self.weekly_schedule = []

        self.llm_cache = {}
        self.llm_cache_hits = 0
        self.llm_cache_misses = 0

    # =====================================================================
    # DATA PREPARATION
    # =====================================================================

    def clean_data(self, prices_df, earnings_df):
        prices = prices_df.copy().drop_duplicates()
        prices["date"] = pd.to_datetime(prices["date"])
        prices = prices.sort_values(["ticker", "date"]).reset_index(drop=True)

        if earnings_df is None:
            earnings = pd.DataFrame()
        else:
            earnings = earnings_df.copy().drop_duplicates()
            if not earnings.empty and "date" in earnings.columns:
                earnings["date"] = pd.to_datetime(earnings["date"])
                sort_cols = [col for col in ("ticker", "date") if col in earnings.columns]
                if sort_cols:
                    earnings = earnings.sort_values(sort_cols).reset_index(drop=True)
                else:
                    earnings = earnings.reset_index(drop=True)
            else:
                earnings = earnings.reset_index(drop=True)

        return prices, earnings

    # =====================================================================
    # ANALYTICS CALCULATION
    # =====================================================================

    def calculate_analytics(self, prices_df):
        print("Computing advanced technicals using fully vectorized Pandas groupby...")
        df = prices_df.copy().sort_values(["ticker", "date"])

        ticker_index = df["ticker"]
        grouped = df.groupby("ticker", sort=False)
        daily_return = grouped["close"].pct_change()
        close_rolling = df["close"].groupby(ticker_index, sort=False)
        volume_rolling = df["volume"].groupby(ticker_index, sort=False)

        df["daily_return"] = daily_return
        df["ma_200"] = (
            close_rolling.rolling(200, min_periods=200).mean().reset_index(level=0, drop=True)
        )

        dollar_volume = df["close"] * df["volume"]
        amihud_daily = daily_return.abs().div(dollar_volume.replace(0, np.nan))
        df["amihud_60"] = (
            amihud_daily.groupby(ticker_index, sort=False)
            .rolling(60, min_periods=60)
            .mean()
            .reset_index(level=0, drop=True)
        )

        prev_close = grouped["close"].shift()
        if "high" in df.columns and "low" in df.columns:
            true_range = pd.concat(
                [
                    df["high"] - df["low"],
                    (df["high"] - prev_close).abs(),
                    (df["low"] - prev_close).abs(),
                ],
                axis=1,
            ).max(axis=1)

            safe_high = df["high"].replace(0, np.nan)
            safe_low = df["low"].replace(0, np.nan)
            log_hl = np.log(safe_high.div(safe_low))
            park_daily_var = log_hl.pow(2).div(4.0 * np.log(2.0))
            park_daily_var = park_daily_var.replace([np.inf, -np.inf], np.nan)
            park_var_10 = (
                park_daily_var.groupby(ticker_index, sort=False)
                .rolling(10, min_periods=10)
                .mean()
                .reset_index(level=0, drop=True)
            )
            park_var_60 = (
                park_daily_var.groupby(ticker_index, sort=False)
                .rolling(60, min_periods=60)
                .mean()
                .reset_index(level=0, drop=True)
            )
            df["park_vol_10"] = np.sqrt(252.0 * park_var_10)
            df["park_vol_60"] = np.sqrt(252.0 * park_var_60)
        else:
            true_range = (df["close"] - prev_close).abs()
            daily_return_rolling = daily_return.groupby(ticker_index, sort=False)
            df["park_vol_10"] = (
                daily_return_rolling.rolling(10, min_periods=10).std().mul(np.sqrt(252.0))
                .reset_index(level=0, drop=True)
            )
            df["park_vol_60"] = (
                daily_return_rolling.rolling(60, min_periods=60).std().mul(np.sqrt(252.0))
                .reset_index(level=0, drop=True)
            )

        df["atr_14"] = (
            true_range.groupby(ticker_index, sort=False)
            .rolling(14, min_periods=14)
            .mean()
            .reset_index(level=0, drop=True)
        )

        delta = grouped["close"].diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)
        avg_gain = (
            gain.groupby(ticker_index, sort=False)
            .rolling(14, min_periods=14)
            .mean()
            .reset_index(level=0, drop=True)
        )
        avg_loss = (
            loss.groupby(ticker_index, sort=False)
            .rolling(14, min_periods=14)
            .mean()
            .reset_index(level=0, drop=True)
        )
        rs = avg_gain.div(avg_loss.replace(0, np.nan))
        df["rsi_14"] = 100.0 - (100.0 / (1.0 + rs))

        df["volume_ma_20"] = (
            volume_rolling.rolling(20, min_periods=20).mean().reset_index(level=0, drop=True)
        )

        result_df = df[
            [
                "ticker",
                "date",
                "open",
                "close",
                "volume",
                "daily_return",
                "park_vol_10",
                "park_vol_60",
                "amihud_60",
                "rsi_14",
                "volume_ma_20",
                "ma_200",
                "atr_14",
            ]
        ].copy()
        result_df["date"] = result_df["date"].dt.strftime("%Y-%m-%d")

        self.universe_analytics_df = result_df
        print(
            f"  Analytics computed: {len(result_df):,} rows for "
            f"{result_df['ticker'].nunique()} tickers"
        )
        return result_df

    @staticmethod
    def _get_quarter_key(date_str):
        dt = pd.to_datetime(date_str)
        return f"{dt.year}-Q{(dt.month - 1) // 3 + 1}"

    def _build_weekly_schedule(self, prices_df):
        min_date = pd.to_datetime(prices_df["date"]).min()
        max_date = pd.to_datetime(prices_df["date"]).max()
        return pd.date_range(start=min_date, end=max_date, freq="W-FRI").strftime("%Y-%m-%d").tolist()

    def _build_weekly_analytics_frame(self, analytics_df, weekly_schedule):
        print("Aligning analytics to weekly schedule...")
        if analytics_df.empty or not weekly_schedule:
            return analytics_df.iloc[0:0].copy()

        sorted_df = analytics_df.copy()
        sorted_df["date"] = pd.to_datetime(sorted_df["date"])
        sorted_df = sorted_df.sort_values(["date", "ticker"]).reset_index(drop=True)

        week_dates = pd.to_datetime(pd.Index(weekly_schedule))
        all_tickers = sorted_df["ticker"].unique()

        weeks_expanded = (
            pd.DataFrame({"date": week_dates})
            .merge(pd.DataFrame({"ticker": all_tickers}), how="cross")
            .sort_values(["date", "ticker"])
            .reset_index(drop=True)
        )

        weekly_df = pd.merge_asof(
            weeks_expanded,
            sorted_df,
            on="date",
            by="ticker",
            direction="backward",
        ).dropna(subset=["close"])

        weekly_df["date"] = weekly_df["date"].dt.strftime("%Y-%m-%d")
        print(f"  Weekly analytics aligned: {len(weekly_df):,} rows")
        return weekly_df

    def _build_weekly_market_views(self, weekly_df, allowed_tickers=None):
        if weekly_df.empty:
            return {}, {}

        if allowed_tickers:
            weekly_df = weekly_df[weekly_df["ticker"].isin(allowed_tickers)]

        weekly_records = {}
        weekly_prices = {}

        for week_date, group in weekly_df.groupby("date", sort=False):
            records = group.to_dict("records")
            weekly_records[week_date] = [(record["ticker"], record) for record in records]
            weekly_prices[week_date] = {
                record["ticker"]: record["close"]
                for record in records
                if record.get("close") is not None and not np.isnan(record["close"])
            }

        return weekly_records, weekly_prices

    def _build_weekly_earnings_lookup(self):
        if self.earnings is None or self.earnings.empty:
            return {}

        earnings_df = self.earnings[["ticker", "date", "transcript"]].copy()
        earnings_df["date"] = pd.to_datetime(earnings_df["date"])
        earnings_df["week_end"] = (
            earnings_df["date"]
            + pd.to_timedelta((4 - earnings_df["date"].dt.weekday) % 7, unit="D")
        ).dt.strftime("%Y-%m-%d")

        latest_earnings = (
            earnings_df.sort_values(["ticker", "date"])
            .drop_duplicates(subset=["ticker", "week_end"], keep="last")
        )

        return {
            (row.ticker, row.week_end): row.transcript
            for row in latest_earnings.itertuples(index=False)
        }

    def _buy_target(self, portfolio, ticker, price, date, target_value=5000):
        if ticker in portfolio["positions"]:
            return 0
        max_shares = int(target_value // price)
        if max_shares <= 0:
            return 0
        cost = min(max_shares * price, portfolio["cash"])
        shares = int(cost // price)
        if shares <= 0:
            return 0
        actual_cost = shares * price
        portfolio["cash"] -= actual_cost
        portfolio["positions"][ticker] = {"shares": shares, "buy_price": price}
        portfolio["trades"].append(
            {
                "date": date,
                "ticker": ticker,
                "action": "BUY",
                "shares": shares,
                "price": price,
                "value": actual_cost,
            }
        )
        return shares

    def _sell_position(self, portfolio, ticker, price, date):
        if ticker not in portfolio["positions"]:
            return 0
        position = portfolio["positions"][ticker]
        shares = position["shares"]
        proceeds = shares * price
        del portfolio["positions"][ticker]
        portfolio["cash"] += proceeds
        portfolio["trades"].append(
            {
                "date": date,
                "ticker": ticker,
                "action": "SELL",
                "shares": shares,
                "price": price,
                "value": proceeds,
            }
        )
        return shares

    def _get_portfolio_value(self, portfolio, current_prices):
        total = portfolio["cash"]
        for ticker, pos in portfolio["positions"].items():
            price = current_prices.get(ticker)
            if price is not None:
                total += pos["shares"] * price
        return total

    def _get_portfolio_state(self, portfolio):
        return {
            "cash": portfolio["cash"],
            "positions": portfolio["positions"],
        }

    def _run_fast_backtest(self, weekly_records, weekly_prices, weekly_earnings, verbose=False):
        print("Running fast backtest loop...")
        portfolio = {"cash": STARTING_CASH, "positions": {}, "trades": []}
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
                print(f"  Week {i + 1}/{len(self.weekly_schedule)}: {week_date}")

            current_prices = get_week_prices(week_date, {})
            portfolio_state = self._get_portfolio_state(portfolio)

            for ticker, analytics in get_week_records(week_date, ()):
                transcript = get_weekly_earnings((ticker, week_date))
                decision = make_decision(ticker, week_date, transcript, portfolio_state, analytics)
                price = analytics.get("close")
                if price is None or np.isnan(price) or price <= 0:
                    continue
                if decision == "BUY":
                    buy_target(portfolio, ticker, price, week_date, target_value=5000)
                elif decision == "SELL":
                    sell_position(portfolio, ticker, price, week_date)

            portfolio_value = get_portfolio_value(portfolio, current_prices)
            portfolio_history.append(
                {
                    "date": week_date,
                    "portfolio_value": portfolio_value,
                    "cash": portfolio["cash"],
                    "positions": len(portfolio["positions"]),
                }
            )

        final_date = self.weekly_schedule[-1]
        final_prices = get_week_prices(final_date, {})
        return {
            "trades": portfolio["trades"],
            "portfolio_history": portfolio_history,
            "final_portfolio": {
                "cash": portfolio["cash"],
                "positions": {
                    ticker: {"shares": pos["shares"], "buy_price": pos["buy_price"]}
                    for ticker, pos in portfolio["positions"].items()
                },
                "total_value": get_portfolio_value(portfolio, final_prices),
            },
            "final_prices": final_prices,
        }

    # =====================================================================
    # UNIVERSE SELECTION
    # =====================================================================

    @staticmethod
    def _build_universe_buckets(scored):
        if scored.empty:
            return {}

        scored = scored.reset_index(drop=True).copy()

        top_composite = (
            scored["park_vol_60"].rank(ascending=True, method="average")
            + scored["amihud_60"].rank(ascending=False, method="average")
        )
        bottom_composite = (
            scored["park_vol_60"].rank(ascending=False, method="average")
            + scored["amihud_60"].rank(ascending=False, method="average")
        )

        top_threshold = top_composite.quantile(0.75)
        bottom_threshold = bottom_composite.quantile(0.25)

        bucket_map = {}
        for ticker, top_score, bottom_score in zip(
            scored["ticker"],
            top_composite.to_numpy(dtype=float),
            bottom_composite.to_numpy(dtype=float),
        ):
            in_top = top_score >= top_threshold
            in_bottom = bottom_score <= bottom_threshold

            if in_top and not in_bottom:
                bucket_map[ticker] = "top"
            elif in_bottom and not in_top:
                bucket_map[ticker] = "bottom"
            elif in_top and in_bottom:
                top_distance = top_score - top_threshold
                bottom_distance = bottom_threshold - bottom_score
                bucket_map[ticker] = "top" if top_distance >= bottom_distance else "bottom"

        return bucket_map

    def _select_universe(self, analytics_slice):
        latest = analytics_slice.groupby("ticker").last().reset_index()
        scored = latest.dropna(subset=["park_vol_60", "amihud_60"]).copy()
        if scored.empty:
            return {}
        return self._build_universe_buckets(scored[["ticker", "park_vol_60", "amihud_60"]])

    def _select_universe_from_latest(self, latest_dict):
        if not latest_dict:
            return {}

        tickers = list(latest_dict.keys())
        park60s = np.array([values[0] for values in latest_dict.values()], dtype=float)
        amihud60s = np.array([values[1] for values in latest_dict.values()], dtype=float)

        valid = ~(np.isnan(park60s) | np.isnan(amihud60s))
        if not valid.any():
            return {}

        scored = pd.DataFrame(
            {
                "ticker": [ticker for ticker, is_valid in zip(tickers, valid) if is_valid],
                "park_vol_60": park60s[valid],
                "amihud_60": amihud60s[valid],
            }
        )
        return self._build_universe_buckets(scored)

    def _precompute_monthly_universes(self, weekly_df):
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
            week_date: group[["ticker", "park_vol_60", "amihud_60"]]
            for week_date, group in weekly_df.groupby("date", sort=False)
        }

        for month_key, cutoff_date in sorted(first_week_by_month.items()):
            group = weekly_groups.get(cutoff_date)
            if group is None or group.empty:
                continue

            latest = {
                ticker: (park_vol_60, amihud_60)
                for ticker, park_vol_60, amihud_60 in group.itertuples(index=False, name=None)
            }
            bucket_map = self._select_universe_from_latest(latest)
            if bucket_map:
                self.precomputed_universes[month_key] = bucket_map

        print(f"  Universe snapshots ready for {len(self.precomputed_universes)} months")

    def _update_universe(self, current_date):
        month_key = current_date[:7]
        if month_key in self.precomputed_universes:
            self.universe_buckets = dict(self.precomputed_universes[month_key])
            self.universe = set(self.universe_buckets)
            self.universe_last_updated_month = month_key
            return

        if self.universe_analytics_df is None:
            return

        analytics_slice = self.universe_analytics_df[self.universe_analytics_df["date"] <= current_date]
        if analytics_slice.empty:
            return

        self.universe_buckets = self._select_universe(analytics_slice)
        self.universe = set(self.universe_buckets)
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
        if isinstance(result, list):
            scores = {row["label"].lower(): row.get("score", 0.0) for row in result}
            return (
                scores.get("positive", 0.0),
                scores.get("negative", 0.0),
                scores.get("neutral", 0.0),
            )
        if isinstance(result, dict):
            label = result.get("label", "neutral").lower()
            score = result.get("score", 1.0)
            if "positive" in label:
                return score, 0.0, 1.0 - score
            if "negative" in label:
                return 0.0, score, 1.0 - score
            return 0.0, 0.0, 1.0
        return 0.0, 0.0, 1.0

    def _build_sentiment_payload(self, pos, neg, neu):
        total = pos + neg + neu
        if total == 0:
            return None
        net_sentiment = (pos - neg) / total
        dominant = "positive" if pos >= neg and pos >= neu else (
            "negative" if neg >= pos and neg >= neu else "neutral"
        )
        return {
            "sentiment": dominant,
            "net_sentiment": net_sentiment,
            "positive_ratio": pos / total,
            "chunks_processed": total,
        }

    def _infer_sentiment_counts(self, valid_chunks):
        pos = neg = neu = 0.0
        try:
            results = list(
                self.finbert_pipeline(
                    valid_chunks,
                    batch_size=self.sentiment_batch_size,
                    truncation=True,
                    max_length=512,
                )
            )
            if isinstance(results, dict):
                results = [results]
            for result in results:
                p, n, u = self._extract_scores(result)
                pos += p
                neg += n
                neu += u
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
        earnings_df = self.earnings[["ticker", "date", "transcript"]].copy()
        if allowed_tickers:
            earnings_df = earnings_df[earnings_df["ticker"].isin(allowed_tickers)]
            print(
                f"  Filtering to {len(allowed_tickers)} universe tickers "
                f"({len(earnings_df):,} transcripts)"
            )
        if earnings_df.empty:
            print("  No eligible earnings transcripts")
            return

        earnings_df["date"] = pd.to_datetime(earnings_df["date"]).dt.strftime("%Y-%m-%d")
        earnings_df = earnings_df.sort_values(["ticker", "date"]).reset_index(drop=True)

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

        try:
            model = self.finbert_pipeline.model
            if next(model.parameters()).is_cuda:
                model.half()
                print("  Model converted to FP16 for inference")
        except Exception:
            pass

        print(
            f"  Running FinBERT on {len(chunk_texts):,} chunks "
            f"(batch_size={self.sentiment_batch_size})..."
        )

        sentiment_sums = defaultdict(lambda: [0.0, 0.0, 0.0])
        try:
            batch_results = list(
                self.finbert_pipeline(
                    chunk_texts,
                    batch_size=self.sentiment_batch_size,
                    truncation=True,
                    max_length=512,
                )
            )
            if isinstance(batch_results, dict):
                batch_results = [batch_results]
        except Exception as exc:
            print(f"  FinBERT inference failed: {exc} - marking all as neutral")
            batch_results = None

        if batch_results is not None and len(batch_results) == len(chunk_texts):
            for owner, result in zip(chunk_owners, batch_results):
                p, n, u = self._extract_scores(result)
                sums = sentiment_sums[owner]
                sums[0] += p
                sums[1] += n
                sums[2] += u
        else:
            for owner in chunk_owners:
                sentiment_sums[owner][2] += 1.0

        for cache_key, (p, n, u) in sentiment_sums.items():
            payload = self._build_sentiment_payload(p, n, u)
            if payload is not None:
                self.precomputed_llm_results[cache_key] = payload

        quarter_sentiments = {}
        transcript_rows = []
        for ticker, date in earnings_df[["ticker", "date"]].itertuples(index=False, name=None):
            cache_key = f"{ticker}_{date}"
            payload = self.precomputed_llm_results.get(cache_key)
            if payload is None:
                continue
            quarter_key = self._get_quarter_key(date)
            quarter_sentiments[(ticker, quarter_key)] = payload["net_sentiment"]
            transcript_rows.append((ticker, date, cache_key))

        self.sentiment_cache.update(quarter_sentiments)
        for ticker, date, cache_key in transcript_rows:
            payload = self.precomputed_llm_results[cache_key]
            prev_key = self._get_previous_quarter_key(date)
            prev_sentiment = quarter_sentiments.get((ticker, prev_key))
            full_result = dict(payload)
            full_result["sentiment_acceleration"] = (
                prev_sentiment is not None and payload["net_sentiment"] > prev_sentiment
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
                result["sentiment_acceleration"] = (
                    prev_sentiment is not None and payload["net_sentiment"] > prev_sentiment
                )
                self.sentiment_cache[(ticker, quarter_key)] = payload["net_sentiment"]
                self.precomputed_llm_results[cache_key] = result

            self.llm_cache[cache_key] = result
            return result
        except Exception:
            return None

    # =====================================================================
    # DECISION LOGIC
    # =====================================================================

    def _get_bucket_vol_mult(self, bucket):
        if bucket == "top":
            return self.top_vol_mult
        return self.bottom_vol_mult

    def _get_bucket_park_veto_mult(self, bucket):
        if bucket == "top":
            return self.top_park_veto_mult
        return self.bottom_park_veto_mult

    def _get_bucket_atr_mult(self, bucket):
        if bucket == "top":
            return self.top_atr_mult
        return self.bottom_atr_mult

    def make_decision(self, ticker, date, transcript, portfolio_state, analytics):
        price = analytics.get("close", 0)
        if price <= 0:
            return "HOLD"
        has_pos = ticker in portfolio_state.get("positions", {})

        if self.universe_last_updated_month != date[:7]:
            self._update_universe(date)

        current_bucket = self.universe_buckets.get(ticker)
        atr_14 = analytics.get("atr_14", price * 0.05)

        if ticker in self.sleeve1_positions:
            pos_info = self.sleeve1_positions[ticker]
            pos_info["peak_price"] = max(pos_info["peak_price"], price)
            atr_mult = self._get_bucket_atr_mult(pos_info.get("bucket", current_bucket))
            if price <= pos_info["peak_price"] - (atr_mult * atr_14) or date > pos_info["entry_date"]:
                del self.sleeve1_positions[ticker]
                return "SELL"

        if ticker in self.sleeve2_positions:
            pos_info = self.sleeve2_positions[ticker]
            pos_info["weeks_held"] += 1
            pos_info["peak_price"] = max(pos_info["peak_price"], price)
            if (
                price <= pos_info["peak_price"] - (self.s2_atr_mult * atr_14)
                or pos_info["weeks_held"] >= self.sleeve2_exit_weeks
            ):
                del self.sleeve2_positions[ticker]
                return "SELL"
            return "HOLD"

        in_univ = ticker in self.universe
        if has_pos and not in_univ and ticker not in self.sleeve1_positions and ticker not in self.sleeve2_positions:
            return "SELL"
        if not in_univ or has_pos or current_bucket is None:
            return "HOLD"

        park_vol_10 = analytics.get("park_vol_10")
        park_vol_60 = analytics.get("park_vol_60")
        ma_200 = analytics.get("ma_200")
        park_veto_mult = self._get_bucket_park_veto_mult(current_bucket)

        vol_veto = (
            park_vol_10 is not None
            and park_vol_60 is not None
            and not np.isnan(park_vol_10)
            and not np.isnan(park_vol_60)
            and park_vol_10 > park_veto_mult * park_vol_60
        )
        is_downtrend = ma_200 is not None and price < ma_200

        rsi = analytics.get("rsi_14")
        daily_ret = analytics.get("daily_return")
        vol = analytics.get("volume")
        vol_ma = analytics.get("volume_ma_20")
        vol_mult = self._get_bucket_vol_mult(current_bucket)

        s1_trigger = False
        if rsi is not None and not np.isnan(rsi) and rsi < self.rsi_threshold and not vol_veto:
            s1_trigger = True

        if not s1_trigger and not is_downtrend and not vol_veto:
            if all(value is not None and not np.isnan(value) for value in (daily_ret, vol, vol_ma)):
                if daily_ret > 0 and vol_ma > 0 and vol > vol_mult * vol_ma:
                    s1_trigger = True

        if s1_trigger:
            self.sleeve1_positions[ticker] = {
                "entry_date": date,
                "peak_price": price,
                "bucket": current_bucket,
            }
            return "BUY"

        if transcript is not None and not is_downtrend and not vol_veto:
            sent_res = self.llm_analysis(ticker, transcript, date)
            if sent_res is not None and sent_res.get("sentiment_acceleration", False):
                self.sleeve2_positions[ticker] = {
                    "entry_date": date,
                    "weeks_held": 0,
                    "peak_price": price,
                }
                return "BUY"

        return "HOLD"

    # =====================================================================
    # EVALUATION ORCHESTRATION
    # =====================================================================

    def evaluate(self, verbose=False):
        if self.prices is None:
            raise ValueError("Must call set_data() before evaluate()")

        self._reset_run_state()
        self.weekly_schedule = self._build_weekly_schedule(self.prices)

        print("Running evaluation...")

        with ThreadPoolExecutor(max_workers=2) as executor:
            analytics_future = executor.submit(self.calculate_analytics, self.prices)
            earnings_future = executor.submit(self._build_weekly_earnings_lookup)
            analytics = analytics_future.result()
            weekly_earnings = earnings_future.result()

        weekly_analytics = self._build_weekly_analytics_frame(analytics, self.weekly_schedule)
        self._precompute_monthly_universes(weekly_analytics)

        universe_tickers = set()
        for bucket_map in self.precomputed_universes.values():
            universe_tickers.update(bucket_map)
        print(
            f"  Universe covers {len(universe_tickers)} unique tickers across all months "
            f"(out of {self.prices['ticker'].nunique()} total)"
        )

        if universe_tickers:
            weekly_earnings = {
                key: transcript
                for key, transcript in weekly_earnings.items()
                if key[0] in universe_tickers
            }

        self._precompute_llm_analysis(allowed_tickers=universe_tickers)

        weekly_records, weekly_prices = self._build_weekly_market_views(
            weekly_analytics,
            allowed_tickers=universe_tickers,
        )
        return self._run_fast_backtest(weekly_records, weekly_prices, weekly_earnings, verbose)
