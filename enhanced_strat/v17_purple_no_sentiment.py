"""
EnhancedStrategy V16: Hollow Purple (Technical-Only)
====================================================
Based on V16 universe selection, with the sentiment sleeve removed.

Key changes:
  - removes all sentiment preprocessing and FinBERT usage
  - keeps both the bottom 25% and top 25% of the composite score
  - supports separate ATR and volume multipliers for each quartile bucket

New optional parameters:
  - bottom_vol_mult, top_vol_mult
  - bottom_atr_mult, top_atr_mult

Legacy parameters `vol_mult` and `s1_atr_mult` are still accepted and used as
fallbacks for both buckets to preserve notebook compatibility.

Notebook globals required: BaseStrategy, STARTING_CASH
"""

import numpy as np
import pandas as pd


class EnhancedStrategy(BaseStrategy):

    def __init__(
        self,
        finbert_pipeline=None,
        rsi_threshold=40,
        vol_mult=2,
        s1_atr_mult=2,
        # s2_atr_mult=2,
        # sleeve2_exit_weeks=5,
        # max_transcript_chars=4000,
        park_veto_mult=1.75,
        bottom_vol_mult=1.5,
        top_vol_mult=2,
        bottom_atr_mult=2,
        top_atr_mult=2,
    ):
        # Keep the original constructor shape for notebook compatibility.
        super().__init__(finbert_pipeline)
        self.rsi_threshold = rsi_threshold
        self.vol_mult = vol_mult
        self.s1_atr_mult = s1_atr_mult
        # self.s2_atr_mult = s2_atr_mult
        # self.sleeve2_exit_weeks = sleeve2_exit_weeks
        # self.max_transcript_chars = max_transcript_chars
        self.park_veto_mult = park_veto_mult

        self.bottom_vol_mult = vol_mult if bottom_vol_mult is None else bottom_vol_mult
        self.top_vol_mult = vol_mult if top_vol_mult is None else top_vol_mult
        self.bottom_atr_mult = s1_atr_mult if bottom_atr_mult is None else bottom_atr_mult
        self.top_atr_mult = s1_atr_mult if top_atr_mult is None else top_atr_mult

        self.universe = set()
        self.universe_buckets = {}
        self.universe_last_updated_month = None
        self.universe_analytics_df = None
        self.precomputed_universes = {}

        self.sleeve1_positions = {}
        self.sleeve2_positions = {}
        self.weekly_schedule = []

    def _reset_run_state(self):
        self.universe = set()
        self.universe_buckets = {}
        self.universe_last_updated_month = None
        self.universe_analytics_df = None
        self.precomputed_universes = {}
        self.sleeve1_positions = {}
        self.sleeve2_positions = {}
        self.weekly_schedule = []

        if hasattr(self, "llm_cache"):
            self.llm_cache = {}
        if hasattr(self, "llm_cache_hits"):
            self.llm_cache_hits = 0
        if hasattr(self, "llm_cache_misses"):
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

    def _run_fast_backtest(self, weekly_records, weekly_prices, verbose=False):
        print("Running fast backtest loop...")
        portfolio = {"cash": STARTING_CASH, "positions": {}, "trades": []}
        portfolio_history = []
        get_week_prices = weekly_prices.get
        get_week_records = weekly_records.get
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
                decision = make_decision(ticker, week_date, None, portfolio_state, analytics)
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
    def _build_universe_buckets(tickers, composite_scores):
        if len(tickers) == 0:
            return {}

        lower_threshold = np.quantile(composite_scores, 0.25)
        upper_threshold = np.quantile(composite_scores, 0.75)

        bucket_map = {}
        for ticker, score in zip(tickers, composite_scores):
            if score <= lower_threshold:
                bucket_map[ticker] = "bottom"
            elif score >= upper_threshold:
                bucket_map[ticker] = "top"

        return bucket_map

    def _select_universe(self, analytics_slice):
        latest = analytics_slice.groupby("ticker").last().reset_index()
        scored = latest.dropna(subset=["park_vol_60", "amihud_60"]).copy()
        if scored.empty:
            return {}

        scored["vol_rank"] = scored["park_vol_60"].rank(ascending=True, method="average")
        scored["amihud_rank"] = scored["amihud_60"].rank(ascending=False, method="average")
        scored["composite_score"] = scored["vol_rank"] + scored["amihud_rank"]
        return self._build_universe_buckets(
            scored["ticker"].tolist(),
            scored["composite_score"].to_numpy(dtype=float),
        )

    def _select_universe_from_latest(self, latest_dict):
        if not latest_dict:
            return {}

        tickers = list(latest_dict.keys())
        park60s = np.array([v[0] for v in latest_dict.values()], dtype=float)
        amihud60s = np.array([v[1] for v in latest_dict.values()], dtype=float)

        valid = ~(np.isnan(park60s) | np.isnan(amihud60s))
        if not valid.any():
            return {}

        valid_tickers = [ticker for ticker, is_valid in zip(tickers, valid) if is_valid]
        park_valid = park60s[valid]
        amihud_valid = amihud60s[valid]

        vol_rank = pd.Series(park_valid).rank(ascending=True, method="average").values
        amihud_rank = pd.Series(amihud_valid).rank(ascending=False, method="average").values
        composite = vol_rank + amihud_rank
        return self._build_universe_buckets(valid_tickers, composite)

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
    # DECISION LOGIC
    # =====================================================================

    def _get_bucket_vol_mult(self, bucket):
        if bucket == "top":
            return self.top_vol_mult
        return self.bottom_vol_mult

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

        in_univ = ticker in self.universe
        if has_pos and not in_univ and ticker not in self.sleeve1_positions:
            return "SELL"
        if not in_univ or has_pos or current_bucket is None:
            return "HOLD"

        park_vol_10 = analytics.get("park_vol_10")
        park_vol_60 = analytics.get("park_vol_60")
        ma_200 = analytics.get("ma_200")

        vol_veto = (
            park_vol_10 is not None
            and park_vol_60 is not None
            and not np.isnan(park_vol_10)
            and not np.isnan(park_vol_60)
            and park_vol_10 > self.park_veto_mult * park_vol_60
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
            if all(v is not None and not np.isnan(v) for v in (daily_ret, vol, vol_ma)):
                if daily_ret > 0 and vol_ma > 0 and vol > vol_mult * vol_ma:
                    s1_trigger = True

        if s1_trigger:
            self.sleeve1_positions[ticker] = {
                "entry_date": date,
                "peak_price": price,
                "bucket": current_bucket,
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
        analytics = self.calculate_analytics(self.prices)
        weekly_analytics = self._build_weekly_analytics_frame(analytics, self.weekly_schedule)
        self._precompute_monthly_universes(weekly_analytics)

        universe_tickers = set()
        for bucket_map in self.precomputed_universes.values():
            universe_tickers.update(bucket_map)
        print(
            f"  Universe covers {len(universe_tickers)} unique tickers across all months "
            f"(out of {self.prices['ticker'].nunique()} total)"
        )

        weekly_records, weekly_prices = self._build_weekly_market_views(
            weekly_analytics,
            allowed_tickers=universe_tickers,
        )
        return self._run_fast_backtest(weekly_records, weekly_prices, verbose)
