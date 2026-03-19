# ============================================================
# V2: Louvain Cluster Strategy (Trailing SL)
# Differences from V1:
#   - EXIT: trailing stop ratchets up each week using current ATR
#   - ENTRY: sentiment mult for high confidence raised 1.5 → 2
# ============================================================
# Depends on notebook globals: BaseStrategy, TradingSimulation, STARTING_CASH
# ============================================================

import re
import numpy as np
import pandas as pd
import networkx as nx
from collections import defaultdict
from statsmodels.stats.diagnostic import acorr_ljungbox
from tqdm import tqdm

from enhanced_helpers.lookups import build_fast_lookup, get_latest
from enhanced_helpers.sentiment import build_sentiment_lookup, get_latest_sentiment


class EnhancedStrategy(BaseStrategy):

    def __init__(self, finbert_pipeline=None):
        super().__init__(finbert_pipeline)
        self.entry_dates = {}                    # {ticker: {date, price, low, atr, ema, tp, sl}}
        self.inefficient_tickers = []
        self.efficient_tickers = []
        self.weekly_clusters = {}                # {date_str: {cid: [tickers]}}
        self.weekly_leaders = {}                 # {date_str: {cid: leader_ticker}}
        self.high_corr_flag = {}                 # {date_str: bool}
        self.avg_volume = {}                     # {ticker: float}
        self.sentiment_cache = {}                # {(ticker, date_str): float}
        self._sentiment_by_ticker = defaultdict(list)

    # ==============================================================
    # 1. DATA CLEANING
    # ==============================================================
    def clean_data(self, prices_df, earnings_df):
        prices = prices_df.copy().drop_duplicates()
        earnings = earnings_df.copy().drop_duplicates()
        prices['date'] = pd.to_datetime(prices['date'])
        earnings['date'] = pd.to_datetime(earnings['date'])
        earnings = earnings[earnings['transcript'].str.len() > 100]

        # Drop tickers with no earnings calls
        tickers_with_earnings = set(earnings['ticker'].unique())
        prices = prices[prices['ticker'].isin(tickers_with_earnings)]

        # Forward-fill close prices per ticker
        prices = prices.sort_values(['ticker', 'date'])
        prices['close'] = prices.groupby('ticker')['close'].ffill()

        return prices, earnings

    # ==============================================================
    # 2. ANALYTICS — Ljung-Box, Louvain, Correlation Flag, SAV, ATR, EMA
    # ==============================================================
    def calculate_analytics(self, prices_df):
        print("Computing enhanced analytics...")
        prices = prices_df.copy().sort_values(['ticker', 'date'])

        # ---- vectorised log returns ----
        prices['log_return'] = prices.groupby('ticker')['close'].transform(
            lambda x: np.log(x / x.shift(1))
        )

        # ---- LIST 1: Ljung-Box → inefficient / efficient tickers ----
        print("  [1/4] Ljung-Box test for market efficiency...")
        lb_pvals = {}
        for ticker, group in prices.groupby('ticker')['log_return']:
            lr = group.dropna()
            if len(lr) < 50:
                continue
            try:
                res = acorr_ljungbox(lr, lags=[20], return_df=True)
                lb_pvals[ticker] = res['lb_pvalue'].iloc[0]
            except Exception:
                continue

        alpha = 0.05
        self.inefficient_tickers = [t for t, p in lb_pvals.items() if p < alpha]
        self.efficient_tickers   = [t for t, p in lb_pvals.items() if p >= alpha]
        print(f"    Inefficient: {len(self.inefficient_tickers)}, "
              f"Efficient: {len(self.efficient_tickers)}")

        # ---- average volume for leader identification ----
        self.avg_volume = (
            prices[prices['ticker'].isin(self.inefficient_tickers)]
            .groupby('ticker')['volume'].mean().to_dict()
        )

        # ---- LIST 2 & 3 + BOOLEAN: weekly Louvain clusters,
        #      leaders, efficient-ticker correlation flag ----
        print("  [2/4] Weekly Louvain clustering & correlation regime flag...")
        returns_pivot = prices.pivot_table(
            index='date', columns='ticker', values='log_return'
        )
        ineff_cols = [t for t in self.inefficient_tickers if t in returns_pivot.columns]
        eff_cols   = [t for t in self.efficient_tickers   if t in returns_pivot.columns]
        ineff_ret  = returns_pivot[ineff_cols]
        eff_ret    = returns_pivot[eff_cols]

        weekly_dates = pd.date_range(
            prices['date'].min(), prices['date'].max(), freq='W-FRI'
        )
        CORR_WINDOW    = 60
        EDGE_THRESHOLD = 0.6

        for wd in tqdm(weekly_dates, desc="    Weeks"):
            ws = wd.strftime('%Y-%m-%d')

            # --- inefficient tickers → Louvain ---
            window = ineff_ret[ineff_ret.index <= wd].tail(CORR_WINDOW)
            valid  = window.columns[window.notna().sum() >= 20]

            if len(valid) < 3:
                self.weekly_clusters[ws] = {}
                self.weekly_leaders[ws]  = {}
                self.high_corr_flag[ws]  = False
                continue

            corr_m = window[valid].corr()
            tlist  = corr_m.columns.tolist()
            cvals  = corr_m.values
            ri, ci = np.triu_indices(len(tlist), k=1)
            rhos   = cvals[ri, ci]
            emask  = rhos >= EDGE_THRESHOLD

            G = nx.Graph()
            G.add_nodes_from(tlist)
            G.add_edges_from([
                (tlist[r], tlist[c], {'weight': float(rho)})
                for r, c, rho in zip(ri[emask], ci[emask], rhos[emask])
            ])

            iso = [n for n in G.nodes() if G.degree(n) == 0]
            Gc  = G.copy()
            Gc.remove_nodes_from(iso)

            if Gc.number_of_nodes() < 2:
                self.weekly_clusters[ws] = {}
                self.weekly_leaders[ws]  = {}
            else:
                comms = nx.community.louvain_communities(
                    Gc, weight='weight', seed=42
                )
                clusters, leaders = {}, {}
                for idx, comm in enumerate(sorted(comms, key=len, reverse=True)):
                    members = sorted(comm)
                    clusters[idx] = members
                    leaders[idx]  = max(
                        members, key=lambda t: self.avg_volume.get(t, 0)
                    )
                self.weekly_clusters[ws] = clusters
                self.weekly_leaders[ws]  = leaders

            # --- efficient tickers → correlation flag ---
            ew = eff_ret[eff_ret.index <= wd].tail(CORR_WINDOW)
            ve = ew.columns[ew.notna().sum() >= 20]
            if len(ve) < 2:
                self.high_corr_flag[ws] = False
            else:
                ec     = ew[ve].corr().values
                r2, c2 = np.triu_indices(len(ve), k=1)
                total  = len(r2)
                high   = (ec[r2, c2] > 0.85).sum()
                self.high_corr_flag[ws] = (high / total) >= 0.50

        # ---- LIST 4 & 5: per-ticker SAV, ATR, EMA (vectorised) ----
        print("  [3/4] SAV, ATR, EMA for inefficient tickers...")
        ip = prices[prices['ticker'].isin(self.inefficient_tickers)].copy()

        # 20-day rolling SAV
        ip['vol_med'] = ip.groupby('ticker')['volume'].transform(
            lambda x: x.rolling(20, min_periods=10).median()
        )
        ip['vol_std'] = ip.groupby('ticker')['volume'].transform(
            lambda x: x.rolling(20, min_periods=10).std()
        )
        ip['sav'] = (ip['volume'] - ip['vol_med']) / ip['vol_std']

        # 14-day ATR
        ip['prev_close'] = ip.groupby('ticker')['close'].shift(1)
        ip['tr'] = np.maximum(
            ip['high'] - ip['low'],
            np.maximum(
                np.abs(ip['high'] - ip['prev_close']),
                np.abs(ip['low']  - ip['prev_close'])
            )
        )
        ip['atr_14'] = ip.groupby('ticker')['tr'].transform(
            lambda x: x.rolling(14, min_periods=7).mean()
        )

        # 20-day EMA
        ip['ema_20'] = ip.groupby('ticker')['close'].transform(
            lambda x: x.ewm(span=20, adjust=False).mean()
        )

        result_df = ip[['ticker', 'date', 'close', 'low', 'high',
                         'volume', 'sav', 'atr_14', 'ema_20']].copy()
        result_df['date'] = result_df['date'].dt.strftime('%Y-%m-%d')

        print(f"  [4/4] Analytics ready: {len(result_df):,} rows, "
              f"{result_df['ticker'].nunique()} tickers")
        return result_df

    # ==============================================================
    # 3. LLM ANALYSIS — split transcript, FinBERT confidence score
    # ==============================================================
    def llm_analysis(self, ticker, transcript, date):
        if transcript is None or self.finbert_pipeline is None:
            return None

        date_str = (date if isinstance(date, str)
                    else pd.Timestamp(date).strftime('%Y-%m-%d'))
        cache_key = (ticker, date_str)
        if cache_key in self.sentiment_cache:
            return {'confidence': self.sentiment_cache[cache_key]}

        pattern = re.compile(
            r'(question[- ]and[- ]answer session|q&a session|q n a)',
            re.IGNORECASE
        )
        match = pattern.search(transcript)

        if match:
            prepared = transcript[:match.start()][-2000:]
            qa       = transcript[match.start():][:2000]
        else:
            prepared = transcript[:2000]
            qa       = transcript[-2000:]

        try:
            res_p = self.finbert_pipeline(prepared, top_k=None)
            if isinstance(res_p[0], list):
                res_p = res_p[0]
            probs_p    = {r['label']: r['score'] for r in res_p}
            prep_score = probs_p.get('positive', 0) - probs_p.get('negative', 0)

            res_q = self.finbert_pipeline(qa, top_k=None)
            if isinstance(res_q[0], list):
                res_q = res_q[0]
            probs_q  = {r['label']: r['score'] for r in res_q}
            qa_score = probs_q.get('positive', 0) - probs_q.get('negative', 0)

            confidence = (prep_score + qa_score) / 2.0
        except Exception:
            confidence = 0.0

        self.sentiment_cache[cache_key] = confidence
        return {'confidence': confidence}

    # ==============================================================
    # 4. MAKE DECISION (stub — logic lives in evaluate)
    # ==============================================================
    def make_decision(self, ticker, date, transcript, portfolio_state, analytics):
        return 'HOLD'

    # ==============================================================
    # 5. EVALUATE — custom backtest loop with trailing stop
    # ==============================================================
    def evaluate(self, verbose=False):
        if self.prices is None or self.earnings is None:
            raise ValueError("Must call set_data() before evaluate()")

        print("\n=== Enhanced Strategy Evaluation ===")

        # ---- step 1: compute all analytics ----
        analytics_df = self.calculate_analytics(self.prices)
        lookup       = build_fast_lookup(analytics_df)

        # ---- step 2: pre-compute leader sentiments ----
        print("Pre-computing leader sentiments...")
        all_leaders = set()
        for leaders_dict in self.weekly_leaders.values():
            all_leaders.update(leaders_dict.values())

        leader_earn = (
            self.earnings[self.earnings['ticker'].isin(all_leaders)]
            .sort_values(['ticker', 'date'])
        )
        tickers_arr     = leader_earn['ticker'].values
        transcripts_arr = leader_earn['transcript'].tolist()
        dates_arr       = leader_earn['date'].values

        for i in tqdm(range(len(leader_earn)), desc="  Sentiments"):
            self.llm_analysis(
                tickers_arr[i],
                transcripts_arr[i],
                pd.Timestamp(dates_arr[i]).strftime('%Y-%m-%d')
            )
        self._sentiment_by_ticker = build_sentiment_lookup(self.sentiment_cache)

        # ---- step 3: simulation loop ----
        print("Running backtest...")
        sim = TradingSimulation(self.prices, self.earnings, STARTING_CASH)
        self.entry_dates = {}
        portfolio_history = []
        BASE_TARGET = 5000

        for i, week_date in enumerate(sim.weekly_schedule):
            if verbose and i % 20 == 0:
                print(f"  Week {i+1}/{len(sim.weekly_schedule)}: {week_date}")

            current_prices = sim._get_current_prices(week_date)
            clusters  = self.weekly_clusters.get(week_date, {})
            leaders   = self.weekly_leaders.get(week_date, {})
            high_corr = self.high_corr_flag.get(week_date, False)

            # ==================== EXITS ====================
            for ticker in list(self.entry_dates.keys()):
                # Position already closed externally
                if ticker not in sim.portfolio.positions:
                    del self.entry_dates[ticker]
                    continue

                price = sim._get_price_on_date(ticker, week_date)
                if price is None:
                    continue

                entry = self.entry_dates[ticker]

                # Update trailing stop: ratchet upward using current ATR
                ta_now = get_latest(ticker, week_date, lookup)
                if ta_now is not None:
                    cur_atr = ta_now.get('atr_14', entry['atr'])
                    if cur_atr and not np.isnan(cur_atr) and cur_atr > 0:
                        new_sl = price - 1.5 * cur_atr
                        if new_sl > entry['sl']:
                            entry['sl'] = new_sl

                # Take-profit (1:3 R:R)
                if price >= entry['tp']:
                    sim.portfolio.sell(ticker, price, week_date)
                    del self.entry_dates[ticker]
                    continue

                # Trailing stop-loss
                if price <= entry['sl']:
                    sim.portfolio.sell(ticker, price, week_date)
                    del self.entry_dates[ticker]
                    continue

                # EMA exit — close falls below EMA at date of entry
                if price < entry['ema']:
                    sim.portfolio.sell(ticker, price, week_date)
                    del self.entry_dates[ticker]
                    continue

                # High-correlation regime — exit all positions
                if high_corr:
                    sim.portfolio.sell(ticker, price, week_date)
                    del self.entry_dates[ticker]
                    continue

            # ==================== ENTRIES ====================
            for cid, leader in leaders.items():
                la = get_latest(leader, week_date, lookup)
                if la is None:
                    continue

                l_close = la.get('close', 0)
                l_ema   = la.get('ema_20', 0)
                l_sav   = la.get('sav', 0)

                if l_ema <= 0:
                    continue
                if isinstance(l_sav, float) and np.isnan(l_sav):
                    continue

                # Entry signal: leader close > EMA  AND  SAV > 2
                if l_close > l_ema and l_sav > 2:

                    # --- sentiment-based bet sizing ---
                    mult = 1.0
                    sent = get_latest_sentiment(self._sentiment_by_ticker, leader, week_date)
                    if sent is not None:
                        if sent > 0.5:
                            mult = 2
                        elif sent < 0.3:
                            mult = 0.5

                    target = BASE_TARGET * mult

                    # Buy every ticker in the cluster
                    for ticker in clusters.get(cid, []):
                        if ticker in sim.portfolio.positions:
                            continue

                        price = sim._get_price_on_date(ticker, week_date)
                        if price is None or price <= 0:
                            continue

                        ta = get_latest(ticker, week_date, lookup)
                        if ta is None:
                            continue

                        atr = ta.get('atr_14', 0)
                        low = ta.get('low', price)
                        ema = ta.get('ema_20', price)

                        if not atr or np.isnan(atr) or atr <= 0:
                            continue

                        # Stop-loss: entry-day low − 1.5 × ATR
                        sl   = low - 1.5 * atr
                        # Risk per share
                        risk = price - sl
                        if risk <= 0:
                            continue
                        # Take-profit: 1:3 risk-reward
                        tp = price + 3.0 * risk

                        sim.portfolio.buy_target(
                            ticker, price, week_date, target_value=target
                        )

                        # Record entry only if the buy actually went through
                        if ticker in sim.portfolio.positions:
                            self.entry_dates[ticker] = {
                                'date':  week_date,
                                'price': price,
                                'low':   low,
                                'atr':   atr,
                                'ema':   ema,
                                'tp':    tp,
                                'sl':    sl,
                            }

            # ---- record portfolio snapshot ----
            portfolio_history.append({
                'date':            week_date,
                'portfolio_value': sim.portfolio.get_value(current_prices),
                'cash':            sim.portfolio.cash,
                'positions':       len(sim.portfolio.positions),
            })

        # ---- return results in the expected format ----
        final_date   = sim.weekly_schedule[-1]
        final_prices = sim._get_current_prices(final_date)
        return {
            'trades':            sim.portfolio.trades,
            'portfolio_history': portfolio_history,
            'final_portfolio':   sim.portfolio.get_state(final_prices),
            'final_prices':      final_prices,
        }
