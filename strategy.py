"""
strategy.py — EnhancedStrategy v13
===================================
Base: v11.3 (all logic preserved exactly)
Key change: FinBERT replaced with inline LM + custom earnings scorer
  - LM (Loughran-McDonald) word lists bundled inline — no external packages
  - Custom earnings-call phrase vocabulary supplements LM
  - Combined score = 0.5 * LM_polarity + 0.5 * earnings_phrase_polarity
  - Same sentiment_acceleration logic as v11.3

Execution-engine safe:
  - No __file__ references
  - No ProcessPoolExecutor  (ThreadPoolExecutor only)
  - No subprocess / HTTP requests
  - No pysentiment2 or other non-standard libraries
  - Single file, no relative imports

Standard library + (pandas, numpy, torch, transformers) only.

Interface contract (unchanged):
  - Class: EnhancedStrategy
  - Methods: set_data(prices_df, earnings_df), evaluate(verbose=False)
  - evaluate() returns: {trades, portfolio_history, final_portfolio, final_prices}
"""

import re
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict

import numpy as np
import pandas as pd

# ─── Constants ────────────────────────────────────────────────────────────────
STARTING_CASH = 100_000

SENTENCE_SPLIT_RE = re.compile(r'(?<=[.!?])\s+')

# ─── Loughran-McDonald word lists (inline, no external package) ───────────────
# Source: Loughran & McDonald (2011) Master Dictionary, public domain
_LM_POSITIVE = frozenset("""
ABLE ABUNDANCE ABUNDANT ACCLAIMED ACCOMPLISH ACCOMPLISHED ACCOMPLISHES
ACCOMPLISHING ACCOMPLISHMENT ACCOMPLISHMENTS ACHIEVE ACHIEVED ACHIEVEMENT
ACHIEVEMENTS ACHIEVES ACHIEVING ADEQUATELY ADVANCEMENT ADVANCEMENTS ADVANCES
ADVANCING ADVANTAGE ADVANTAGED ADVANTAGEOUS ADVANTAGEOUSLY ADVANTAGES ALLIANCE
ALLIANCES ASSURE ASSURED ASSURES ASSURING ATTAIN ATTAINED ATTAINING ATTAINMENT
ATTAINMENTS ATTAINS ATTRACTIVE ATTRACTIVENESS BEAUTIFUL BEAUTIFULLY BENEFICIAL
BENEFICIALLY BENEFIT BENEFITED BENEFITING BENEFITTED BENEFITTING BEST BETTER
BOLSTERED BOLSTERING BOLSTERS BOOM BOOMING BOOST BOOSTED BREAKTHROUGH
BREAKTHROUGHS BRILLIANT CHARITABLE COLLABORATE COLLABORATED COLLABORATES
COLLABORATING COLLABORATION COLLABORATIONS COLLABORATIVE COLLABORATOR
COLLABORATORS COMPLIMENT COMPLIMENTARY COMPLIMENTED COMPLIMENTING COMPLIMENTS
CONCLUSIVE CONCLUSIVELY CONDUCIVE CONFIDENT CONSTRUCTIVE CONSTRUCTIVELY
COURTEOUS CREATIVE CREATIVELY CREATIVENESS CREATIVITY DELIGHT DELIGHTED
DELIGHTFUL DELIGHTFULLY DELIGHTING DELIGHTS DEPENDABILITY DEPENDABLE DESIRABLE
DESIRED DESPITE DESTINED DILIGENT DILIGENTLY DISTINCTION DISTINCTIONS DISTINCTIVE
DISTINCTIVELY DISTINCTIVENESS DREAM EASIER EASILY EASY EFFECTIVE EFFICIENCIES
EFFICIENCY EFFICIENT EFFICIENTLY EMPOWER EMPOWERED EMPOWERING EMPOWERS ENABLE
ENABLED ENABLES ENABLING ENCOURAGED ENCOURAGEMENT ENCOURAGES ENCOURAGING ENHANCE
ENHANCED ENHANCEMENT ENHANCEMENTS ENHANCES ENHANCING ENJOY ENJOYABLE ENJOYABLY
ENJOYED ENJOYING ENJOYMENT ENJOYS ENTHUSIASM ENTHUSIASTIC ENTHUSIASTICALLY
EXCELLENCE EXCELLENT EXCELLING EXCELS EXCEPTIONAL EXCEPTIONALLY EXCITED
EXCITEMENT EXCITING EXCLUSIVE EXCLUSIVELY EXCLUSIVENESS EXCLUSIVES EXCLUSIVITY
EXEMPLARY FANTASTIC FAVORABLE FAVORABLY FAVORED FAVORING FAVORITE FAVORITES
FRIENDLY GAIN GAINED GAINING GAINS GOOD GREAT GREATER GREATEST GREATLY GREATNESS
HAPPIEST HAPPILY HAPPINESS HAPPY HIGHEST HONOR HONORABLE HONORED HONORING HONORS
IDEAL IMPRESS IMPRESSED IMPRESSES IMPRESSING IMPRESSIVE IMPRESSIVELY IMPROVE
IMPROVED IMPROVEMENT IMPROVEMENTS IMPROVES IMPROVING INCREDIBLE INCREDIBLY
INFLUENTIAL INFORMATIVE INGENUITY INNOVATE INNOVATED INNOVATES INNOVATING
INNOVATION INNOVATIONS INNOVATIVE INNOVATIVENESS INNOVATOR INNOVATORS INSIGHTFUL
INSPIRATION INSPIRATIONAL INTEGRITY INVENT INVENTED INVENTING INVENTION
INVENTIONS INVENTIVE INVENTIVENESS INVENTOR INVENTORS LEADERSHIP LEADING LOYAL
LUCRATIVE MERITORIOUS OPPORTUNITIES OPPORTUNITY OPTIMISTIC OUTPERFORM
OUTPERFORMED OUTPERFORMING OUTPERFORMS PERFECT PERFECTED PERFECTLY PERFECTS
PLEASANT PLEASANTLY PLEASED PLEASURE PLENTIFUL POPULAR POPULARITY POSITIVE
POSITIVELY PREEMINENCE PREEMINENT PREMIER PREMIERE PRESTIGE PRESTIGIOUS
PROACTIVE PROACTIVELY PROFICIENCY PROFICIENT PROFICIENTLY PROFITABILITY
PROFITABLE PROFITABLY PROGRESS PROGRESSED PROGRESSES PROGRESSING PROSPERED
PROSPERING PROSPERITY PROSPEROUS PROSPERS REBOUND REBOUNDED REBOUNDING
RECEPTIVE REGAIN REGAINED REGAINING RESOLVE REVOLUTIONIZE REVOLUTIONIZED
REVOLUTIONIZES REVOLUTIONIZING REWARD REWARDED REWARDING REWARDS SATISFACTION
SATISFACTORILY SATISFACTORY SATISFIED SATISFIES SATISFY SATISFYING SMOOTH
SMOOTHING SMOOTHLY SMOOTHS SOLVES SOLVING SPECTACULAR SPECTACULARLY STABILITY
STABILIZATION STABILIZATIONS STABILIZE STABILIZED STABILIZES STABILIZING STABLE
STRENGTH STRENGTHEN STRENGTHENED STRENGTHENING STRENGTHENS STRENGTHS STRONG
STRONGER STRONGEST SUCCEED SUCCEEDED SUCCEEDING SUCCEEDS SUCCESS SUCCESSES
SUCCESSFUL SUCCESSFULLY SUPERIOR SURPASS SURPASSED SURPASSES SURPASSING
TRANSPARENCY TREMENDOUS TREMENDOUSLY UNMATCHED UNPARALLELED UNSURPASSED UPTURN
UPTURNS VALUABLE VERSATILE VERSATILITY VIBRANCY VIBRANT WIN WINNER WINNERS
WINNING WORTHY
""".split())

_LM_NEGATIVE = frozenset("""
ABANDON ABANDONED ABANDONING ABANDONMENT ABANDONMENTS ABANDONS ABDICATED
ABDICATES ABDICATING ABDICATION ABDICATIONS ABERRANT ABERRATION ABERRATIONS
ABETTING ABNORMAL ABNORMALITIES ABNORMALITY ABNORMALLY ABOLISH ABOLISHES
ABOLISHING ABOLISHMENT ABRUPTLY ABSENCE ABSENT ABSENTEEISM ABUSE ABUSED ABUSES
ABUSING ABUSIVE ABUSIVELY ACCIDENT ACCIDENTAL ACCIDENTS ACCUSATION ACCUSATIONS
ACCUSE ACCUSED ACCUSES ACCUSING ACRIMONY ADAMANTLY ADDICTION ADDICTIONS
INADEQUATE INADEQUATELY INADEQUACY ADVERSE ADVERSELY ADVERSITIES ADVERSITY
AFFLICT AFFLICTION AFFLICTIONS AFRAID AGGRAVATE AGGRAVATED AGGRAVATING
AGGRAVATION AGGRESSIVE ALLEGATION ALLEGATIONS ALLEGE ALLEGED ALLEGEDLY ALLEGES
ALLEGING AMBIGUITY AMBIGUOUS AMBIGUOUSLY ANOMALOUS ANOMALOUSLY ANOMALY
APPREHENSION APPREHENSIVE ARBITRARY ARREAR ARREARS ASTONISHMENT ATTRITION
BANKRUPT BANKRUPTCIES BANKRUPTCY BARRIER BARRIERS BEARISH BELOW BLOW BREACH
BREACHED BREACHES BREACHING BREAKDOWN BREAKDOWNS BRIBERY BURDEN BURDENSOME
CANCEL CANCELLED CANCELLATION CATASTROPHE CATASTROPHIC CEASE CEASES CENSURE
CLAIMS COLLAPSE COLLATERAL COMPLICATION COMPLICATIONS CONCERN CONCERNED
CONCERNS CONDEMN CONDEMNED CONFLICT CONFLICTS CONFISCATE CONFUSION CONSTRAINED
CONSTRAINTS CONTAMINATED CONTAMINATION CONTESTED CONTROVERSIAL CONTROVERSIALLY
CONTROVERSY CORRUPTION CRASH CRISES CRISIS CRITIC CRITICAL CRITICALLY CRITICISM
CURTAIL CURTAILED CURTAILMENT DAMAGE DAMAGED DAMAGES DANGEROUS DANGEROUSLY
DECLINE DECLINED DECLINES DECLINING DECREMENT DEFAULTS DEFAULT DEFERRAL
DEFICIENCIES DEFICIENCY DEFICIENT DEFICIT DEFICITS DELAY DELAYED DELAYS
DELETERIOUS DEPENDENT DEPRESSED DEPRESSION DEPRIVATION DERELICTION DESTROY
DETERIORATE DETERIORATED DETERIORATION DETRIMENTAL DEVIATE DEVIATION
DIFFICULTIES DIFFICULTY DIMINISH DIMINISHED DIMINISHING DISAPPOINTING
DISAPPOINTS DISAPPOINTED DISAPPOINTMENT DISCONTINUE DISCONTINUED DISPUTES
DISRUPTION DISRUPTIONS DISRUPTIVE DOWN DOWNTURN DOWNWARD DRAWBACK DRAWBACKS
DOUBT DOUBTFUL DOWNGRADE DOWNGRADED DOWNGRADE DROPPED DROPPING DROPS DUBIOUS
EMBEZZLEMENT ERRONEOUS ERRONEOUSLY ERRORS ERROR EXCESSIVE FAILED FAILING
FAILURE FAILURES FALLEN FALLING FALLS FAULT FAULTS FAULTY FEDERAL FINE FINES
FORCED FRAUD FRAUDULENT FRAUDULENTLY FREEZE FREEZING FRUSTRATED FRUSTRATING
FRUSTRATION GRAVE GRIEVANCE GRIEVANCES GUILTY HALT HALTED HARM HARMFUL
HARSHLY HARSH HARSH HAZARD HAZARDOUS HEADWINDS HINDER HINDRANCE ILLEGAL
ILLEGALLY ILLEGALITY IMPAIR IMPAIRMENT IMPAIRMENTS IMPAIRED IMPROPER
IMPROPERLY INABILITY INADEQUATE INCAPABLE INCIDENT INCIDENTS INCOMPETENT
INDEBTED INDEBTEDNESS INFERIOR INFRINGEMENT INJURY INSUFFICIENT
INSUFFICIENTLY IRREGULARITIES IRREGULARITY ISSUES JEOPARDIZE JEOPARDIZED
LAWSUIT LAWSUITS LAYOFFS LIABILITIES LIABILITY LIQUIDATION LITIGATION
LITIGATIONS LOSS LOSSES LOWERED MANIPULATE MANIPULATED MANIPULATION
MISCONDUCT MISREPRESENTATION MISSTATED MISSTATEMENT MISUSE MONEY NEGATIVE
NEGLECT NONCOMPLIANCE OBSTACLE OBSTACLES OBSOLETE OFFENSES OPPOSE OPPOSED
OPPOSITION OUTSOURCING OVERDUE OVERLOOK OVERSTATEMENT OVERSTATED PENALTY
PENALTIES POOR POORLY PROBE PROBLEM PROBLEMS PROCEEDED PROTEST RECALL
RECESSION REDUCED REDUCTION REDUNDANCIES REDUNDANCY REFUSAL REFUSE REFUSED
RESTATEMENT RESTRUCTURE RESTRUCTURING REVENUE RISK RISKS RISKY SANCTION
SANCTIONS SCANDAL SHORTFALL SHORTFALLS SHORTAGE SIGNIFICANT SLOW SLOWING
SLOWS SUBOPTIMAL SUSPECT SUSPENDED SUSPENSION TERMINATE TERMINATED TERMINATION
TROUBLED UNCERTAIN UNCERTAINTIES UNCERTAINTY UNETHICAL UNFAVORABLE UNFAVORABLY
UNFORESEEN UNFORTUNATE UNFORTUNATE UNJUST UNNECESSARY UNRELIABLE UNSUITABLE
UNSTABLE VIOLATION VIOLATIONS VOLATILITY VULNERABLE WARN WARNING WARNINGS
WEAK WEAKENED WEAKENING WEAKENS WEAKNESS WEAKNESSES WORRIES WORRY WORRYING
WORSE WORSEN WORSENED WORSENING WORSENS WORST WORTHLESS WRITEDOWN WRITEDOWNS
WRITEOFF WRITEOFFS WRONG WRONGDOING WRONGDOINGS WRONGFUL WRONGFULLY WRONGLY
""".split())

# ─── Custom earnings-call vocabulary (supplements LM) ────────────────────────
# Phrases more likely to signal earnings beats/misses that LM misses
_EARNINGS_POS_WORDS = frozenset("""
BEAT BEATING BEATS EXCEEDED EXCEEDING EXCEEDS SURPASSED SURPASSING SURPASSES
RECORD RECORDS RAISED RAISING RAISES UPGRADE UPGRADED UPGRADES UPGRADING
OUTPERFORMED OUTPERFORMING OUTPERFORMS AHEAD ACCELERATED ACCELERATING
EXPANDED EXPANDING EXPANDS EXPANSION GREW GROWING INCREASED INCREASING
INCREASES PROFITABLE PROFITABLY MOMENTUM ROBUST SOLID ENCOURAGING CONFIDENT
DELIVERED DELIVERING DELIVERS AHEAD FAVORABLE STRONG STRENGTH BETTER BEST
ABOVE EXCEPTIONAL EXCELLENT IMPROVED IMPROVING IMPROVEMENTS
""".split())

_EARNINGS_NEG_WORDS = frozenset("""
MISSED MISSING MISSES DISAPPOINTING DISAPPOINTED DISAPPOINTS BELOW LOWERED
LOWERING LOWERS DOWNGRADE DOWNGRADED DOWNGRADES DOWNGRADING SHORTFALL
SHORTFALLS CHALLENGING CHALLENGE CHALLENGES HEADWIND HEADWINDS PRESSURE
PRESSURES PRESSURED PRESSURING DECLINED DECLINING DECLINES DETERIORATED
DETERIORATING IMPAIRMENT IMPAIRMENTS WRITEDOWN WRITEDOWNS WRITEOFF WRITEOFFS
RESTRUCTURING RESTRUCTURED CHARGED CHARGES UNCERTAIN UNCERTAINTY RISKS
RISKY CONCERN CONCERNED CONCERNS DIFFICULT DIFFICULTIES WEAK WEAKENED
WEAKENING MISS WORSE WORSENED WORSENING REVISION REVISED LOWERED SLOWING
""".split())


def _score_transcript(text: str):
    """
    Combined LM + earnings phrase sentiment scorer.
    Returns (combined_polarity, earnings_phrase_polarity):
      combined  = 0.5 * LM_polarity + 0.5 * earnings_phrase_polarity
      ep_pol    = raw earnings phrase polarity (used as quality gate)
    Both in [-1, 1].
    """
    if not text or len(str(text).strip()) < 20:
        return 0.0, 0.0

    # Tokenise: uppercase words only
    words = re.findall(r'[A-Z]{2,}', str(text).upper())
    if not words:
        return 0.0, 0.0

    # ── LM component ──────────────────────────────────────────────────────────
    lm_pos = sum(1 for w in words if w in _LM_POSITIVE)
    lm_neg = sum(1 for w in words if w in _LM_NEGATIVE)
    lm_total = lm_pos + lm_neg
    lm_pol = (lm_pos - lm_neg) / lm_total if lm_total > 0 else 0.0

    # ── Earnings phrase component ─────────────────────────────────────────────
    ep_pos = sum(1 for w in words if w in _EARNINGS_POS_WORDS)
    ep_neg = sum(1 for w in words if w in _EARNINGS_NEG_WORDS)
    ep_total = ep_pos + ep_neg
    ep_pol = (ep_pos - ep_neg) / ep_total if ep_total > 0 else 0.0

    # ── Combine ────────────────────────────────────────────────────────────────
    if lm_total > 0 and ep_total > 0:
        combined = 0.5 * lm_pol + 0.5 * ep_pol
    elif lm_total > 0:
        combined = lm_pol
    elif ep_total > 0:
        combined = ep_pol
    else:
        combined = 0.0

    return float(combined), float(ep_pol)


# ─── Base class ───────────────────────────────────────────────────────────────
class BaseStrategy:
    def __init__(self, finbert_pipeline=None):
        self.finbert_pipeline = finbert_pipeline  # kept for interface compat; unused
        self.llm_cache = {}
        self.llm_cache_hits = 0
        self.llm_cache_misses = 0
        self.prices = None
        self.earnings = None

    def set_data(self, prices_df, earnings_df):
        print("Cleaning and preprocessing data...")
        self.prices, self.earnings = self.clean_data(prices_df, earnings_df)
        print(f"  Data ready: {len(self.prices):,} price records, {len(self.earnings):,} earnings records")

    def clean_data(self, prices_df, earnings_df):
        prices = prices_df.copy().drop_duplicates()
        earnings = earnings_df.copy().drop_duplicates()
        prices['date'] = pd.to_datetime(prices['date'])
        earnings['date'] = pd.to_datetime(earnings['date'])
        earnings = earnings[earnings['transcript'].fillna('').str.len() > 100]
        prices = prices.sort_values(['ticker', 'date']).reset_index(drop=True)
        earnings = earnings.sort_values(['ticker', 'date']).reset_index(drop=True)
        return prices, earnings


# ─── Enhanced Strategy ────────────────────────────────────────────────────────
class EnhancedStrategy(BaseStrategy):

    def __init__(self, finbert_pipeline=None, rsi_threshold=40, vol_mult=2,
                 s1_atr_mult=2, s2_atr_mult=2, sleeve2_exit_weeks=5,
                 max_transcript_chars=2000):
        super().__init__(finbert_pipeline)
        self.rsi_threshold       = rsi_threshold
        self.vol_mult            = vol_mult
        self.s1_atr_mult         = s1_atr_mult
        self.s2_atr_mult         = s2_atr_mult
        self.sleeve2_exit_weeks  = sleeve2_exit_weeks
        self.max_transcript_chars = max_transcript_chars

        self.universe = set()
        self.universe_last_updated_month = None
        self.universe_analytics_df = None
        self.precomputed_universes = {}

        self.sleeve1_positions = {}
        self.sleeve2_positions = {}
        self.sentiment_cache   = {}

        self.sentiment_batch_size    = 128
        self.sentiment_chunk_workers = 4
        self.precomputed_llm_results = {}
        self.weekly_schedule = []

        # dynamic position sizing signal (set in make_decision, read in backtest)
        self._pending_buy_sizes = {}   # {ticker: target_value}

        # Market breadth: fraction of universe above MA200 per week
        self._market_breadth = {}      # {week_date: float 0-1}

    # =========================================================================
    # DATA PREPARATION
    # =========================================================================

    def clean_data(self, prices_df, earnings_df):
        prices = prices_df.copy().drop_duplicates()
        earnings = earnings_df.copy().drop_duplicates()
        prices['date'] = pd.to_datetime(prices['date'])
        earnings['date'] = pd.to_datetime(earnings['date'])
        earnings = earnings[earnings['transcript'].fillna('').str.len() > 100]
        prices = prices.sort_values(['ticker', 'date']).reset_index(drop=True)
        earnings = earnings.sort_values(['ticker', 'date']).reset_index(drop=True)
        return prices, earnings

    # =========================================================================
    # ANALYTICS CALCULATION
    # =========================================================================

    def calculate_analytics(self, prices_df):
        print("  Computing analytics (vectorised)...")
        df = prices_df.copy().sort_values(['ticker', 'date'])
        ticker_index = df['ticker']
        grouped = df.groupby('ticker', sort=False)
        annualisation = np.sqrt(252.0)

        daily_return = grouped['close'].pct_change()
        close_rolling = df['close'].groupby(ticker_index, sort=False)
        volume_rolling = df['volume'].groupby(ticker_index, sort=False)

        df['daily_return'] = daily_return
        df['ma_200'] = close_rolling.rolling(200, min_periods=50).mean().reset_index(level=0, drop=True)
        df['ma_50']  = close_rolling.rolling(50,  min_periods=20).mean().reset_index(level=0, drop=True)

        # 3-month (~63 trading days) price momentum
        df['return_63d'] = grouped['close'].pct_change(periods=63)

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
            'ma_200', 'ma_50', 'return_63d', 'atr_14'
        ]].copy()
        result_df['date'] = result_df['date'].dt.strftime('%Y-%m-%d')
        self.universe_analytics_df = result_df
        print(f"  Analytics: {len(result_df):,} rows, {result_df['ticker'].nunique()} tickers")
        return result_df

    @staticmethod
    def _get_quarter_key(date_str):
        dt = pd.to_datetime(date_str)
        return f"{dt.year}-Q{(dt.month - 1) // 3 + 1}"

    @staticmethod
    def _build_analytics_lookup_vectorised(analytics_df):
        lookup = {}
        sorted_df = analytics_df.sort_values(['ticker', 'date'])
        for ticker, group in sorted_df.groupby('ticker', sort=False):
            dates = group['date'].tolist()
            records = group.to_dict('records')
            lookup[ticker] = list(zip(dates, records))
        return lookup

    def _build_weekly_schedule(self, prices_df):
        min_date = pd.to_datetime(prices_df['date']).min()
        max_date = pd.to_datetime(prices_df['date']).max()
        return pd.date_range(start=min_date, end=max_date, freq='W-FRI').strftime('%Y-%m-%d').tolist()

    def _build_weekly_analytics_lookup(self, analytics_df, weekly_schedule):
        print("  Aligning analytics to weekly schedule...")
        if analytics_df.empty or not weekly_schedule:
            return {}
        sorted_df = analytics_df.copy()
        sorted_df['date'] = pd.to_datetime(sorted_df['date'])
        sorted_df = sorted_df.sort_values('date').reset_index(drop=True)
        week_dates = pd.to_datetime(pd.Index(weekly_schedule))
        all_tickers = sorted_df['ticker'].unique()
        weeks_expanded = (
            pd.DataFrame({'date': week_dates})
            .merge(pd.DataFrame({'ticker': all_tickers}), how='cross')
            .sort_values('date')
            .reset_index(drop=True)
        )
        weekly_df = pd.merge_asof(
            weeks_expanded, sorted_df, on='date', by='ticker', direction='backward'
        ).dropna(subset=['close'])
        weekly_df['date'] = weekly_df['date'].dt.strftime('%Y-%m-%d')
        print(f"  Weekly analytics: {len(weekly_df):,} rows")
        return self._build_analytics_lookup_vectorised(weekly_df)

    def _build_weekly_market_views(self, analytics_lookup):
        weekly_records = defaultdict(list)
        weekly_prices  = defaultdict(dict)
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
            p = current_prices.get(ticker)
            if p is not None:
                total += pos['shares'] * p
        return total

    def _get_portfolio_state(self, portfolio, current_prices):
        return {
            'cash': portfolio['cash'],
            'positions': {
                t: {'shares': p['shares'], 'buy_price': p['buy_price']}
                for t, p in portfolio['positions'].items()
            },
            'total_value': self._get_portfolio_value(portfolio, current_prices)
        }

    def _run_fast_backtest(self, weekly_records, weekly_prices, weekly_earnings, verbose=False):
        print("  Running backtest...")
        portfolio = {'cash': STARTING_CASH, 'positions': {}, 'trades': []}
        portfolio_history = []

        for i, week_date in enumerate(self.weekly_schedule):
            if verbose and i % 10 == 0:
                print(f"    Week {i+1}/{len(self.weekly_schedule)}: {week_date}")

            current_prices  = weekly_prices.get(week_date, {})
            portfolio_state = self._get_portfolio_state(portfolio, current_prices)

            for ticker, analytics in weekly_records.get(week_date, []):
                transcript = weekly_earnings.get((ticker, week_date))
                decision = self.make_decision(ticker, week_date, transcript, portfolio_state, analytics)
                price = analytics.get('close')
                if price is None or np.isnan(price) or price <= 0:
                    continue
                if decision == 'BUY':
                    target = self._pending_buy_sizes.pop(ticker, 5000)
                    self._buy_target(portfolio, ticker, price, week_date, target_value=target)
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

    # =========================================================================
    # UNIVERSE SELECTION
    # =========================================================================

    def _select_universe(self, analytics_slice):
        latest = analytics_slice.groupby('ticker').last().reset_index()
        scored = latest.dropna(subset=['vol_60', 'amihud_60']).copy()
        if scored.empty:
            return set()
        scored['vol_rank']    = scored['vol_60'].rank(ascending=True,  method='average')
        scored['amihud_rank'] = scored['amihud_60'].rank(ascending=False, method='average')
        scored['composite_score'] = scored['vol_rank'] + scored['amihud_rank']
        threshold = scored['composite_score'].quantile(0.80)
        return set(scored[scored['composite_score'] >= threshold]['ticker'].tolist())

    def _select_universe_from_latest(self, latest_dict):
        if not latest_dict:
            return set()
        tickers   = list(latest_dict.keys())
        vol60s    = np.array([v[0] for v in latest_dict.values()], dtype=float)
        amihud60s = np.array([v[1] for v in latest_dict.values()], dtype=float)
        valid = ~(np.isnan(vol60s) | np.isnan(amihud60s))
        if not valid.any():
            return set()
        valid_tickers = [t for t, v in zip(tickers, valid) if v]
        vol_valid    = vol60s[valid]
        amihud_valid = amihud60s[valid]
        vol_rank    = pd.Series(vol_valid).rank(ascending=True,  method='average').values
        amihud_rank = pd.Series(amihud_valid).rank(ascending=False, method='average').values
        composite   = vol_rank + amihud_rank
        threshold   = np.quantile(composite, 0.80)
        return {t for t, s in zip(valid_tickers, composite) if s >= threshold}

    def _precompute_monthly_universes(self, weekly_schedule):
        self.precomputed_universes = {}
        if self.universe_analytics_df is None or not weekly_schedule:
            return
        print("  Precomputing monthly universes...")
        df = self.universe_analytics_df[['ticker', 'date', 'vol_60', 'amihud_60']]
        ticker_data = {}
        for ticker, grp in df.groupby('ticker', sort=False):
            grp = grp.sort_values('date')
            ticker_data[ticker] = {
                'dates':    grp['date'].values,
                'vol60':    grp['vol_60'].values,
                'amihud60': grp['amihud_60'].values,
            }
        first_week_by_month = {}
        for week_date in weekly_schedule:
            month_key = week_date[:7]
            if month_key not in first_week_by_month:
                first_week_by_month[month_key] = week_date
        for month_key, cutoff_date in sorted(first_week_by_month.items()):
            latest = {}
            for ticker, td in ticker_data.items():
                i = int(np.searchsorted(td['dates'], cutoff_date, side='right')) - 1
                if i >= 0:
                    latest[ticker] = (td['vol60'][i], td['amihud60'][i])
            if latest:
                self.precomputed_universes[month_key] = self._select_universe_from_latest(latest)
        print(f"  Universe snapshots: {len(self.precomputed_universes)} months")

    def _update_universe(self, current_date):
        month_key = current_date[:7]
        if month_key in self.precomputed_universes:
            self.universe = self.precomputed_universes[month_key]
            self.universe_last_updated_month = month_key
            return
        if self.universe_analytics_df is None:
            return
        analytics_slice = self.universe_analytics_df[
            self.universe_analytics_df['date'] <= current_date
        ]
        if analytics_slice.empty:
            return
        self.universe = self._select_universe(analytics_slice)
        self.universe_last_updated_month = month_key

    # =========================================================================
    # SENTIMENT (inline LM + custom earnings vocabulary)
    # =========================================================================

    @staticmethod
    def _extract_finbert_score(result):
        """Parse FinBERT pipeline output → net_sentiment in [-1, 1]."""
        if isinstance(result, list):
            scores = {r['label'].lower(): r.get('score', 0.0) for r in result}
            pos = scores.get('positive', 0.0)
            neg = scores.get('negative', 0.0)
            neu = scores.get('neutral', 0.0)
        elif isinstance(result, dict):
            label = result.get('label', 'neutral').lower()
            score = result.get('score', 1.0)
            if 'positive' in label:
                pos, neg, neu = score, 0.0, 1.0 - score
            elif 'negative' in label:
                pos, neg, neu = 0.0, score, 1.0 - score
            else:
                pos, neg, neu = 0.0, 0.0, 1.0
        else:
            return 0.0
        total = pos + neg + neu
        return (pos - neg) / total if total > 0 else 0.0

    def _precompute_llm_analysis(self, allowed_tickers=None):
        """
        Two-pass sentiment scoring:
        Pass 1 (instant): LM+earnings dictionary on all transcripts.
        Pass 2 (fast):    FinBERT on top 35% by ep_pol (first 300 chars only).
        """
        self.precomputed_llm_results = {}
        if self.earnings is None or self.earnings.empty:
            return

        earnings_df = self.earnings[['ticker', 'date', 'transcript']].copy()
        if allowed_tickers:
            earnings_df = earnings_df[earnings_df['ticker'].isin(allowed_tickers)]
            print(f"  Filtering to {len(allowed_tickers)} universe tickers "
                  f"({len(earnings_df):,} transcripts)")
        if earnings_df.empty:
            return
        earnings_df['date'] = pd.to_datetime(earnings_df['date']).dt.strftime('%Y-%m-%d')

        # --- Pass 1: fast LM scoring ---
        print("  Pass 1: inline LM+earnings scoring...")
        count = 0
        for row in earnings_df.itertuples(index=False):
            cache_key = f"{row.ticker}_{row.date}"
            if cache_key in self.precomputed_llm_results:
                continue
            net_sentiment, ep_pol = _score_transcript(row.transcript)
            dominant = ('positive' if net_sentiment > 0
                        else 'negative' if net_sentiment < 0
                        else 'neutral')
            self.precomputed_llm_results[cache_key] = {
                'net_sentiment':  net_sentiment,
                'ep_pol':         ep_pol,
                'sentiment':      dominant,
                'positive_ratio': max(0.0, net_sentiment),
                'chunks_processed': 1,
            }
            count += 1
        print(f"  LM done: {count:,} transcripts")

        # --- Pass 2: FinBERT on top candidates (fast — 300 chars, no chunking) ---
        if self.finbert_pipeline is None:
            return

        # Select top 35% by ep_pol where LM score is positive
        scored = sorted(
            ((k, v['ep_pol']) for k, v in self.precomputed_llm_results.items() if v['ep_pol'] > 0.0),
            key=lambda x: x[1], reverse=True
        )
        n_finbert = max(1, int(len(scored) * 0.35))
        finbert_keys = {k for k, _ in scored[:n_finbert]}

        # Build (cache_key → short_text) for FinBERT
        key_to_text = {}
        for row in earnings_df.itertuples(index=False):
            ck = f"{row.ticker}_{row.date}"
            if ck in finbert_keys:
                key_to_text[ck] = str(row.transcript)[:300]

        if not key_to_text:
            return

        items = list(key_to_text.items())
        print(f"  Pass 2: FinBERT on {len(items):,} transcripts (300 chars each)...")
        fb_scores = {}
        batch_sz = self.sentiment_batch_size
        try:
            for i in range(0, len(items), batch_sz):
                batch = items[i:i + batch_sz]
                texts = [t for _, t in batch]
                keys  = [k for k, _ in batch]
                results = self.finbert_pipeline(
                    texts, batch_size=batch_sz, truncation=True, max_length=128
                )
                if isinstance(results, dict):
                    results = [results]
                for key, res in zip(keys, results):
                    fb_scores[key] = self._extract_finbert_score(res)
        except Exception as e:
            print(f"  FinBERT inference error: {e} — using LM fallback")
            return

        # Update net_sentiment with FinBERT score for selected transcripts
        for key, fb_score in fb_scores.items():
            if key in self.precomputed_llm_results:
                self.precomputed_llm_results[key]['net_sentiment'] = fb_score
                dominant = ('positive' if fb_score > 0 else 'negative' if fb_score < 0 else 'neutral')
                self.precomputed_llm_results[key]['sentiment'] = dominant
                self.precomputed_llm_results[key]['positive_ratio'] = max(0.0, fb_score)

        print(f"  FinBERT upgraded {len(fb_scores):,} transcripts")

    def llm_analysis(self, ticker, transcript, date):
        if transcript is None:
            return None

        cache_key = f"{ticker}_{date}"

        if cache_key in self.llm_cache:
            self.llm_cache_hits += 1
            return self.llm_cache[cache_key]

        self.llm_cache_misses += 1
        try:
            base_result = self.precomputed_llm_results.get(cache_key)
            if base_result is None:
                net_sentiment, ep_pol = _score_transcript(transcript)
                dominant = ('positive' if net_sentiment > 0
                            else 'negative' if net_sentiment < 0
                            else 'neutral')
                base_result = {
                    'net_sentiment':     net_sentiment,
                    'ep_pol':            ep_pol,
                    'sentiment':         dominant,
                    'positive_ratio':    max(0.0, net_sentiment),
                    'chunks_processed':  1,
                }
                self.precomputed_llm_results[cache_key] = base_result

            quarter_key   = self._get_quarter_key(date)
            net_sentiment = base_result['net_sentiment']
            self.sentiment_cache[(ticker, quarter_key)] = net_sentiment

            dt       = pd.to_datetime(date)
            prev_dt  = dt - pd.DateOffset(months=3)
            prev_key = f"{prev_dt.year}-Q{(prev_dt.month - 1) // 3 + 1}"
            prev_sentiment = self.sentiment_cache.get((ticker, prev_key))
            accel = (net_sentiment > prev_sentiment) if prev_sentiment is not None else False

            result = dict(base_result)
            result['sentiment_acceleration'] = accel
            self.llm_cache[cache_key] = result
            return result
        except Exception:
            return None

    # =========================================================================
    # DECISION LOGIC (identical to v11.3)
    # =========================================================================

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
            s1_mult = pos_info.get('atr_mult', self.s1_atr_mult)
            if price <= pos_info['peak_price'] - (s1_mult * atr_14) or date > pos_info['entry_date']:
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

        vol_10     = analytics.get('vol_10')
        vol_60     = analytics.get('vol_60')
        ma_200     = analytics.get('ma_200')

        vol_veto = (
            vol_10 is not None and vol_60 is not None
            and not np.isnan(vol_10) and not np.isnan(vol_60)
            and vol_10 > 1.5 * vol_60
        )
        is_downtrend = ma_200 is not None and price < ma_200

        # Market breadth regime: true bear market only (<30% of universe above MA200)
        bear_market = self._market_breadth.get(date, 0.5) < 0.30

        rsi       = analytics.get('rsi_14')
        daily_ret = analytics.get('daily_return')
        vol       = analytics.get('volume')
        vol_ma    = analytics.get('volume_ma_20')

        s1_trigger = False
        # RSI oversold: veto during bear markets (>60% of universe in downtrend)
        if (rsi is not None and not np.isnan(rsi) and rsi < self.rsi_threshold
                and not vol_veto and not bear_market):
            s1_trigger = True

        # Volume breakout: uptrend only (unchanged)
        if not s1_trigger and not is_downtrend and not vol_veto:
            if all(v is not None and not np.isnan(v) for v in (daily_ret, vol, vol_ma)):
                if daily_ret > 0 and vol_ma > 0 and vol > self.vol_mult * vol_ma:
                    s1_trigger = True

        if s1_trigger:
            target_val = 5000 if not is_downtrend else 3000
            self.sleeve1_positions[ticker] = {
                'entry_date': date, 'peak_price': price, 'atr_mult': self.s1_atr_mult
            }
            self._pending_buy_sizes[ticker] = target_val
            return 'BUY'

        if transcript is not None and not is_downtrend and not vol_veto:
            sent_res = self.llm_analysis(ticker, transcript, date)
            if sent_res is not None:
                net_s = sent_res.get('net_sentiment', 0.0)
                accel = sent_res.get('sentiment_acceleration', False)
                prev_quarter = self._get_prev_sentiment(ticker, date)
                if accel and net_s > 0.0 and (prev_quarter is None or net_s > prev_quarter + 0.05):
                    self.sleeve2_positions[ticker] = {
                        'entry_date': date, 'weeks_held': 0, 'peak_price': price
                    }
                    self._pending_buy_sizes[ticker] = 7500
                    return 'BUY'

        return 'HOLD'

    def _get_prev_sentiment(self, ticker, date):
        dt       = pd.to_datetime(date)
        prev_dt  = dt - pd.DateOffset(months=3)
        prev_key = f"{prev_dt.year}-Q{(prev_dt.month - 1) // 3 + 1}"
        return self.sentiment_cache.get((ticker, prev_key))

    # =========================================================================
    # EVALUATION ORCHESTRATION
    # =========================================================================

    def evaluate(self, verbose=False):
        if self.prices is None or self.earnings is None:
            raise ValueError("Must call set_data() before evaluate()")

        _pre_injected_analytics = self.universe_analytics_df  # preserve cache injection
        self.sleeve1_positions, self.sleeve2_positions, self.universe = {}, {}, set()
        self.universe_last_updated_month = None
        self.universe_analytics_df = _pre_injected_analytics  # restore (None if not injected)
        self.precomputed_universes = {}
        self.precomputed_llm_results = {}
        self._pending_buy_sizes = {}
        self._market_breadth = {}
        self.weekly_schedule = self._build_weekly_schedule(self.prices)

        print("Running evaluation...")

        if self.universe_analytics_df is not None:
            print("  Using pre-injected analytics (cache hit)")
            analytics = self.universe_analytics_df
            weekly_earnings = self._build_weekly_earnings_lookup()
        else:
            with ThreadPoolExecutor(max_workers=2) as executor:
                af = executor.submit(self.calculate_analytics, self.prices)
                ef = executor.submit(self._build_weekly_earnings_lookup)
                analytics       = af.result()
                weekly_earnings = ef.result()

        with ThreadPoolExecutor(max_workers=2) as executor:
            uf  = executor.submit(self._precompute_monthly_universes, self.weekly_schedule)
            alf = executor.submit(self._build_weekly_analytics_lookup, analytics, self.weekly_schedule)
            uf.result()
            analytics_lookup = alf.result()

        universe_tickers = set()
        for tickers in self.precomputed_universes.values():
            universe_tickers.update(tickers)
        print(f"  Universe: {len(universe_tickers)} unique tickers")

        # inline sentiment — fast (no model)
        self._precompute_llm_analysis(allowed_tickers=universe_tickers)

        weekly_records, weekly_prices = self._build_weekly_market_views(analytics_lookup)

        # Precompute market breadth: fraction of universe tickers above MA200 per week
        print("  Computing market breadth...")
        for week_date, records in weekly_records.items():
            total = len(records)
            if total > 0:
                above = sum(
                    1 for _, a in records
                    if a.get('ma_200') is not None and not np.isnan(a.get('ma_200', float('nan')))
                    and a.get('close', 0) > a['ma_200']
                )
                self._market_breadth[week_date] = above / total
            else:
                self._market_breadth[week_date] = 0.5

        return self._run_fast_backtest(weekly_records, weekly_prices, weekly_earnings, verbose)
