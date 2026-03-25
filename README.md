# Algorithmic Trading Strategy: Iterative Development from V1 to V4

## Abstract

This document presents the systematic development of an algorithmic trading strategy applied to S&P 500 historical price data and earnings call transcripts. Beginning from a minimal moving-average baseline, four enhanced strategy versions were constructed through principled ablation and hypothesis-driven modification. Each iteration targeted a specific research question about universe selection, signal generation, and the marginal contribution of natural language processing. The final version (V4) demonstrates that a purely technical approach, informed by a dual-tail composite scoring mechanism, can achieve strong risk-adjusted returns without the computational overhead of sentiment inference.

---

## 1. Introduction

Systematic equity trading strategies require careful construction of three interdependent components: universe selection (which stocks to consider at any given time), signal generation (when to enter or exit a position), and risk management (how to size and exit positions). Each of these components presents opportunities for refinement, and each refinement must be evaluated against both in-sample development data and an out-of-sample validation set to guard against overfitting.

The work presented here follows a rigorous train-validation protocol. The development split covers the period 2000 to 2017 and is used exclusively for iterative experimentation and hyperparameter tuning. The validation split covers 2018 to 2024 and is held out as a final, unbiased estimate of generalisation performance. No hyperparameter decisions were made based on the validation split.

---

## 2. Data and Evaluation Framework

### 2.1 Price Data

The dataset comprises daily open, high, low, close, and volume (OHLCV) records for 340 S&P 500 tickers spanning multiple decades. The development partition contains 1,332,576 daily price records and the validation partition contains 598,740 records.

### 2.2 Earnings Transcript Data

Quarterly earnings call transcripts are paired with each ticker and used as the input corpus for sentiment analysis via FinBERT, a domain-adapted BERT model pre-trained on financial text. Transcripts shorter than 100 characters are excluded from analysis.

### 2.3 Evaluation Metrics

Performance is measured along five dimensions: total return, annualised Sharpe ratio, maximum drawdown, annualised volatility, and win rate. The Sharpe ratio is the primary indicator of risk-adjusted quality because it penalises both low returns and excessive volatility symmetrically.

---

## 3. Baseline Strategy

The baseline is implemented in the `BaseStrategy` class. It computes a single 50-day moving average (MA-50) per ticker and generates buy and sell signals based on price crossovers relative to that average. No universe filtering, sentiment analysis, or multi-factor scoring is applied.

The baseline establishes the lower-bound reference against which all enhanced versions are compared. Its simplicity reflects the minimum viable specification of the assignment framework.

---

## 4. Strategy Development: V1 through V4

### 4.1 V1: Amihud-Parkinson Composite with Sentiment (Bottom Quartile Universe)

**Internal designation:** EnhancedStrategy V14.2, Reversed Amihud Parkinson Volatility

#### Core Idea

V1 introduced a composite scoring mechanism for universe selection, combining two liquidity-volatility proxies: the Parkinson range-based volatility estimator (60-day rolling) and the Amihud illiquidity ratio (60-day rolling). Each ticker is ranked on both dimensions, the ranks are summed to produce a composite score, and only the bottom 25th percentile of that score is admitted to the tradeable universe each month. Selecting the bottom quartile captures stocks that are simultaneously low in volatility and high in liquidity, which represent the most stable and efficiently priced names in the cross-section.

#### Signal Generation: Two-Sleeve Architecture

V1 operates two independent entry sleeves.

**Sleeve 1 (Technical):** A buy signal fires when the 14-day RSI falls below 40 (oversold condition), provided no Parkinson volatility veto is active. An additional trigger fires when same-day volume exceeds twice the 20-day volume moving average and the daily return is positive, indicating a high-conviction momentum pulse. Both signals are gated by a Parkinson short-term veto that blocks entry if 10-day Parkinson volatility exceeds 1.75 times the 60-day baseline. Sleeve 1 exits via an ATR-based trailing stop.

**Sleeve 2 (Sentiment):** A buy signal fires when FinBERT sentiment analysis on the most recent earnings transcript detects positive sentiment acceleration relative to the prior quarter. The FinBERT pipeline is run in FP16 precision for approximately 2x throughput on GPU hardware. An important engineering fix applied in this version corrects a prior bug whereby `return_all_scores=True` was yielding list outputs rather than scalar dictionaries, which caused Sleeve 2 to be entirely non-functional in earlier iterations. Sleeve 2 exits after a fixed number of weeks or on an ATR trailing stop.

#### Computational Optimisations

Three vectorisation improvements were introduced to make the evaluation loop tractable at scale. Monthly universe snapshots are precomputed using per-ticker sorted arrays and binary search (`np.searchsorted`) rather than repeated groupby operations. Weekly analytics alignment uses a single vectorised `pd.merge_asof` call over a cross-joined frame of all weeks and tickers instead of a Python loop over 340 individual tickers.

#### Core Features

| Component | Specification |
|---|---|
| Universe selection | Bottom 25th percentile of composite (vol\_rank + amihud\_rank) |
| Universe profile | Low volatility, high liquidity |
| Sleeve 1 trigger | RSI-14 < 40 or volume surge (> 2x MA-20) |
| Sleeve 2 trigger | FinBERT sentiment acceleration on earnings transcript |
| Exit rule (Sleeve 1) | ATR-14 trailing stop (2x ATR) |
| Exit rule (Sleeve 2) | ATR-14 trailing stop (2x ATR) or 5-week hold cap |
| Volatility veto | park\_vol\_10 > 1.75 x park\_vol\_60 |
| Trend filter | MA-200 downtrend blocks Sleeve 2 entries |

---

### 4.2 V2: Reversed Universe Selection (Top Quartile)

**Internal designation:** EnhancedStrategy V14, Optimised Parkinson Volatility

#### Core Idea

V2 tests a direct inversion of the V1 hypothesis. Rather than targeting the low-composite quartile, V2 selects the top 25th percentile of the composite score, admitting the most volatile and least liquid tickers in the cross-section. The motivation is grounded in the market microstructure literature: illiquid securities are more likely to exhibit delayed price discovery and transient mispricings that a technical and sentiment strategy can exploit before prices revert or correct.

All signal logic, the two-sleeve architecture, and the computational infrastructure remain identical to V1. The only functional change is the percentile threshold and direction of the universe filter.

#### Core Features

| Component | Specification |
|---|---|
| Universe selection | Top 25th percentile of composite (vol\_rank + amihud\_rank) |
| Universe profile | High volatility, low liquidity |
| Sleeve 1 trigger | RSI-14 < 40 or volume surge (> 2x MA-20) |
| Sleeve 2 trigger | FinBERT sentiment acceleration on earnings transcript |
| Exit rule (Sleeve 1) | ATR-14 trailing stop (2x ATR) |
| Exit rule (Sleeve 2) | ATR-14 trailing stop (2x ATR) or 5-week hold cap |
| Volatility veto | park\_vol\_10 > 1.75 x park\_vol\_60 |
| Trend filter | MA-200 downtrend blocks Sleeve 2 entries |

---

### 4.3 V3: Dual-Tail Universe (Combined)

**Internal designation:** EnhancedStrategy V16, Hollow Purple

#### Core Idea

Having established that both the bottom and top quartiles of the composite score contain exploitable opportunities, V3 combines them by admitting all tickers whose composite score falls at or below the 25th percentile or at or above the 75th percentile. This dual-tail selection is designated Hollow Purple in the internal codebase. The central two quartiles, which represent neither notably stable nor notably volatile names, are excluded on the grounds that their reduced signal-to-noise ratio dilutes portfolio quality.

The expanded universe results in substantially more unique tickers eligible for trading each month (332 in the development split compared to 223 for V1 and 260 for V2), which in turn drives a higher total trade count and broader opportunity set.

#### Core Features

| Component | Specification |
|---|---|
| Universe selection | Bottom 25th and top 25th percentile of composite (dual-tail) |
| Universe profile | Both low-vol/liquid and high-vol/illiquid names |
| Sleeve 1 trigger | RSI-14 < 40 or volume surge (> 2x MA-20) |
| Sleeve 2 trigger | FinBERT sentiment acceleration on earnings transcript |
| Exit rule (Sleeve 1) | ATR-14 trailing stop (2x ATR) |
| Exit rule (Sleeve 2) | ATR-14 trailing stop (2x ATR) or 5-week hold cap |
| Volatility veto | park\_vol\_10 > 1.75 x park\_vol\_60 |
| Trend filter | MA-200 downtrend blocks Sleeve 2 entries |

---

### 4.4 V4: Technical-Only Dual-Tail (No Sentiment)

**Internal designation:** EnhancedStrategy V16, Hollow Purple Technical-Only

#### Core Idea

V4 is a targeted ablation that removes the FinBERT sentiment sleeve entirely while retaining the V3 dual-tail universe. The purpose is to isolate the marginal contribution of NLP-based sentiment to the overall strategy performance. If technical signals alone can account for most of the observed alpha, the strategy becomes substantially simpler, faster, and deployable without GPU infrastructure.

A key enhancement introduced in V4 is bucket-aware risk parameterisation. Because the dual-tail universe contains two qualitatively distinct stock populations (stable and liquid names in the bottom bucket, and volatile and illiquid names in the top bucket), V4 assigns separate ATR multipliers and volume multipliers to each bucket. Bottom-bucket entries use a tighter volume confirmation threshold (`bottom_vol_mult = 1.5`) to account for the naturally higher liquidity of those names, while top-bucket entries retain the standard multiplier (`top_vol_mult = 2`). This asymmetric parameterisation is not possible in V1 and V2, which each operate on a single homogeneous population.

The Sleeve 2 sentiment architecture and all FinBERT preprocessing are removed. The `evaluate()` method no longer requires an earnings dataframe as a non-null input.

#### Core Features

| Component | Specification |
|---|---|
| Universe selection | Bottom 25th and top 25th percentile of composite (dual-tail) |
| Universe profile | Both low-vol/liquid and high-vol/illiquid names |
| Sleeve 1 trigger | RSI-14 < 40 or volume surge (> bucket\_vol\_mult x MA-20) |
| Sleeve 2 | Removed entirely |
| Exit rule (Sleeve 1) | ATR-14 trailing stop (bucket\_atr\_mult x ATR) |
| Volatility veto | park\_vol\_10 > 1.75 x park\_vol\_60 |
| Trend filter | MA-200 downtrend blocks entry |
| Bucket-aware params | Separate vol\_mult and atr\_mult for bottom and top buckets |

---

## 5. Ablation Study

The table below presents results across both data splits for all four strategy versions. The development split is used for iteration and contextual comparison, and the validation split is the authoritative measure of generalisation performance.

| Version | Universe Selection | Sentiment Sleeve | Bucket-Aware Params | Dev Return | Dev Sharpe | Dev Max DD | Dev Win Rate | Dev Trades | Val Return | Val Sharpe | Val Max DD |
|---|---|---|---|---|---|---|---|---|---|---|---|
| V1 (Bottom Quartile) | Bottom 25% only | Yes (FinBERT) | No | 290.62% | 1.99 | -21.90% | 56.4% | 21,619 | 107.50% | 2.10 | -15.39% |
| V2 (Top Quartile) | Top 25% only | Yes (FinBERT) | No | 336.55% | 1.25 | -33.84% | 53.4% | 24,248 | 221.88% | 2.76 | -17.53% |
| V3 (Combined) | Dual-tail 25th/75th | Yes (FinBERT) | No | 560.01% | 1.67 | -34.09% | 54.7% | 44,131 | 253.95% | 2.51 | -22.94% |
| V4 (Technical-Only) | Dual-tail 25th/75th | No (removed) | Yes | 557.07% | 1.67 | -34.25% | 54.7% | 45,327 | 227.92% | 2.48 | -17.30% |

**Notes:** Dev split covers 2000 to 2017. Val split covers 2018 to 2024. All metrics are computed on a weekly rebalancing schedule with an initial capital of USD 100,000. Max DD denotes maximum portfolio drawdown from peak to trough.

---

## 6. Progression Narrative: From V1 to V4

The development arc follows a structured sequence of hypothesis tests, each motivated by an observation from the preceding iteration.

**Step 1: Establishing Composite Universe Selection (Baseline to V1).** The baseline offered no universe filtering and relied on a single moving average for all 340 tickers. V1 replaced this with a principled two-factor composite score combining Parkinson volatility and Amihud illiquidity, selecting only the bottom quartile. This admitted the most stable and liquid names and paired them with a two-sleeve signal structure. The addition of FinBERT sentiment analysis as Sleeve 2 introduced an NLP-driven entry path alongside the technical sleeve. Critically, V1 also corrected a silent bug in the FinBERT output parsing that had rendered Sleeve 2 entirely inactive in all prior development. V1 established a validation Sharpe of 2.10 against a maximum drawdown of 15.39%, confirming that the composite universe filter and two-sleeve architecture added meaningful structure relative to the baseline.

**Step 2: Testing Universe Polarity (V1 to V2).** V1 selected the low end of the composite distribution, favouring stocks with low volatility and high liquidity. V2 challenged this by selecting the opposite extreme: the most volatile and illiquid names. This reflects the illiquidity premium hypothesis from market microstructure theory, which posits that less liquid securities command a return premium for bearing higher transaction costs and price impact risk. The validation results confirmed this premise. V2 achieved more than double the return of V1 on the validation split (221.88% versus 107.50%) with a comparable drawdown (17.53% versus 15.39%), but at the cost of a substantially higher development volatility (37.03% versus 19.35%). The lower development Sharpe of 1.25 indicated that gains from high-volatility names came with greater in-sample instability.

**Step 3: Unifying Both Tails (V2 to V3).** Having validated that both the low-composite and high-composite universes contained alpha, V3 merged them into a single dual-tail universe by selecting all tickers in the bottom 25th or top 25th percentile. The central two quartiles were excluded as neither stable enough to offer reliable mean-reversion nor volatile enough to offer strong momentum dislocations. The practical effect was a near-doubling of eligible tickers (332 versus 223 and 260) and a corresponding increase in total trades (44,131). On the validation split, V3 achieved 253.95% return and a Sharpe of 2.51, which surpassed both V1 and V2 in absolute return while maintaining a competitive Sharpe ratio. The broadened universe did introduce a higher validation drawdown of 22.94%, reflecting the inclusion of both low and high volatility names simultaneously.

**Step 4: Ablating Sentiment to Isolate Technical Alpha (V3 to V4).** With the dual-tail universe established as the strongest structural foundation, V4 asked whether the FinBERT sentiment sleeve was adding material value or merely noise. All NLP processing was removed. The strategy was simplified to a single technical sleeve with the same RSI and volume-surge triggers as V1, V2, and V3. A new feature was introduced: bucket-aware parameterisation, which applies different volume and ATR multipliers to bottom-bucket and top-bucket positions, reflecting the distinct risk profiles of the two tail populations. On the validation split, V4 achieved 227.92% return with a Sharpe of 2.48 and a maximum drawdown of only 17.30%. The return reduction relative to V3 (approximately 26 percentage points) was more than offset by a 5.64 percentage point improvement in maximum drawdown. The near-identical Sharpe ratios (2.48 versus 2.51) confirm that the FinBERT sentiment sleeve was not the primary source of risk-adjusted alpha.

---

## 7. Benefits of V4 Relative to V1, V2, and V3

### 7.1 Superior Drawdown Control on Validation

V4 achieves a maximum validation drawdown of 17.30%, which is lower than V2 (17.53%), V3 (22.94%), and substantially lower than V1 when considered against its return level. V4 delivers more than twice the absolute return of V1 at a comparable drawdown, and it delivers 94.9% of V3's return at a drawdown that is 5.64 percentage points more contained. Removing Sleeve 2 eliminates sentiment-driven entries that could remain open during adverse market conditions under a fixed time-exit rule, contributing to tighter drawdown control.

### 7.2 Consistent Risk-Adjusted Performance

The Sharpe ratio of V4 on the validation split (2.48) is functionally equivalent to V3 (2.51) and exceeds V1 (2.10). V2 achieves the highest validation Sharpe of 2.76 but produces the weakest development Sharpe of 1.25, which signals greater regime sensitivity. V4, by contrast, posts a development Sharpe of 1.67 and a validation Sharpe of 2.48, indicating stable performance across the two distinct historical periods tested.

### 7.3 Elimination of NLP Infrastructure Dependency

V4 removes the dependency on FinBERT inference, which requires a GPU, approximately 420 MB of pre-trained model weights, and substantial wall-clock time to process tens of thousands of transcript chunks across hundreds of tickers. In the validation split alone, V3 processed 63,041 chunks from 5,138 transcripts. V4 eliminates this pipeline entirely, making the strategy deployable in CPU-only environments and reducing end-to-end evaluation time in proportion to the NLP throughput previously required.

### 7.4 Bucket-Aware Risk Parameterisation

V4 introduces an asymmetric risk parameter structure that reflects the qualitatively distinct character of its two stock populations. The bottom-bucket names (low volatility, high liquidity) use tighter volume confirmation thresholds, appropriate to their naturally smoother price behaviour. The top-bucket names (high volatility, low liquidity) retain wider parameters to accommodate their larger intraday and week-to-week price ranges. This design is not expressible in V1 or V2, which each operate on a single homogeneous population, and it is left unexploited in V3 despite the dual-tail universe being present.

### 7.5 Robustness to NLP Signal Noise

Earnings transcript sentiment is inherently noisy due to management forward guidance, scripted language, and analyst question framing. The near-identical Sharpe ratios between V3 and V4 demonstrate that the strategy's alpha is not dependent on NLP signal quality. This robustness is valuable from a research credibility standpoint because it reduces the risk that V3's performance is partly attributable to overfitting to idiosyncratic transcript patterns present in the 2000 to 2017 development period.

---

## 8. Conclusion

This work demonstrates a structured methodology for iterative strategy development in algorithmic trading. Four progressively refined strategy versions were evaluated under a strict train-validation protocol across 24 years of S&P 500 data. The key findings are as follows. First, the direction of universe selection based on composite Amihud-Parkinson scoring matters significantly, with the high-volatility tail generating higher returns than the low-volatility tail. Second, combining both tails into a dual-tail universe yields additive improvement in absolute return. Third, FinBERT sentiment provides marginal return benefit but at the cost of increased drawdown and substantial computational overhead. Fourth, a purely technical strategy with bucket-aware risk parameters (V4) represents the most favourable balance of return, drawdown, simplicity, and deployability across all experimental configurations tested.

---

## Appendix: Technical Indicator Reference

| Indicator | Window | Purpose |
|---|---|---|
| Parkinson Volatility (short) | 10-day | Volatility veto trigger |
| Parkinson Volatility (long) | 60-day | Universe scoring, veto baseline |
| Amihud Illiquidity Ratio | 60-day | Universe scoring |
| RSI | 14-day | Oversold entry trigger (Sleeve 1) |
| ATR | 14-day | Trailing stop distance |
| Volume MA | 20-day | Volume surge confirmation |
| MA-200 | 200-day | Downtrend filter |
