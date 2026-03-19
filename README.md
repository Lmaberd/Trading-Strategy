# EnhancedStrategy Guide

This README explains the last `EnhancedStrategy` defined in [trading_assignment.ipynb](c:\Users\Sebert\Desktop\SIT Lambert\Y2T2\INF2006 - Cloud Computing and Big Data\Project\Algorithmic Trading\trading_assignment.ipynb).

It is a two-sleeve strategy built on top of `BaseStrategy`:

- `Sleeve 1`: short-term technical entries
- `Sleeve 2`: earnings-transcript sentiment acceleration entries
- Shared filters: universe selection, volatility veto, and trend filter

## High-Level Flow

For each ticker and date, the strategy follows this order:

1. Reject invalid prices.
2. Update the tradable universe if a new month has started.
3. Check open-position exits first.
4. If the stock is not tradable, stop.
5. Check technical entry conditions for `Sleeve 1`.
6. If that fails, check transcript-based entry conditions for `Sleeve 2`.
7. Otherwise return `HOLD`.

This means exits always have priority over new entries.

## Data Cleaning

Before trading starts, the strategy:

- removes duplicate price rows
- removes duplicate earnings rows
- converts both `date` columns to pandas datetimes
- removes earnings transcripts with length `<= 100`
- sorts prices by `ticker` and `date`

The inherited `set_data()` method from `BaseStrategy` stores the cleaned price and earnings data on the strategy instance.

## Indicators Calculated

The strategy precomputes the following analytics for every ticker:

- `daily_return`: percentage change in close price
- `ma_200`: 200-day moving average, with `min_periods=50`
- `ma_50`: 50-day moving average, with `min_periods=1`
- `vol_60`: annualized 60-day rolling standard deviation of daily returns
- `vol_10`: annualized 10-day rolling standard deviation of daily returns
- `amihud_60`: 60-day average of `abs(return) / dollar_volume`
- `atr_14`: 14-day average true range
- `rsi_14`: 14-day RSI
- `volume_ma_20`: 20-day moving average of volume, with `min_periods=10`

Notes:

- `ATR` uses `high`, `low`, and previous close when `high` and `low` exist.
- If `high` and `low` are missing, ATR falls back to `abs(close - prev_close)`.
- `ma_50` is calculated but is not currently used by the final decision logic.

## Hypotheses tested
1. Hypothesis 1 (Low Volatility Anomaly) & Hypothesis 5 (Illiquidity Premium)

Where it is: _update_universe() function.

How it works: The code ranks all stocks by their 60-day Volatility (ascending) and 60-day Amihud ratio (descending). It combines these into a composite_score and strictly limits your tradable universe to the top tier of assets. This structurally forces you into stocks that benefit from both anomalies simultaneously.

2. Hypothesis 2 (Volume-Confirmed Momentum)

Where it is: make_decision() function (Sleeve 1, Trigger B).

How it works: if daily_ret > 0 and vol_ma > 0 and vol > self.vol_mult * vol_ma: captures the exact momentum breakout you validated.

3. Hypothesis 3 (RSI Mean Reversion)

Where it is: make_decision() function (Sleeve 1, Trigger A).

How it works: if rsi < self.rsi_threshold: executes the oversold snap-back trades.

4. Hypothesis 4 (Sentiment Acceleration)

Where it is: llm_analysis() function & make_decision() (Sleeve 2).

How it works: The code compares the current FinBERT net sentiment to the prev_sentiment stored in the cache. If it is mathematically higher, it triggers the 'sentiment_acceleration' flag and authorizes the fundamental trade.

5. Hypothesis 5 (Volatility Acceleration)

Where it is: make_decision() function (The vol_veto).

How it works: vol_10 > 1.5 * vol_60. Before any trade is authorized, the system checks this. If true, it vetoes the trade completely to prevent you from catching an idiosyncratic falling knife.

6. Hypothesis 6 (200-day SMA Trend Filter)

Where it is: make_decision() function (The is_downtrend flag).

How it works: price < ma_200. Momentum and Earnings trades are strictly blocked if the asset is below its 200-day moving average, shielding you from structural bear markets.


## Universe Selection

The strategy does not allow every stock to trade every day. It first builds a monthly universe.

### When the universe is updated

The universe is refreshed when `make_decision()` sees a date from a new month:

- current month key = `date[:7]`
- if that month differs from `self.universe_last_updated_month`, the universe is rebuilt

### How stocks are scored

For each ticker, it takes the latest available analytics up to the current date and drops rows missing either:

- `vol_60`
- `amihud_60`

It then ranks stocks as follows:

- `vol_rank`: `vol_60.rank(ascending=True)`
- `amihud_rank`: `amihud_60.rank(ascending=False)`
- `composite_score = vol_rank + amihud_rank`

### Which stocks are kept

The code keeps:

- tickers with `composite_score >= composite_score.quantile(0.80)`

Important implementation note:

- the comment says "Expanded to Top 40%", but the code actually uses the `0.80` quantile cutoff
- lower ranks are assigned to lower volatility and higher Amihud values, but the code keeps the highest composite scores
- this README describes the code exactly as implemented

## Shared Trading Filters

These filters affect whether a stock can enter a position.

### 1. Price validity filter

If `close <= 0`, the strategy immediately returns `HOLD`.

### 2. Universe filter

If the ticker is not in the current monthly universe:

- a new position cannot be opened
- the strategy returns `HOLD` unless a special cleanup sell is triggered for an already-held position

### 3. Volatility acceleration veto

This is the main short-term risk veto:

- `vol_veto = vol_10 > 1.5 * vol_60`

If true:

- `Sleeve 1` mean-reversion entry is blocked
- `Sleeve 1` momentum entry is blocked
- `Sleeve 2` transcript entry is blocked

### 4. Trend filter

The strategy checks whether the stock is below its long-term trend:

- `is_downtrend = price < ma_200`

Effect:

- `Sleeve 1` mean-reversion is still allowed in a downtrend
- `Sleeve 1` momentum is blocked in a downtrend
- `Sleeve 2` transcript entry is blocked in a downtrend

### 5. Existing position filter

If the ticker is already present in `portfolio_state['positions']`:

- the strategy will not open another new position
- it will either manage the existing sleeve-specific exit logic or return `HOLD`

## Entry Logic

The strategy has two entry sleeves. `Sleeve 1` is checked first. If `Sleeve 1` triggers, the strategy returns `BUY` immediately and does not evaluate `Sleeve 2`.

## Sleeve 1 Entry: Technical / Tactical

`Sleeve 1` can enter through either of two triggers.

### Trigger A: Mean reversion

Conditions:

- `rsi_14` exists and is not NaN
- `rsi_14 < rsi_threshold`
- no volatility veto

Default parameter:

- `rsi_threshold = 40`

Interpretation:

- the stock looks oversold on RSI
- short-term volatility has not exploded relative to medium-term volatility
- this trigger is allowed even when the stock is below `ma_200`

### Trigger B: Volume-backed momentum

This trigger is only checked if Trigger A did not already fire.

Conditions:

- not in downtrend
- no volatility veto
- `daily_return`, `volume`, and `volume_ma_20` all exist and are not NaN
- `daily_return > 0`
- `volume_ma_20 > 0`
- `volume > vol_mult * volume_ma_20`

Default parameter:

- `vol_mult = 2`

Interpretation:

- the stock is up on the day
- current volume is at least 2x its 20-day average volume
- the long-term trend is not bearish

### What happens on a Sleeve 1 buy

The strategy stores:

- `entry_date`
- `peak_price`

Then it returns `BUY`.

## Sleeve 2 Entry: Earnings Sentiment Acceleration

`Sleeve 2` is only checked if `Sleeve 1` did not trigger.

Conditions:

- transcript is available
- not in downtrend
- no volatility veto
- `llm_analysis()` returns `sentiment_acceleration = True`

### How `llm_analysis()` works

For a given transcript:

1. Split the transcript into sentence-like chunks.
2. Build text chunks of about 400 characters.
3. Run FinBERT on each chunk, capped at 512 characters.
4. Count how many chunks are positive, negative, or neutral.
5. Compute:
   - `net_sentiment = (positive - negative) / total_chunks`
   - dominant label = whichever of positive, negative, neutral has the highest count
6. Store the quarter's `net_sentiment` in `self.sentiment_cache`.
7. Compare the current quarter's net sentiment with the previous quarter's net sentiment.

Acceleration rule:

- `sentiment_acceleration = current_quarter_net_sentiment > previous_quarter_net_sentiment`

If that condition is true, the strategy opens a `Sleeve 2` position.

### What happens on a Sleeve 2 buy

The strategy stores:

- `entry_date`
- `weeks_held = 0`
- `peak_price`

Then it returns `BUY`.

## Exit Logic

Exit logic is checked before any entry logic.

## Sleeve 1 Exit

If the ticker is already inside `self.sleeve1_positions`, the strategy:

1. updates `peak_price` to the maximum of old peak and current price
2. checks two exit rules

It exits if either condition is true:

- `price <= peak_price - (s1_atr_mult * atr_14)`
- `date > entry_date`

Default parameter:

- `s1_atr_mult = 2`

Interpretation:

- there is a 2-ATR trailing stop from the highest price seen since entry
- there is also a time-based exit on the first later evaluation date after entry

Practical consequence:

- `Sleeve 1` is a very short holding-period trade
- because `date > entry_date` becomes true on the next later trading date, the position is usually exited on the next evaluation after entry unless the position is already sold by the ATR stop condition

## Sleeve 2 Exit

If the ticker is already inside `self.sleeve2_positions`, the strategy:

1. increments `weeks_held` by 1
2. updates `peak_price`
3. checks exit conditions

It exits if either condition is true:

- `price <= peak_price - (s2_atr_mult * atr_14)`
- `weeks_held >= sleeve2_exit_weeks`

Default parameters:

- `s2_atr_mult = 2`
- `sleeve2_exit_weeks = 5`

Interpretation:

- the position uses a 2-ATR trailing stop
- it also has a maximum holding horizon of 5 decision periods

Practical note:

- despite the variable name `weeks_held`, the counter increases every time `make_decision()` is called while the position is open
- the true calendar holding time depends on how often the backtest engine evaluates the strategy

## Position Cleanup Rule

There is one extra sell rule:

- if the portfolio says a ticker is held
- and the ticker is no longer in the monthly universe
- and the ticker is not tracked inside either sleeve dictionary

then the strategy returns `SELL`

This acts as a cleanup path for positions that exist in the portfolio state but are not being actively tracked by either sleeve.

## Why the Strategy Can Return `HOLD`

The strategy returns `HOLD` when:

- price is invalid
- the stock is outside the current universe
- the stock is already held and no exit is triggered
- volatility acceleration veto blocks entries
- downtrend blocks momentum or transcript entries
- technical triggers fail
- transcript sentiment acceleration fails

## Parameter Summary

Default parameters for the final notebook strategy:

- `rsi_threshold = 40`
- `vol_mult = 2`
- `s1_atr_mult = 2`
- `s2_atr_mult = 2`
- `sleeve2_exit_weeks = 5`

## In Plain English

This strategy first narrows the market to a monthly universe. Inside that universe, it tries to buy either:

- technically oversold stocks that are not in a volatility spike
- strong up days with unusually high volume, as long as the stock is not below its 200-day trend
- stocks whose earnings-call sentiment has improved versus the previous quarter, as long as trend and volatility filters are acceptable

Once in a trade:

- `Sleeve 1` is meant to be short-lived and exits quickly
- `Sleeve 2` is allowed to run longer but still uses a trailing stop and a fixed maximum holding horizon

## Notebook Scope

This README is based on the last `EnhancedStrategy` class currently present in [trading_assignment.ipynb](c:\Users\Sebert\Desktop\SIT Lambert\Y2T2\INF2006 - Cloud Computing and Big Data\Project\Algorithmic Trading\trading_assignment.ipynb). If the notebook logic changes later, this file should be updated to match.


