import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from scipy.stats import ttest_ind


POSITIVE_WORDS = {
    "growth",
    "strong",
    "profit",
    "success",
    "opportunity",
    "excellent",
    "positive",
    "gain",
    "improvement",
    "best",
    "higher",
    "increase",
    "rose",
    "record",
    "benefit",
    "up",
    "expanding",
    "revenue",
    "income",
    "outperform",
}

NEGATIVE_WORDS = {
    "loss",
    "decline",
    "negative",
    "risk",
    "fail",
    "difficult",
    "challenging",
    "weak",
    "worst",
    "uncertainty",
    "lower",
    "decrease",
    "fell",
    "drop",
    "down",
    "missed",
    "adverse",
    "struggle",
    "concerns",
    "volatile",
}


# Expected to be defined by the parent script / notebook:
# prices_dev, prices_val, earnings_dev, earnings_val


def calculate_sentiment(text):
    if not isinstance(text, str):
        return 0.0

    words = text.lower().split()
    pos_count = sum(1 for word in words if word in POSITIVE_WORDS)
    neg_count = sum(1 for word in words if word in NEGATIVE_WORDS)
    total = pos_count + neg_count

    if total == 0:
        return 0.0

    return (pos_count - neg_count) / total


def print_section(title):
    print(f"\n{'=' * 80}")
    print(title)
    print(f"{'=' * 80}")


def run_low_vs_high_volatility_anomaly():
    print_section("Hypothesis 1: Low vs High Volatility Anomaly")
    print("Using preloaded prices_dev...")
    df = prices_dev.copy()

    print("Preprocessing...")
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["ticker", "date"])
    df["daily_ret"] = df.groupby("ticker")["close"].pct_change()

    print("Calculating rolling volatility...")
    df["volatility"] = df.groupby("ticker")["daily_ret"].transform(
        lambda series: series.rolling(window=252, min_periods=200).std() * np.sqrt(252)
    )

    print("Resampling to monthly frequency...")
    df = df.set_index("date")

    try:
        monthly_df = df.groupby("ticker").resample("ME").agg({"close": "last", "volatility": "last"})
    except ValueError:
        monthly_df = df.groupby("ticker").resample("M").agg({"close": "last", "volatility": "last"})

    monthly_df = monthly_df.reset_index()
    monthly_df["fwd_ret"] = monthly_df.groupby("ticker")["close"].pct_change().shift(-1)
    monthly_df = monthly_df.dropna(subset=["volatility", "fwd_ret"])

    if monthly_df.empty:
        print("No data available after processing.")
        return

    print("Ranking stocks into deciles...")
    monthly_df["decile"] = monthly_df.groupby("date")["volatility"].transform(
        lambda series: pd.qcut(series, 10, labels=False, duplicates="drop") + 1
        if len(series) >= 20
        else np.nan
    )
    monthly_df = monthly_df.dropna(subset=["decile"])

    print("Computing portfolio stats...")
    portfolio_ts = monthly_df.groupby(["date", "decile"])["fwd_ret"].mean().reset_index()
    stats_df = portfolio_ts.groupby("decile")["fwd_ret"].agg(["mean", "std", "count"])

    stats_df["ann_ret"] = stats_df["mean"] * 12
    stats_df["ann_vol"] = stats_df["std"] * np.sqrt(12)
    stats_df["sharpe"] = stats_df["ann_ret"] / stats_df["ann_vol"]

    print("\n--- Low vs High Volatility Anomaly Results ---")
    print(stats_df[["ann_ret", "ann_vol", "sharpe"]])

    try:
        low_vol_sharpe = stats_df.loc[1.0, "sharpe"]
        high_vol_sharpe = stats_df.loc[10.0, "sharpe"]

        print(f"\nLow Volatility (Decile 1) Sharpe: {low_vol_sharpe:.4f}")
        print(f"High Volatility (Decile 10) Sharpe: {high_vol_sharpe:.4f}")

        if low_vol_sharpe > high_vol_sharpe:
            print("\nConclusion: Low Volatility Anomaly CONFIRMED (Low Vol > High Vol).")
        else:
            print("\nConclusion: Low Volatility Anomaly REJECTED (High Vol > Low Vol).")
    except KeyError:
        print("\nCould not extract Decile 1 or 10 statistics.")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    stats_df["sharpe"].plot(kind="bar", ax=ax1, color="skyblue", edgecolor="black")
    ax1.set_title("Sharpe Ratio by Volatility Decile")
    ax1.set_xlabel("Volatility Decile (1=Lowest, 10=Highest)")
    ax1.set_ylabel("Annualized Sharpe Ratio")
    ax1.grid(axis="y", alpha=0.3)

    ax2.scatter(stats_df["ann_vol"], stats_df["ann_ret"], c="blue", s=100, alpha=0.7)
    ax2.plot(stats_df["ann_vol"], stats_df["ann_ret"], "b--", alpha=0.3)

    for idx, row in stats_df.iterrows():
        ax2.annotate(
            f"D{int(idx)}",
            (row["ann_vol"], row["ann_ret"]),
            xytext=(5, 5),
            textcoords="offset points",
        )

    ax2.set_title("Risk-Return Profile")
    ax2.set_xlabel("Annualized Volatility")
    ax2.set_ylabel("Annualized Return")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def run_volume_confirmation_experiment():
    print_section("Hypothesis 2: High-Volume vs Low-Volume Up Days")
    print("Using preloaded prices_dev...")
    df = prices_dev.copy()
    print("Dataset loaded successfully.")

    df.sort_values(by=["ticker", "date"], inplace=True)
    df["prev_close"] = df.groupby("ticker")["close"].shift(1)
    df["return"] = (df["close"] - df["prev_close"]) / df["prev_close"]
    df["next_return"] = df.groupby("ticker")["return"].shift(-1)
    df["vol_ma_20"] = df.groupby("ticker")["volume"].transform(lambda series: series.rolling(window=20).mean())

    df_clean = df.dropna(subset=["return", "next_return", "vol_ma_20", "volume"]).copy()
    up_days = df_clean[df_clean["return"] > 0].copy()

    print(f"Total Up Days processed: {len(up_days)}")

    high_vol_mask = up_days["volume"] > (1.5 * up_days["vol_ma_20"])
    low_vol_mask = up_days["volume"] <= up_days["vol_ma_20"]

    high_vol_returns = up_days.loc[high_vol_mask, "next_return"]
    low_vol_returns = up_days.loc[low_vol_mask, "next_return"]

    n_high = len(high_vol_returns)
    n_low = len(low_vol_returns)

    if n_high == 0 or n_low == 0:
        print("Insufficient data points for one of the groups.")
        return

    mean_high = high_vol_returns.mean()
    mean_low = low_vol_returns.mean()
    std_high = high_vol_returns.std()
    std_low = low_vol_returns.std()

    print("\n--- Results ---")
    print(
        f"High Volume Up-Days (>150% MA) (n={n_high}): Mean Next-Day Return = "
        f"{mean_high:.6f} ({mean_high * 100:.4f}%), Std = {std_high:.6f}"
    )
    print(
        f"Low Volume Up-Days  (<=100% MA) (n={n_low}): Mean Next-Day Return = "
        f"{mean_low:.6f} ({mean_low * 100:.4f}%), Std = {std_low:.6f}"
    )

    t_stat, p_val = stats.ttest_ind(high_vol_returns, low_vol_returns, equal_var=False)

    print("\n--- T-Test (Welch's) ---")
    print(f"T-Statistic: {t_stat:.4f}")
    print(f"P-Value: {p_val:.4e}")

    if p_val < 0.05:
        if t_stat > 0:
            print("Result: Significant Positive Difference (High Vol > Low Vol).")
        else:
            print("Result: Significant Negative Difference (High Vol < Low Vol).")
    else:
        print("Result: No Statistically Significant Difference.")

    plt.figure(figsize=(10, 6))
    means = [mean_high * 100, mean_low * 100]
    standard_errors = [std_high * 100 / np.sqrt(n_high), std_low * 100 / np.sqrt(n_low)]
    labels = ["High Volume (>150% MA)", "Low Volume (<=100% MA)"]

    plt.bar(labels, means, yerr=standard_errors, capsize=10, color=["#d62728", "#1f77b4"], alpha=0.8)
    plt.title("Mean Next-Day Return: High Vol vs Low Vol Up-Days")
    plt.ylabel("Average Next-Day Return (%)")
    plt.grid(axis="y", linestyle="--", alpha=0.5)

    for idx, value in enumerate(means):
        plt.text(idx, value + (0.01 if value >= 0 else -0.05), f"{value:.4f}%", ha="center", fontweight="bold")

    plt.show()


def run_rsi_mean_reversion_experiment():
    print_section("Hypothesis 3: RSI Oversold vs Overbought")
    print("Using preloaded prices_dev...")
    df = prices_dev.copy()

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

    print("Calculating RSI...")
    df["delta"] = df.groupby("ticker")["close"].diff()

    up = df["delta"].clip(lower=0)
    down = -1 * df["delta"].clip(upper=0)

    ma_up = up.groupby(df["ticker"]).ewm(alpha=1 / 14, adjust=False).mean()
    ma_down = down.groupby(df["ticker"]).ewm(alpha=1 / 14, adjust=False).mean()

    df["avg_gain"] = ma_up.reset_index(level=0, drop=True)
    df["avg_loss"] = ma_down.reset_index(level=0, drop=True)

    with np.errstate(divide="ignore", invalid="ignore"):
        rs = df["avg_gain"] / df["avg_loss"]
        df["rsi"] = 100 - (100 / (1 + rs))

    df.loc[df["avg_loss"] == 0, "rsi"] = 100
    df["next_close"] = df.groupby("ticker")["close"].shift(-1)
    df["next_ret"] = (df["next_close"] - df["close"]) / df["close"]

    valid_data = df.dropna(subset=["rsi", "next_ret"])
    oversold = valid_data[valid_data["rsi"] < 30]["next_ret"]
    overbought = valid_data[valid_data["rsi"] > 70]["next_ret"]

    print(f"Oversold (RSI < 30) count: {len(oversold)}")
    print(f"Overbought (RSI > 70) count: {len(overbought)}")

    mean_os = oversold.mean()
    mean_ob = overbought.mean()

    print(f"Mean Next-Day Return (Oversold): {mean_os:.6f}")
    print(f"Mean Next-Day Return (Overbought): {mean_ob:.6f}")

    t_stat, p_val = ttest_ind(oversold, overbought, equal_var=False)
    print(f"T-statistic: {t_stat:.4f}")
    print(f"P-value: {p_val:.4e}")

    plt.figure(figsize=(10, 6))
    plt.hist(oversold, bins=100, range=(-0.05, 0.05), density=True, alpha=0.5, label="Oversold (RSI < 30)")
    plt.hist(overbought, bins=100, range=(-0.05, 0.05), density=True, alpha=0.5, label="Overbought (RSI > 70)")
    plt.title("Next-Day Return Distribution: Oversold vs Overbought")
    plt.xlabel("Next Day Return")
    plt.ylabel("Density")
    plt.legend()
    plt.show()


def run_illiquidity_premium_experiment():
    print_section("Hypothesis 4: Illiquidity Premium")
    print("Using preloaded prices_dev...")
    df = prices_dev.copy()

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["ticker", "date"])

    print("Calculating Amihud Ratio...")
    df["ret"] = df.groupby("ticker")["close"].pct_change()
    df["dollar_vol"] = (df["close"] * df["volume"]).replace(0, np.nan)
    df["amihud"] = (df["ret"].abs() / df["dollar_vol"]).replace([np.inf, -np.inf], np.nan)

    print("Aggregating to monthly frequency...")
    monthly_data = df.groupby(["ticker", pd.Grouper(key="date", freq="ME")]).agg({"amihud": "mean", "close": "last"}).reset_index()
    monthly_data["ret_m"] = monthly_data.groupby("ticker")["close"].pct_change()
    monthly_data["fwd_ret_m"] = monthly_data.groupby("ticker")["ret_m"].shift(-1)

    valid_data = monthly_data.dropna(subset=["amihud", "fwd_ret_m"]).copy()

    print("Ranking stocks into deciles...")

    def rank_deciles(series):
        try:
            return pd.qcut(series, 10, labels=False, duplicates="drop")
        except ValueError:
            return np.nan

    valid_data["decile"] = valid_data.groupby("date")["amihud"].transform(rank_deciles)
    valid_data = valid_data.dropna(subset=["decile"])

    print("Constructing portfolios...")
    portfolios = valid_data.groupby(["date", "decile"])["fwd_ret_m"].mean().unstack()

    if 0.0 not in portfolios.columns or 9.0 not in portfolios.columns:
        print("Error: Deciles 0 and 9 not found in aggregated data.")
        print("Columns found:", portfolios.columns)
        return

    illiquid_returns = portfolios[9.0]
    liquid_returns = portfolios[0.0]
    spread_returns = illiquid_returns - liquid_returns

    mean_spread = spread_returns.mean()
    t_stat, p_val = stats.ttest_1samp(spread_returns, 0)

    print("\n--- Illiquidity Premium Analysis Results ---")
    print(f"Observation Period: {valid_data['date'].min().date()} to {valid_data['date'].max().date()}")
    print(f"Total Months: {len(spread_returns)}")
    print(f"Average Monthly Spread Return: {mean_spread:.4%}")
    print(f"Annualized Spread Return: {(1 + mean_spread) ** 12 - 1:.4%}")
    print(f"T-Statistic: {t_stat:.4f}")
    print(f"P-Value: {p_val:.4f}")

    cum_illiquid = (1 + illiquid_returns).cumprod()
    cum_liquid = (1 + liquid_returns).cumprod()
    cum_spread = (1 + spread_returns).cumprod()

    plt.figure(figsize=(10, 6))
    plt.plot(cum_illiquid.index, cum_illiquid, label="Illiquid (Top Decile)")
    plt.plot(cum_liquid.index, cum_liquid, label="Liquid (Bottom Decile)")
    plt.plot(cum_spread.index, cum_spread, label="Long Illiquid / Short Liquid", linestyle="--", color="black")
    plt.title("Illiquidity Premium: Cumulative Returns")
    plt.xlabel("Date")
    plt.ylabel("Growth of $1")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def run_sentiment_acceleration_experiment():
    print_section("Hypothesis 5: Sentiment Acceleration vs Post-Earnings Sharpe")
    print("Using preloaded earnings_dev, earnings_val, prices_dev, and prices_val...")
    df_earnings = pd.concat([earnings_dev, earnings_val], ignore_index=True)
    df_prices = pd.concat([prices_dev, prices_val], ignore_index=True)

    print(f"Earnings events loaded: {len(df_earnings)}")
    print(f"Price records loaded: {len(df_prices)}")

    print("Calculating sentiment scores...")
    df_earnings["sentiment"] = df_earnings["transcript"].apply(calculate_sentiment)

    print("Calculating sentiment acceleration...")
    df_earnings["date"] = pd.to_datetime(df_earnings["date"]).dt.tz_localize(None).dt.normalize()
    df_earnings = df_earnings.sort_values(by=["ticker", "date"])
    df_earnings["prev_sentiment"] = df_earnings.groupby("ticker")["sentiment"].shift(1)
    df_earnings["sentiment_change"] = df_earnings["sentiment"] - df_earnings["prev_sentiment"]
    df_earnings_clean = df_earnings.dropna(subset=["sentiment_change"]).copy()

    print("Processing price data...")
    df_prices["date"] = pd.to_datetime(df_prices["date"]).dt.tz_localize(None).dt.normalize()
    df_prices = df_prices.sort_values(["ticker", "date"])
    df_prices["return"] = df_prices.groupby("ticker")["close"].pct_change()

    print("Building price lookup dictionary...")
    price_dict = {ticker: frame.set_index("date")["return"] for ticker, frame in df_prices.groupby("ticker")}

    sharpe_ratios = []
    valid_indices = []

    print(f"Calculating post-earnings Sharpe Ratio for {len(df_earnings_clean)} events...")

    for idx, row in df_earnings_clean.iterrows():
        ticker = row["ticker"]
        earnings_date = row["date"]

        if ticker not in price_dict:
            continue

        ticker_returns = price_dict[ticker]

        try:
            idx_loc = ticker_returns.index.searchsorted(earnings_date, side="left")

            if idx_loc >= len(ticker_returns):
                continue

            current_date_at_idx = ticker_returns.index[idx_loc]
            start_pos = idx_loc + 1 if current_date_at_idx == earnings_date else idx_loc
            end_pos = start_pos + 20

            if end_pos > len(ticker_returns):
                continue

            window_returns = ticker_returns.iloc[start_pos:end_pos].values
            window_returns = window_returns[~np.isnan(window_returns)]

            if len(window_returns) < 10:
                continue

            mean_ret = np.mean(window_returns)
            std_ret = np.std(window_returns)
            sharpe = 0 if std_ret == 0 or np.isnan(std_ret) else mean_ret / std_ret

            sharpe_ratios.append(sharpe)
            valid_indices.append(idx)
        except Exception:
            continue

    df_analysis = df_earnings_clean.loc[valid_indices].copy()
    df_analysis["sharpe_ratio"] = sharpe_ratios

    print(f"Successfully computed Sharpe Ratios for {len(df_analysis)} events.")

    if len(df_analysis) < 50:
        print("Insufficient data points for meaningful regression.")
        return

    print("\nRunning Regression Analysis...")
    df_analysis = df_analysis.dropna(subset=["sharpe_ratio", "sentiment", "sentiment_change"])
    df_analysis["sentiment_z"] = (df_analysis["sentiment"] - df_analysis["sentiment"].mean()) / df_analysis["sentiment"].std()
    df_analysis["sentiment_change_z"] = (
        (df_analysis["sentiment_change"] - df_analysis["sentiment_change"].mean()) / df_analysis["sentiment_change"].std()
    )

    X1 = sm.add_constant(df_analysis["sentiment_z"])
    y = df_analysis["sharpe_ratio"]
    model1 = sm.OLS(y, X1).fit()

    X2 = sm.add_constant(df_analysis["sentiment_change_z"])
    model2 = sm.OLS(y, X2).fit()

    X3 = sm.add_constant(df_analysis[["sentiment_z", "sentiment_change_z"]])
    model3 = sm.OLS(y, X3).fit()

    print("\n=== Regression 1: Sharpe ~ Absolute Sentiment (Z-scored) ===")
    print(model1.summary().tables[1])
    print(f"R-squared: {model1.rsquared:.6f}")

    print("\n=== Regression 2: Sharpe ~ Sentiment Change (Z-scored) ===")
    print(model2.summary().tables[1])
    print(f"R-squared: {model2.rsquared:.6f}")

    print("\n=== Regression 3: Sharpe ~ Absolute + Change ===")
    print(model3.summary().tables[1])
    print(f"R-squared: {model3.rsquared:.6f}")

    df_analysis["group"] = df_analysis["sentiment_change"].apply(lambda value: "Improving" if value > 0 else "Deteriorating")
    group_means = df_analysis.groupby("group")["sharpe_ratio"].mean()
    group_counts = df_analysis.groupby("group")["sharpe_ratio"].count()

    print("\n=== Group Comparison (Average Daily Sharpe Ratio) ===")
    print(group_means)
    print("\nGroup Counts:")
    print(group_counts)

    plt.figure(figsize=(10, 6))
    group_means.plot(kind="bar", color=["red", "green"], alpha=0.7)
    plt.title("Average Post-Earnings Sharpe Ratio (t+1 to t+20)\nby Sentiment Momentum")
    plt.ylabel("Average Daily Sharpe Ratio")
    plt.xlabel("Sentiment Change")
    plt.axhline(0, color="black", linewidth=0.8)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.show()


def run_volatility_acceleration_experiment():
    print_section("Hypothesis 6: Volatility Acceleration vs Forward Drawdown")
    df = pd.concat([prices_dev, prices_val], ignore_index=True)
    df.sort_values(["ticker", "date"], inplace=True)

    df["ret"] = df.groupby("ticker")["close"].pct_change()
    df["vol_10"] = df.groupby("ticker")["ret"].transform(lambda series: series.rolling(10, min_periods=10).std())
    df["vol_60"] = df.groupby("ticker")["ret"].transform(lambda series: series.rolling(60, min_periods=60).std())
    df["vol_ratio"] = df["vol_10"] / df["vol_60"]

    df["next_low"] = df.groupby("ticker")["low"].shift(-1)

    indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=20)
    df["fwd_min_low"] = df.groupby("ticker")["next_low"].transform(
        lambda series: series.rolling(window=indexer, min_periods=20).min()
    )
    df["fwd_drawdown"] = (df["fwd_min_low"] - df["close"]) / df["close"]

    valid_data = df.dropna(subset=["vol_ratio", "fwd_drawdown"])
    shock_group = valid_data[valid_data["vol_ratio"] > 1.5]["fwd_drawdown"]
    control_group = valid_data[(valid_data["vol_ratio"] >= 0.9) & (valid_data["vol_ratio"] <= 1.1)]["fwd_drawdown"]

    ks_stat, p_value = stats.ks_2samp(shock_group, control_group)

    print("=== Volatility Acceleration Experiment Results ===")
    print(f"Shock Group Size: {len(shock_group)}")
    print(f"Control Group Size: {len(control_group)}")
    print(f"Shock Group Mean Forward Drawdown (Next 20 Days): {shock_group.mean():.4f}")
    print(f"Control Group Mean Forward Drawdown (Next 20 Days): {control_group.mean():.4f}")
    print(f"KS Test Statistic: {ks_stat:.4f}")
    print(f"KS Test p-value: {p_value:.4e}")

    shock_prob = (shock_group < -0.10).mean()
    control_prob = (control_group < -0.10).mean()
    print(f"Probability of >10% price drop in Shock group: {shock_prob:.2%}")
    print(f"Probability of >10% price drop in Control group: {control_prob:.2%}")

    plt.figure(figsize=(10, 6))
    plt.hist(control_group, bins=100, range=(-0.5, 0.1), density=True, alpha=0.5, label="Control (Normal Vol)", color="blue")
    plt.hist(shock_group, bins=100, range=(-0.5, 0.1), density=True, alpha=0.5, label="Shock (Vol Accel > 1.5x)", color="red")
    plt.title("Distribution of Maximum Forward Drawdown (Next 20 Days)")
    plt.xlabel("Max Drawdown (Relative to Entry Price)")
    plt.ylabel("Density")
    plt.axvline(-0.10, color="black", linestyle="--", alpha=0.7, label="10% Drop Threshold")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


def run_trend_filter_crash_experiment():
    print_section("Hypothesis 7: Trend Filter During Crash Periods")
    print("Using preloaded prices_dev...")
    df = prices_dev.copy()

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["ticker", "date"])

    print("Calculating SMA and Returns...")
    grouped = df.groupby("ticker")
    df["sma_200"] = grouped["close"].transform(lambda series: series.rolling(200).mean())
    df["prev_close"] = grouped["close"].shift(1)
    df["return"] = df["close"] / df["prev_close"] - 1.0
    df["signal"] = (df["close"] > df["sma_200"]).astype(int)
    df["strategy_pos"] = grouped["signal"].shift(1).fillna(0)
    df["strat_ret"] = df["return"] * df["strategy_pos"]

    market_series = df.groupby("date")["close"].mean()
    market_peak = market_series.cummax()
    market_dd = (market_series - market_peak) / market_peak

    global_low_date = market_dd.idxmin()
    peak_date = market_series.loc[:global_low_date].idxmax()

    crash_start = peak_date
    crash_end = global_low_date
    print(f"Crash Period Identified: {crash_start.date()} to {crash_end.date()}")
    print(f"Market Drawdown: {market_dd.min():.2%}")

    lookback_start = crash_start - pd.Timedelta(days=180)
    pre_crash_data = df[(df["date"] >= lookback_start) & (df["date"] < crash_start)]
    vol_stats = pre_crash_data.groupby("ticker")["return"].std()

    vol_threshold = vol_stats.quantile(0.8)
    high_vol_tickers = vol_stats[vol_stats >= vol_threshold].index
    print(f"High Volatility Threshold (Std Dev): {vol_threshold:.4f}")
    print(f"Selected {len(high_vol_tickers)} high volatility stocks.")

    sim_data = df[
        (df["date"] >= crash_start)
        & (df["date"] <= crash_end)
        & (df["ticker"].isin(high_vol_tickers))
    ].copy()

    results = []

    for ticker, group in sim_data.groupby("ticker"):
        if len(group) < 10:
            continue

        bh_equity = (1 + group["return"].fillna(0)).cumprod()
        bh_peak = bh_equity.cummax()
        bh_dd_series = (bh_equity - bh_peak) / bh_peak
        bh_mdd = bh_dd_series.min()

        tr_equity = (1 + group["strat_ret"].fillna(0)).cumprod()
        tr_peak = tr_equity.cummax()
        tr_dd_series = (tr_equity - tr_peak) / tr_peak
        tr_mdd = tr_dd_series.min()

        results.append({"ticker": ticker, "BH_MDD": bh_mdd, "Trend_MDD": tr_mdd})

    results_df = pd.DataFrame(results).dropna()

    mean_bh_mdd = results_df["BH_MDD"].mean()
    mean_tr_mdd = results_df["Trend_MDD"].mean()

    print("\n=== Simulation Results ===")
    print(f"Average Max Drawdown (Buy & Hold): {mean_bh_mdd:.2%}")
    print(f"Average Max Drawdown (Trend Filter): {mean_tr_mdd:.2%}")

    improvement = mean_tr_mdd - mean_bh_mdd
    print(f"Average Improvement: {improvement * 100:.2f} percentage points")

    t_stat, p_val = stats.ttest_rel(results_df["BH_MDD"], results_df["Trend_MDD"])
    print(f"Paired t-test: t-statistic={t_stat:.4f}, p-value={p_val:.4e}")

    if p_val < 0.05:
        print("Result: Statistically significant difference.")
    else:
        print("Result: No statistically significant difference.")

    plt.figure(figsize=(8, 6))
    plt.boxplot([results_df["BH_MDD"], results_df["Trend_MDD"]], tick_labels=["Buy & Hold", "Trend Filter"])
    plt.title("Distribution of Maximum Drawdowns (High Volatility Stocks)")
    plt.ylabel("Maximum Drawdown")
    plt.grid(True, axis="y", alpha=0.3)
    plt.show()

    piv_bh = sim_data.pivot(index="date", columns="ticker", values="return").mean(axis=1).fillna(0)
    piv_tr = sim_data.pivot(index="date", columns="ticker", values="strat_ret").mean(axis=1).fillna(0)

    cum_bh = (1 + piv_bh).cumprod()
    cum_tr = (1 + piv_tr).cumprod()

    plt.figure(figsize=(12, 6))
    plt.plot(cum_bh, label="Buy & Hold Portfolio", color="red")
    plt.plot(cum_tr, label="Trend Filter Portfolio", color="blue")
    plt.title(f"Performance of High Volatility Stocks during Crash ({crash_start.date()} - {crash_end.date()})")
    plt.xlabel("Date")
    plt.ylabel("Normalized Equity")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

def run_parkinson_volatility_persistence_experiment():
    print_section("Hypothesis 9: Parkinson Volatility vs Historical Volatility Persistence")
    df = prices_dev.copy()

    df = df.copy()
    df.sort_values(by=["ticker", "date"], inplace=True)

    df["prev_close"] = df.groupby("ticker")["close"].shift(1)
    df["log_ret"] = np.log(df["close"] / df["prev_close"])

    df["hist_vol"] = df.groupby("ticker")["log_ret"].transform(lambda series: series.rolling(window=20).std())

    df["log_hl_sq"] = (np.log(df["high"] / df["low"])) ** 2
    const_factor = 1.0 / (4.0 * np.log(2.0))
    df["park_vol_sq"] = df.groupby("ticker")["log_hl_sq"].transform(lambda series: series.rolling(window=20).mean())
    df["park_vol"] = np.sqrt(const_factor * df["park_vol_sq"])

    df["future_vol"] = df.groupby("ticker")["hist_vol"].shift(-20)
    valid_df = df.dropna(subset=["hist_vol", "park_vol", "future_vol"])

    print(f"Data points available for correlation analysis: {len(valid_df)}")
    if len(valid_df) == 0:
        print("No valid data points found.")
        return

    corr_hv = valid_df["hist_vol"].corr(valid_df["future_vol"])
    corr_pv = valid_df["park_vol"].corr(valid_df["future_vol"])

    print("\n--- Correlation Results ---")
    print(f"Historical Volatility vs Future Realized Volatility: {corr_hv:.4f}")
    print(f"Parkinson Volatility vs Future Realized Volatility:  {corr_pv:.4f}")

    print("\n--- Conclusion ---")
    if corr_pv > corr_hv:
        print("Parkinson Volatility has higher predictive persistence (higher correlation with future volatility).")
    else:
        print("Historical Volatility has higher predictive persistence (higher correlation with future volatility).")

    plot_df = valid_df.sample(min(5000, len(valid_df)), random_state=42)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Hypothesis 9: Parkinson vs Historical Volatility Persistence", fontsize=14, fontweight="bold")

    axes[0].scatter(plot_df["hist_vol"], plot_df["future_vol"], alpha=0.2, s=10, color="#1f77b4")
    axes[0].set_title(f"Historical Vol vs Future Vol\nCorr = {corr_hv:.4f}")
    axes[0].set_xlabel("20-Day Historical Volatility")
    axes[0].set_ylabel("Future Realized Volatility")
    axes[0].grid(True, alpha=0.3)

    axes[1].scatter(plot_df["park_vol"], plot_df["future_vol"], alpha=0.2, s=10, color="#d62728")
    axes[1].set_title(f"Parkinson Vol vs Future Vol\nCorr = {corr_pv:.4f}")
    axes[1].set_xlabel("20-Day Parkinson Volatility")
    axes[1].set_ylabel("Future Realized Volatility")
    axes[1].grid(True, alpha=0.3)

    labels = ["Historical Vol", "Parkinson Vol"]
    correlations = [corr_hv, corr_pv]
    colors = ["#1f77b4", "#d62728"]
    axes[2].bar(labels, correlations, color=colors, edgecolor="black", alpha=0.85)
    axes[2].set_title("Correlation With Future Realized Volatility")
    axes[2].set_ylabel("Correlation")
    axes[2].grid(axis="y", alpha=0.3)
    for idx, value in enumerate(correlations):
        axes[2].text(idx, value + 0.005, f"{value:.4f}", ha="center", fontweight="bold")

    plt.tight_layout()
    plt.show()


def run_amihud_vol_window_search_experiment():
    """
    Hypothesis 8: Amihud & Volatility Lookback Window Grid Search.

    Tests whether the 60-day lookback currently used for both the Amihud
    illiquidity ratio and the volatility metric in universe selection is
    optimal, or whether a shorter/longer window (20, 40, 60, 120 days)
    produces a better-quality investable universe.

    Method:
      For each (amihud_window, vol_window) combination:
        1. Compute rolling Amihud(W) and rolling Vol(W) per ticker.
        2. Monthly: rank tickers by each metric, form composite score,
           select top 40% (universe).
        3. Measure equal-weighted forward 1-month return of that universe.
        4. Compute annualised Sharpe of the strategy.

    A higher Sharpe = better predictive quality from that window.
    """
    print_section("Hypothesis 8: Amihud & Volatility Lookback Window Grid Search")
    print("Using preloaded prices_dev...")

    df = prices_dev.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["ticker", "date"])

    df["ret"] = df.groupby("ticker")["close"].pct_change()
    df["dollar_vol"] = (df["close"] * df["volume"]).replace(0, np.nan)
    df["amihud_daily"] = (df["ret"].abs() / df["dollar_vol"]).replace([np.inf, -np.inf], np.nan)

    windows = [20, 40, 60, 120]
    results = []

    for amihud_w in windows:
        for vol_w in windows:
            label = f"Amihud-{amihud_w}d / Vol-{vol_w}d"
            print(f"  Testing {label}...")

            # --- Compute windowed metrics for every row ---
            df["amihud_roll"] = (
                df.groupby("ticker")["amihud_daily"]
                .transform(lambda s: s.rolling(amihud_w, min_periods=max(10, amihud_w // 2)).mean())
            )
            df["vol_roll"] = (
                df.groupby("ticker")["ret"]
                .transform(lambda s: s.rolling(vol_w, min_periods=max(10, vol_w // 2)).std() * np.sqrt(252))
            )

            # --- Resample to month-end snapshots ---
            df_indexed = df.set_index("date")
            try:
                monthly = (
                    df_indexed.groupby("ticker")
                    .resample("ME")
                    .agg({"close": "last", "amihud_roll": "last", "vol_roll": "last"})
                )
            except ValueError:
                monthly = (
                    df_indexed.groupby("ticker")
                    .resample("M")
                    .agg({"close": "last", "amihud_roll": "last", "vol_roll": "last"})
                )
            monthly = monthly.reset_index()
            monthly["fwd_ret"] = monthly.groupby("ticker")["close"].pct_change().shift(-1)
            monthly = monthly.dropna(subset=["amihud_roll", "vol_roll", "fwd_ret"])

            # --- Monthly universe selection: top 40% composite score ---
            def score_month(group):
                if len(group) < 5:
                    group["selected"] = False
                    return group
                group = group.copy()
                group["vol_rank"] = group["vol_roll"].rank(ascending=True, method="average")
                group["amihud_rank"] = group["amihud_roll"].rank(ascending=False, method="average")
                group["composite"] = group["vol_rank"] + group["amihud_rank"]
                threshold = group["composite"].quantile(0.60)  # top 40%
                group["selected"] = group["composite"] >= threshold
                return group

            monthly = monthly.groupby("date", group_keys=False).apply(score_month)

            # --- Equal-weighted portfolio return of selected universe ---
            portfolio = (
                monthly[monthly["selected"]]
                .groupby("date")["fwd_ret"]
                .mean()
            )

            if len(portfolio) < 12:
                print(f"    Insufficient months ({len(portfolio)}) — skipping.")
                continue

            ann_ret = portfolio.mean() * 12
            ann_vol = portfolio.std() * np.sqrt(12)
            sharpe = ann_ret / ann_vol if ann_vol > 0 else np.nan
            hit_rate = (portfolio > 0).mean()

            results.append({
                "label": label,
                "amihud_window": amihud_w,
                "vol_window": vol_w,
                "ann_return": ann_ret,
                "ann_vol": ann_vol,
                "sharpe": sharpe,
                "hit_rate": hit_rate,
                "months": len(portfolio),
            })
            print(f"    Sharpe={sharpe:.3f}  AnnRet={ann_ret:.2%}  AnnVol={ann_vol:.2%}  HitRate={hit_rate:.2%}")

    if not results:
        print("No valid results produced.")
        return

    results_df = pd.DataFrame(results).sort_values("sharpe", ascending=False)

    print("\n--- Grid Search Results (sorted by Sharpe) ---")
    print(results_df[["label", "ann_return", "ann_vol", "sharpe", "hit_rate", "months"]].to_string(index=False))

    best = results_df.iloc[0]
    current = results_df[results_df["label"] == "Amihud-60d / Vol-60d"]
    print(f"\nBest window  : {best['label']} (Sharpe={best['sharpe']:.4f})")
    if not current.empty:
        curr_sharpe = current.iloc[0]["sharpe"]
        print(f"Current (60d): Amihud-60d / Vol-60d (Sharpe={curr_sharpe:.4f})")
        diff = best["sharpe"] - curr_sharpe
        if diff > 0.05:
            print(f"Conclusion: A different window improves Sharpe by {diff:.4f} — consider changing the lookback.")
        else:
            print(f"Conclusion: 60-day window is competitive (difference < 0.05 Sharpe). Current setting is acceptable.")

    # --- Plots ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Hypothesis 8: Amihud & Vol Lookback Window Grid Search", fontsize=14, fontweight="bold")

    # Heatmap: Sharpe by (amihud_w, vol_w)
    pivot_sharpe = results_df.pivot(index="amihud_window", columns="vol_window", values="sharpe")
    im = axes[0].imshow(pivot_sharpe.values, cmap="RdYlGn", aspect="auto")
    axes[0].set_xticks(range(len(pivot_sharpe.columns)))
    axes[0].set_yticks(range(len(pivot_sharpe.index)))
    axes[0].set_xticklabels([f"{c}d" for c in pivot_sharpe.columns])
    axes[0].set_yticklabels([f"{r}d" for r in pivot_sharpe.index])
    axes[0].set_xlabel("Vol Window")
    axes[0].set_ylabel("Amihud Window")
    axes[0].set_title("Sharpe Ratio Heatmap")
    for i in range(len(pivot_sharpe.index)):
        for j in range(len(pivot_sharpe.columns)):
            val = pivot_sharpe.values[i, j]
            if not np.isnan(val):
                axes[0].text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=9,
                             color="black" if 0.2 < val < 0.8 else "white")
    fig.colorbar(im, ax=axes[0])

    # Bar: Sharpe by label
    colors = ["#2E86AB" if r["label"] != "Amihud-60d / Vol-60d" else "#E15759" for _, r in results_df.iterrows()]
    axes[1].barh(results_df["label"], results_df["sharpe"], color=colors, edgecolor="black", alpha=0.85)
    axes[1].axvline(x=0, color="black", linewidth=0.8)
    axes[1].set_xlabel("Annualised Sharpe Ratio")
    axes[1].set_title("Sharpe by Window Combination\n(red = current 60d/60d)")
    axes[1].grid(axis="x", alpha=0.3)

    # Scatter: Ann Return vs Ann Vol
    for _, row in results_df.iterrows():
        color = "#E15759" if row["label"] == "Amihud-60d / Vol-60d" else "#2E86AB"
        axes[2].scatter(row["ann_vol"], row["ann_return"], color=color, s=80, zorder=3)
        axes[2].annotate(
            row["label"].replace(" / ", "\n"),
            (row["ann_vol"], row["ann_return"]),
            xytext=(4, 4), textcoords="offset points", fontsize=7
        )
    axes[2].set_xlabel("Annualised Volatility")
    axes[2].set_ylabel("Annualised Return")
    axes[2].set_title("Risk-Return by Window")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

def main():
    required_datasets = ("prices_dev", "prices_val", "earnings_dev", "earnings_val")
    missing = [name for name in required_datasets if name not in globals()]
    if missing:
        raise NameError(
            "Missing preloaded dataset variables: "
            + ", ".join(missing)
            + ". Define prices_dev, prices_val, earnings_dev, and earnings_val before running this file."
        )

    experiments = [
        run_low_vs_high_volatility_anomaly,
        run_volume_confirmation_experiment,
        run_rsi_mean_reversion_experiment,
        run_illiquidity_premium_experiment,
        run_sentiment_acceleration_experiment,
        run_volatility_acceleration_experiment,
        run_trend_filter_crash_experiment,
        run_amihud_vol_window_search_experiment,
        run_parkinson_volatility_persistence_experiment,
    ]

    for experiment in experiments:
        try:
            experiment()
        except Exception as exc:
            print(f"Experiment '{experiment.__name__}' failed: {exc}")


if __name__ == "__main__":
    main()