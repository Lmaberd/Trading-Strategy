"""
eval_loop.py — Strategy evaluation harness

Runs EnhancedStrategy on DEV and VAL splits, appends metrics to results_log.csv.
Run this after each strategy edit:  python eval_loop.py

Claude Code reads the CSV, improves strategy.py, and re-runs this script.
"""

import pickle
import shutil
import sys
import importlib.util
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import pipeline as hf_pipeline

# ─── Paths ────────────────────────────────────────────────────────────────────
ROOT          = Path(__file__).parent
DATA_DIR      = ROOT / 'INF2006_Data_Students'
CACHE_DIR     = ROOT / 'cache'
VERSIONS_DIR  = ROOT / 'enhanced_strat' / 'auto_versions'
CACHE_DIR.mkdir(exist_ok=True)
VERSIONS_DIR.mkdir(parents=True, exist_ok=True)

STRATEGY_FILE = ROOT / 'strategy.py'
RESULTS_CSV   = ROOT / 'results_log.csv'
STARTING_CASH = 100_000
FINBERT_MODEL = "ProsusAI/finbert"

# ─── Cache helpers ────────────────────────────────────────────────────────────
def _cache_save(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)

def _cache_load(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

# ─── Data loading ─────────────────────────────────────────────────────────────
def load_prices(split):
    import shutil, tempfile
    filepath = DATA_DIR / f'prices_{split}.parquet'
    if not filepath.exists():
        raise FileNotFoundError(f"Not found: {filepath}")
    tmp = Path(tempfile.gettempdir()) / f'prices_{split}.parquet'
    shutil.copy2(filepath, tmp)
    return pd.read_parquet(tmp)

def load_earnings(split):
    import shutil, tempfile
    filepath = DATA_DIR / f'earnings_{split}.parquet'
    if not filepath.exists():
        raise FileNotFoundError(f"Not found: {filepath}")
    tmp = Path(tempfile.gettempdir()) / f'earnings_{split}.parquet'
    shutil.copy2(filepath, tmp)
    return pd.read_parquet(tmp)

# ─── Metrics ──────────────────────────────────────────────────────────────────
def calculate_metrics(results, starting_cash=STARTING_CASH):
    history_df = pd.DataFrame(results['portfolio_history'])
    final_value = results['final_portfolio']['total_value']
    total_return = (final_value - starting_cash) / starting_cash

    history_df['weekly_return'] = history_df['portfolio_value'].pct_change().fillna(0)
    mean_r = history_df['weekly_return'].mean()
    std_r  = history_df['weekly_return'].std()
    sharpe = (mean_r / std_r * np.sqrt(52)) if std_r > 0 else 0.0

    peak = history_df['portfolio_value'].cummax()
    max_drawdown = ((history_df['portfolio_value'] - peak) / peak).min()

    trades_df = pd.DataFrame(results['trades']) if results['trades'] else pd.DataFrame()
    num_trades = len(trades_df)
    win_rate = 0.0
    if not trades_df.empty and 'action' in trades_df.columns:
        sells = trades_df[trades_df['action'] == 'SELL']
        buys  = trades_df[trades_df['action'] == 'BUY']
        if len(sells) > 0 and len(buys) > 0:
            buy_price = buys.groupby('ticker')['price'].last()
            profitable = sum(
                1 for _, s in sells.iterrows()
                if s['ticker'] in buy_price.index and s['price'] > buy_price[s['ticker']]
            )
            win_rate = profitable / len(sells)

    volatility = std_r * np.sqrt(52)

    return {
        'total_return': total_return,
        'sharpe_ratio': sharpe,
        'max_drawdown': max_drawdown,
        'win_rate':     win_rate,
        'volatility':   volatility,
        'num_trades':   num_trades,
        'final_value':  final_value,
    }

# ─── Infrastructure cache ─────────────────────────────────────────────────────
def load_infra(split, strategy_cls):
    ana_path = CACHE_DIR / f'analytics_{split}.parquet'

    print(f"\n[{split.upper()}] Loading data...")
    prices   = load_prices(split)
    earnings = load_earnings(split)

    _tmp = strategy_cls(None)
    prices, earnings = _tmp.clean_data(prices, earnings)
    print(f"  Prices: {len(prices):,}  Earnings: {len(earnings):,}")

    if ana_path.exists():
        print(f"  [CACHE HIT]  Analytics ({split})")
        analytics = pd.read_parquet(ana_path)
    else:
        print(f"  [CACHE MISS] Computing analytics ({split})...")
        _tmp.prices = prices
        analytics = _tmp.calculate_analytics(prices)
        analytics.to_parquet(ana_path, index=False)
        print(f"  Cached -> {ana_path.name}")

    return prices, earnings, analytics

# ─── Strategy loader ──────────────────────────────────────────────────────────
def load_strategy():
    spec = importlib.util.spec_from_file_location("strategy", STRATEGY_FILE)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.EnhancedStrategy

# ─── CSV logging ──────────────────────────────────────────────────────────────
def get_next_iteration():
    if not RESULTS_CSV.exists():
        return 1
    df = pd.read_csv(RESULTS_CSV)
    return int(df['iteration'].max()) + 1 if len(df) > 0 else 1

def append_result(iteration, split, metrics, notes=""):
    row = {
        'iteration':    iteration,
        'timestamp':    datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'split':        split,
        'return_pct':   round(metrics['total_return'] * 100, 3),
        'sharpe':       round(metrics['sharpe_ratio'], 4),
        'max_drawdown': round(metrics['max_drawdown'] * 100, 3),
        'win_rate_pct': round(metrics['win_rate'] * 100, 3),
        'volatility':   round(metrics['volatility'] * 100, 3),
        'num_trades':   metrics['num_trades'],
        'final_value':  round(metrics['final_value'], 2),
        'notes':        notes,
    }
    df = pd.DataFrame([row])
    write_header = not RESULTS_CSV.exists()
    df.to_csv(RESULTS_CSV, mode='a', header=write_header, index=False)
    print(f"  >> {split.upper()} logged to {RESULTS_CSV.name}")
    return row

# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    iteration = get_next_iteration()
    notes = sys.argv[1] if len(sys.argv) > 1 else ""

    print("=" * 65)
    print(f"  EVAL RUN  |  Iteration {iteration}  |  {datetime.now().strftime('%H:%M:%S')}")
    print("=" * 65)

    # Load strategy
    try:
        EnhancedStrategy = load_strategy()
        print("  strategy.py loaded OK")
    except Exception as e:
        print(f"[FATAL] Cannot load strategy.py: {e}")
        traceback.print_exc()
        sys.exit(1)

    # Load FinBERT
    print("\nLoading FinBERT...")
    device = 0 if torch.cuda.is_available() else -1
    finbert = hf_pipeline(
        "sentiment-analysis",
        model=FINBERT_MODEL,
        tokenizer=FINBERT_MODEL,
        device=device,
        return_all_scores=True,
    )
    print(f"  FinBERT ready ({'GPU' if device >= 0 else 'CPU'})")

    # Load infra (cached after first run)
    prices_dev, earnings_dev, analytics_dev = load_infra('dev', EnhancedStrategy)
    prices_val, earnings_val, analytics_val = load_infra('val', EnhancedStrategy)

    results = {}

    for split, prices, earnings, analytics in [
        ('dev', prices_dev, earnings_dev, analytics_dev),
        ('val', prices_val, earnings_val, analytics_val),
    ]:
        print(f"\n{'-'*40}")
        print(f"  Running {split.upper()} split...")
        print(f"{'-'*40}")
        try:
            strat = EnhancedStrategy(finbert)
            strat.set_data(prices, earnings)
            strat.universe_analytics_df = analytics   # inject cached analytics
            raw = strat.evaluate()
            metrics = calculate_metrics(raw)
            row = append_result(iteration, split, metrics, notes)
            results[split] = metrics

            print(f"\n  {split.upper()} RESULTS:")
            print(f"    Return:      {metrics['total_return']:>8.2%}")
            print(f"    Sharpe:      {metrics['sharpe_ratio']:>8.3f}")
            print(f"    Max Drawdown:{metrics['max_drawdown']:>8.2%}")
            print(f"    Win Rate:    {metrics['win_rate']:>8.2%}")
            print(f"    Trades:      {metrics['num_trades']:>8}")
            print(f"    Final Value: ${metrics['final_value']:>12,.2f}")

        except Exception as e:
            print(f"  [ERROR] {split.upper()} eval failed: {e}")
            traceback.print_exc()

    # Save versioned copy of strategy
    version_file = VERSIONS_DIR / f'strategy_v{iteration}.py'
    shutil.copy2(STRATEGY_FILE, version_file)
    print(f"\n  Version saved -> enhanced_strat/auto_versions/strategy_v{iteration}.py")

    print(f"\n{'='*65}")
    print(f"  Done. Results written to {RESULTS_CSV.name}")
    if 'dev' in results and 'val' in results:
        print(f"\n  SUMMARY  iteration={iteration}")
        print(f"  DEV  sharpe={results['dev']['sharpe_ratio']:.3f}  return={results['dev']['total_return']:.2%}")
        print(f"  VAL  sharpe={results['val']['sharpe_ratio']:.3f}  return={results['val']['total_return']:.2%}")
    print("=" * 65)


if __name__ == '__main__':
    main()
