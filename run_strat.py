#!/usr/bin/env python3
"""
run_strat.py — CLI runner for EnhancedStrategy files.

Usage:
    python run_strat.py enhanced_strat/v1_lowvol_lowliq.py
    python run_strat.py v1_lowvol_lowliq                        # auto-resolves from enhanced_strat/
    python run_strat.py v1_lowvol_lowliq --split val
    python run_strat.py v1_lowvol_lowliq --finbert
    python run_strat.py v1_lowvol_lowliq --data-dir /path/to/INF2006_Data_Students
"""

import argparse
import importlib.util
import shutil
import tempfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

# ─── Constants ────────────────────────────────────────────────────────────────

STARTING_CASH = 100_000
SCRIPT_DIR    = Path(__file__).parent
DEFAULT_DATA_DIR = SCRIPT_DIR / 'INF2006_Data_Students'
STRAT_DIR     = SCRIPT_DIR / 'enhanced_strat'

# ─── BaseStrategy ─────────────────────────────────────────────────────────────

class BaseStrategy:
    """
    Minimal base required by all EnhancedStrategy files.
    Enhanced strategies override clean_data, calculate_analytics, evaluate, etc.
    """

    def __init__(self, finbert_pipeline=None):
        self.finbert_pipeline  = finbert_pipeline
        self.llm_cache         = {}
        self.llm_cache_hits    = 0
        self.llm_cache_misses  = 0
        self.prices            = None
        self.earnings          = None

    def set_data(self, prices_df, earnings_df):
        print("Cleaning and preprocessing data...")
        self.prices, self.earnings = self.clean_data(prices_df, earnings_df)
        print(f"Data ready: {len(self.prices):,} price records")

    def clean_data(self, prices_df, earnings_df):
        prices   = prices_df.copy().drop_duplicates()
        earnings = earnings_df.copy().drop_duplicates()
        prices['date']   = pd.to_datetime(prices['date'])
        earnings['date'] = pd.to_datetime(earnings['date'])
        return prices, earnings

# ─── Data loading ─────────────────────────────────────────────────────────────

def _load_parquet(data_dir: Path, name: str) -> pd.DataFrame:
    """Copy to temp dir first to avoid Windows long-path issues, then read."""
    src = data_dir / f'{name}.parquet'
    if not src.exists():
        raise FileNotFoundError(f"Data file not found: {src}")
    tmp = Path(tempfile.gettempdir()) / f'{name}.parquet'
    shutil.copy2(src, tmp)
    return pd.read_parquet(tmp)

# ─── Metrics ──────────────────────────────────────────────────────────────────

def calculate_metrics(results, starting_cash=STARTING_CASH):
    history_df = pd.DataFrame(results['portfolio_history'])
    history_df['date'] = pd.to_datetime(history_df['date'])
    trades_df = pd.DataFrame(results['trades'])

    final_value  = results['final_portfolio']['total_value']
    total_return = (final_value - starting_cash) / starting_cash

    # Portfolio history is weekly; notebook uses sqrt(252) for consistency with
    # the rest of the codebase, so we match that convention here.
    history_df['period_return'] = history_df['portfolio_value'].pct_change()
    mean_ret = history_df['period_return'].mean()
    std_ret  = history_df['period_return'].std()
    sharpe_ratio = (mean_ret / std_ret * np.sqrt(252)) if std_ret > 0 else 0

    peak        = history_df['portfolio_value'].cummax()
    drawdown    = (history_df['portfolio_value'] - peak) / peak
    max_drawdown = drawdown.min()
    volatility  = std_ret * np.sqrt(252) if std_ret > 0 else 0

    win_rate = 0
    if not trades_df.empty and 'action' in trades_df.columns:
        buy_trades  = trades_df[trades_df['action'] == 'BUY']
        sell_trades = trades_df[trades_df['action'] == 'SELL']
        profitable, total = 0, 0
        for _, sell in sell_trades.iterrows():
            prior_buys = buy_trades[buy_trades['ticker'] == sell['ticker']]
            if not prior_buys.empty:
                if sell['value'] > prior_buys.iloc[-1]['value']:
                    profitable += 1
                total += 1
        win_rate = profitable / total if total > 0 else 0

    return {
        'final_value':  final_value,
        'total_return': total_return,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'volatility':   volatility,
        'win_rate':     win_rate,
        'num_trades':   len(trades_df),
    }

def _print_metrics(metrics, strat_name, split):
    w = 52
    print()
    print("=" * w)
    print(f"  {strat_name}")
    print(f"  Split: {split.upper()}")
    print("=" * w)
    print(f"  {'Final Portfolio Value':<24} ${metrics['final_value']:>12,.2f}")
    print(f"  {'Total Return':<24} {metrics['total_return']:>12.2%}")
    print(f"  {'Sharpe Ratio':<24} {metrics['sharpe_ratio']:>12.2f}")
    print(f"  {'Max Drawdown':<24} {metrics['max_drawdown']:>12.2%}")
    print(f"  {'Annualised Volatility':<24} {metrics['volatility']:>12.2%}")
    print(f"  {'Win Rate':<24} {metrics['win_rate']:>12.1%}")
    print(f"  {'Total Trades':<24} {metrics['num_trades']:>12,}")
    print("=" * w)
    print()

# ─── Strategy loader ──────────────────────────────────────────────────────────

def _resolve_path(strat_arg: str) -> Path:
    """Accept a file path, bare name, or name without .py extension."""
    p = Path(strat_arg)
    for candidate in [p, p.with_suffix('.py'),
                      STRAT_DIR / strat_arg,
                      STRAT_DIR / (strat_arg + '.py')]:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(
        f"Cannot find strategy {strat_arg!r}.\n"
        f"Available in enhanced_strat/:\n"
        + '\n'.join(f"  {f.stem}" for f in sorted(STRAT_DIR.glob('*.py'))
                    if not f.name.startswith('_'))
    )

def _load_strategy_class(path: Path):
    """
    Import a strategy .py file and return its EnhancedStrategy class.
    BaseStrategy and STARTING_CASH are injected as module-level globals
    because the strategy files expect them from the notebook namespace.
    """
    spec   = importlib.util.spec_from_file_location('_strat_module', path)
    module = importlib.util.module_from_spec(spec)
    module.BaseStrategy   = BaseStrategy
    module.STARTING_CASH  = STARTING_CASH
    spec.loader.exec_module(module)
    if not hasattr(module, 'EnhancedStrategy'):
        raise AttributeError(f"No EnhancedStrategy class found in {path}")
    return module.EnhancedStrategy

# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Run an EnhancedStrategy .py file and show performance metrics.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        'strat',
        help='Path to strategy .py file, or name (e.g. v1_lowvol_lowliq)',
    )
    parser.add_argument(
        '--split', choices=['dev', 'val'], default='dev',
        help='Data split to evaluate on (default: dev)',
    )
    parser.add_argument(
        '--data-dir', type=Path, default=DEFAULT_DATA_DIR,
        metavar='DIR',
        help=f'Path to INF2006_Data_Students directory (default: auto-detect)',
    )
    parser.add_argument(
        '--finbert', action='store_true',
        help='Load FinBERT for sentiment analysis (downloads ~420 MB on first run)',
    )
    parser.add_argument(
        '--verbose', action='store_true',
        help='Verbose backtest output',
    )
    args = parser.parse_args()

    # ── Resolve strategy file ──────────────────────────────────────────────
    strat_path = _resolve_path(args.strat)
    print(f"\nStrategy : {strat_path.name}")
    print(f"Split    : {args.split}")
    print(f"Data dir : {args.data_dir}")

    # ── FinBERT ───────────────────────────────────────────────────────────
    finbert_pipeline = None
    if args.finbert:
        try:
            import torch
            from transformers import pipeline as hf_pipeline
            device = 0 if torch.cuda.is_available() else -1
            print(f"\nLoading FinBERT on {'GPU' if device >= 0 else 'CPU'}...")
            finbert_pipeline = hf_pipeline(
                'text-classification',
                model='ProsusAI/finbert',
                device=device,
                return_all_scores=True,
            )
            print("FinBERT ready.")
        except ImportError:
            print("WARNING: torch/transformers not installed — skipping FinBERT.")
    else:
        print("\nFinBERT: disabled (pass --finbert to enable)")

    # ── Load data ─────────────────────────────────────────────────────────
    print(f"\nLoading {args.split} data from {args.data_dir}...")
    prices   = _load_parquet(args.data_dir, f'prices_{args.split}')
    earnings = _load_parquet(args.data_dir, f'earnings_{args.split}')
    print(f"  Prices   : {len(prices):,} rows  |  {prices['ticker'].nunique()} tickers")
    print(f"  Earnings : {len(earnings):,} rows")

    # ── Load & run strategy ───────────────────────────────────────────────
    print(f"\nLoading {strat_path.name}...")
    EnhancedStrategy = _load_strategy_class(strat_path)
    strategy = EnhancedStrategy(finbert_pipeline=finbert_pipeline)

    strategy.set_data(prices, earnings)
    results = strategy.evaluate(verbose=args.verbose)

    # ── Metrics ───────────────────────────────────────────────────────────
    metrics = calculate_metrics(results, starting_cash=STARTING_CASH)
    _print_metrics(metrics, strat_path.stem, args.split)


if __name__ == '__main__':
    main()
