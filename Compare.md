======================================================================
ENHANCED STRATEGY V1 EVALUATION (Development)
======================================================================

[DEV SPLIT - For Development]
Loading DEV split data...
Cleaning and preprocessing data...
Data ready: 1,332,576 price records
Running evaluation...
Computing advanced technicals using fully vectorized Pandas groupby...
  Analytics computed: 1,332,576 rows for 340 tickers
Aligning analytics to weekly schedule...
  Weekly analytics aligned: 276,363 rows
Precomputing monthly universe snapshots...
  Universe snapshots ready for 214 months
  Universe covers 223 unique tickers across all months (out of 340 total)
Precomputing transcript sentiment in batched mode...
  Filtering to 223 universe tickers (5,667 transcripts)
  Model converted to FP16 for inference
  Running FinBERT on 68,912 chunks (batch_size=128)...
  Precomputed 5,667 transcript sentiments from 68,912 chunks
Running fast backtest loop...
You seem to be using the pipelines sequentially on GPU. In order to maximize efficiency please use a dataset
Return: 290.62%
Sharpe Ratio: 1.99
Max Drawdown: -21.90%
Win Rate: 56.4%
Volatility: 19.35%
Total Trades: 21,619

[VAL SPLIT - Final Performance]
Loading VAL split data...
Cleaning and preprocessing data...
Data ready: 598,740 price records
Running evaluation...
Computing advanced technicals using fully vectorized Pandas groupby...
  Analytics computed: 598,740 rows for 340 tickers
Aligning analytics to weekly schedule...
  Weekly analytics aligned: 124,100 rows
Precomputing monthly universe snapshots...
  Universe snapshots ready for 82 months
  Universe covers 210 unique tickers across all months (out of 340 total)
Precomputing transcript sentiment in batched mode...
  Filtering to 210 universe tickers (5,138 transcripts)
  Model converted to FP16 for inference
  Running FinBERT on 63,041 chunks (batch_size=128)...
  Precomputed 5,138 transcript sentiments from 63,041 chunks
Running fast backtest loop...
Return: 107.50%
Sharpe Ratio: 2.10
Max Drawdown: -15.39%

======================================================================
ENHANCED STRATEGY V18 EVALUATION (Development)
======================================================================

should be higher sharpe than v1 (FALSE) BUT is v high returns can cut drawdown easily to increase sharpe


[DEV SPLIT - For Development]
Return: 542.76%
Sharpe Ratio: 1.82
Max Drawdown: -24.13%
Win Rate: 54.9%
Volatility: 29.80%
Total Trades: 23,952

[VAL SPLIT - Final Performance]
Return: 185.93%
Sharpe Ratio: 1.73
Max Drawdown: -40.48%

======================================================================
ENHANCED STRATEGY V2 EVALUATION (Development)
======================================================================

[DEV SPLIT - For Development]
Loading DEV split data...
Cleaning and preprocessing data...
Data ready: 1,332,576 price records
Running evaluation...
Computing advanced technicals using fully vectorized Pandas groupby...
  Analytics computed: 1,332,576 rows for 340 tickers
Aligning analytics to weekly schedule...
  Weekly analytics aligned: 276,363 rows
Precomputing monthly universe snapshots...
  Universe snapshots ready for 214 months
  Universe covers 260 unique tickers across all months (out of 340 total)
Precomputing transcript sentiment in batched mode...
  Filtering to 260 universe tickers (8,210 transcripts)
  Model converted to FP16 for inference
  Running FinBERT on 99,672 chunks (batch_size=128)...
  Precomputed 8,210 transcript sentiments from 99,672 chunks
Running fast backtest loop...
You seem to be using the pipelines sequentially on GPU. In order to maximize efficiency please use a dataset
Return: 336.55%
Sharpe Ratio: 1.25
Max Drawdown: -33.84%
Win Rate: 53.4%
Volatility: 37.03%
Total Trades: 24,248


[VAL SPLIT - Final Performance]
Loading VAL split data...
Cleaning and preprocessing data...
Data ready: 598,740 price records
Running evaluation...
Computing advanced technicals using fully vectorized Pandas groupby...
  Analytics computed: 598,740 rows for 340 tickers
Aligning analytics to weekly schedule...
  Weekly analytics aligned: 124,100 rows
Precomputing monthly universe snapshots...
  Universe snapshots ready for 82 months
  Universe covers 233 unique tickers across all months (out of 340 total)
Precomputing transcript sentiment in batched mode...
  Filtering to 233 universe tickers (6,118 transcripts)
  Model converted to FP16 for inference
  Running FinBERT on 75,145 chunks (batch_size=128)...
  Precomputed 6,118 transcript sentiments from 75,145 chunks
Running fast backtest loop...
Return: 221.88%
Sharpe Ratio: 2.76
Max Drawdown: -17.53%

======================================================================
ENHANCED STRATEGY V18.2 EVALUATION (Development)
======================================================================

should be higher returns than V2 (FALSE) its jsut not good

[DEV SPLIT - For Development]
Return: 202.05%
Sharpe Ratio: 1.45
Max Drawdown: -25.42%
Win Rate: 55.2%
Volatility: 22.17%
Total Trades: 22,099

[VAL SPLIT - Final Performance]
Return: 94.56%
Sharpe Ratio: 2.24
Max Drawdown: -8.83%

======================================================================
ENHANCED STRATEGY V3 EVALUATION (Development)
======================================================================
[DEV SPLIT - For Development]
Loading DEV split data...
Cleaning and preprocessing data...
Data ready: 1,332,576 price records
Running evaluation...
Computing advanced technicals using fully vectorized Pandas groupby...
  Analytics computed: 1,332,576 rows for 340 tickers
Aligning analytics to weekly schedule...
  Weekly analytics aligned: 276,363 rows
Precomputing monthly universe snapshots...
  Universe snapshots ready for 214 months
  Universe covers 332 unique tickers across all months (out of 340 total)
Precomputing transcript sentiment in batched mode...
  Filtering to 332 universe tickers (9,464 transcripts)
  Model converted to FP16 for inference
  Running FinBERT on 114,914 chunks (batch_size=128)...
  Precomputed 9,464 transcript sentiments from 114,914 chunks
Running fast backtest loop...
You seem to be using the pipelines sequentially on GPU. In order to maximize efficiency please use a dataset
Return: 560.01%
Sharpe Ratio: 1.67
Max Drawdown: -34.09%
Win Rate: 54.7%
Volatility: 33.69%
Total Trades: 44,131

[VAL SPLIT - Final Performance]
Loading VAL split data...
Cleaning and preprocessing data...
Data ready: 598,740 price records
Running evaluation...
Computing advanced technicals using fully vectorized Pandas groupby...
  Analytics computed: 598,740 rows for 340 tickers
Aligning analytics to weekly schedule...
  Weekly analytics aligned: 124,100 rows
Precomputing monthly universe snapshots...
  Universe snapshots ready for 82 months
  Universe covers 332 unique tickers across all months (out of 340 total)
Precomputing transcript sentiment in batched mode...
  Filtering to 332 universe tickers (8,371 transcripts)
  Model converted to FP16 for inference
  Running FinBERT on 102,849 chunks (batch_size=128)...
  Precomputed 8,371 transcript sentiments from 102,849 chunks
Running fast backtest loop...
Return: 253.95%
Sharpe Ratio: 2.51
Max Drawdown: -22.94%

======================================================================
ENHANCED STRATEGY V4 EVALUATION (Development)
======================================================================

[DEV SPLIT - For Development]
Loading DEV split data...
Cleaning and preprocessing data...
Data ready: 1,332,576 price records
Running evaluation...
Computing advanced technicals using fully vectorized Pandas groupby...
  Analytics computed: 1,332,576 rows for 340 tickers
Aligning analytics to weekly schedule...
  Weekly analytics aligned: 276,363 rows
Precomputing monthly universe snapshots...
  Universe snapshots ready for 213 months
  Universe covers 331 unique tickers across all months (out of 340 total)
Running fast backtest loop...
Return: 557.07%
Sharpe Ratio: 1.67
Max Drawdown: -34.25%
Win Rate: 54.7%
Volatility: 33.63%
Total Trades: 45,327


[VAL SPLIT - Final Performance]
Loading VAL split data...
Cleaning and preprocessing data...
Data ready: 598,740 price records
Running evaluation...
Computing advanced technicals using fully vectorized Pandas groupby...
  Analytics computed: 598,740 rows for 340 tickers
Aligning analytics to weekly schedule...
  Weekly analytics aligned: 124,100 rows
Precomputing monthly universe snapshots...
  Universe snapshots ready for 81 months
  Universe covers 331 unique tickers across all months (out of 340 total)
Running fast backtest loop...
Return: 227.92%
Sharpe Ratio: 2.48
Max Drawdown: -17.30%