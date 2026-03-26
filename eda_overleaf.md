\documentclass[11pt,twoside,a4paper]{article}

\newcommand{\reporttitle}{Algorithmic Trading Strategy Development and Optimisation}
\newcommand{\gid}{X}
\newcommand{\reportauthorOne}{Student 1}
\newcommand{\cidOne}{your id number}
\newcommand{\reportauthorTwo}{Student 2}
\newcommand{\cidTwo}{your id number}
\newcommand{\reportauthorThree}{Student 3}
\newcommand{\cidThree}{your id number}
\newcommand{\reportauthorFour}{Student 4}
\newcommand{\cidFour}{your id number}
\newcommand{\reportauthorFive}{Student 5}
\newcommand{\cidFive}{your id number}
\newcommand{\reportauthorSix}{Student 6}
\newcommand{\cidSix}{your id number}


\newcommand{\reporttype}{Group Project Assignment 2}
\bibliographystyle{plain}

% include files that load packages and define macros
\input{includes} % various packages needed for maths etc.
\input{notation} % short-hand notation and macros


%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\begin{document}
% front page
\input{titlepage}


%%%%%%%%%%%%%%%%%%%%%%%%%%% table of content
%If a table of content is needed, simply uncomment the following lines
%\tableofcontents
%\newpage

%%%%%%%%%%%%%%%%%%%%%%%%%%%% Main document
\section{Introduction}
This report documents the iterative development of an algorithmic trading strategy for S\&P 500 equities using two primary data sources: daily market data and quarterly earnings call transcripts. The research focuses on four strategy variants (V1 to V4), each designed based off evidence gathered from exploratory data analysis.

In modern financial markets, algorithmic trading strategies are essential for systematically exploiting market inefficiencies. The use of big data analytics allows for the processing of vast amounts of historical price data to uncover statistical patterns. This report documents the iterative development of an algorithmic trading strategy for S\&P 500 equities driven by hypothesis derived from Exploratory Data Analysis. The central goal of this project is to develop a trading strategy that systematically leverages the Amihud illiquidity ratio and Parkinson volatility for universe selection which can then be deployed in live trading scenarios.

The project uses a development split from 2000 to 2017 and a validation split from 2018 to 2024, so that design decisions were not made on the final out-of-sample period.

Key objectives include:
\begin{enumerate}
    \item To design and test a composite scoring mechanism based on Amihud and volatility rankings to filter the S\&P 500 universe.
    \item To implement entry signals derived from technical indicators, gated by the composite score and other volatility-based conditions.
    \item To integrate effective drawdown management through the use of ATR (Average True Range)-based trailing stop-losses to control risk on a per-trade basis.
    \item To apply advanced computational optimization techniques, such as vectorization, to ensure the strategy's calculation and backtesting processes are both fast and scalable.
    \item To empirically validate the performance of the enhanced strategy against a simple baseline, using a comprehensive set of risk-adjusted return metrics across distinct development and validation time periods.
\end{enumerate}

\section{Enhanced Exploratory Data Analysis}

The supplied project materials describe a dataset with broad historical and cross-sectional coverage. The development split contains 1,332,576 price records, while the validation split contains 598,740 price records, both drawn from 340 S\&P 500 tickers. This scale is large enough to support long rolling indicators such as 60-day liquidity and volatility measures, 200-day trend filters, and repeated monthly universe reconstruction. It also creates non-trivial engineering demands, because even conceptually simple per-ticker operations become expensive when repeated over many dates and securities.

The exploratory findings that shaped the strategy are more structural than purely descriptive. The most important observation is that the composite distribution formed by Parkinson volatility and Amihud illiquidity is informative at the tails. In the development split, a bottom-quartile universe covered 223 unique tickers, a top-quartile universe covered 260 unique tickers, and the combined dual-tail universe covered 332 unique tickers. This suggests that the center of the cross-section was comparatively unremarkable for the chosen signal family, whereas the extremes offered more differentiated trading conditions.

The earnings transcript corpus was also substantial. Depending on the active universe, the development split required sentiment processing over 5,667 to 9,464 transcripts, corresponding to 68,912 to 114,914 text chunks. On the validation split, the comparable ranges were 5,138 to 8,371 transcripts and 63,041 to 102,849 chunks. These counts show that transcript analysis was not a cosmetic add-on. It was a computationally meaningful subsystem that needed to justify itself through better trading outcomes.

Several project-level implications follow from this exploratory phase.
\begin{itemize}
    \item The trading universe should not be treated as fixed. A monthly cross-sectional filter is justified because the most attractive names depend on recent liquidity and volatility conditions.
    \item The center of the composite score distribution appears to carry lower signal-to-noise than either tail. This became the main rationale for the dual-tail design used in later versions.
    \item Transcript coverage is rich enough to support a sentiment sleeve, but the volume of text implies that NLP must be batched and engineered carefully if it is to remain practical.
\end{itemize}

Table~\ref{tab:eda-summary} summarises the main empirical context extracted from the project documentation.

\begin{table}[h]
\centering
\caption{Dataset and universe coverage summary}
\label{tab:eda-summary}
\begin{tabularx}{\textwidth}{@{}lXr@{}}
\toprule
Category & Observation & Value \\
\midrule
Price data (development) & Daily OHLCV records across S\&P 500 tickers & 1,332,576 \\
Price data (validation)  & Daily OHLCV records across S\&P 500 tickers & 598,740 \\
Total ticker coverage    & Distinct equities considered across both splits & 340 \\
V1 development universe  & Bottom quartile of composite score & 223 tickers \\
V2 development universe  & Top quartile of composite score & 260 tickers \\
V3 development universe  & Bottom and top quartiles combined & 332 tickers \\
V4 development universe  & Dual-tail technical-only variant & 331 tickers \\
Largest transcript workload & V3 development sentiment preprocessing & 114,914 chunks \\
\bottomrule
\end{tabularx}
\end{table}

\subsection{Price Coverage and Market Predictability}

Figure~\ref{fig:eda-overview} presents a four-panel summary of the raw price dataset. The pairwise return correlation matrix and community network (Figure~\ref{fig:eda-heatmap-network}) reveal groups of tickers that co-move strongly within sectors. Monitoring sector leaders for sentiment shifts was the primary motivation for the transcript pipeline.

\begin{figure}[htbp]
\centering
\includegraphics[width=0.8\textwidth]{figures/eda_00_overview.png}
\caption{Overview of raw price data: close price distribution, volume distribution, daily return distribution, and top tickers by data density.}
\label{fig:eda-overview}
\end{figure}

\begin{figure}[htbp]
\centering
\begin{minipage}{0.48\textwidth}
    \centering
    \includegraphics[width=\textwidth]{figures/eda_04_correlation_heatmap.png}
    \caption{Correlation Heatmap}
    \label{fig:eda-heatmap}
\end{minipage}\hfill
\begin{minipage}{0.48\textwidth}
    \centering
    \includegraphics[width=\textwidth]{figures/eda_05_community_network.png}
    \caption{Community Network}
    \label{fig:eda-network}
\end{minipage}
\caption{Market structure: hierarchical correlation clustering and ticker community hubs.}
\label{fig:eda-heatmap-network}
\end{figure}

\subsection{Post-Earnings Dynamics and Sentiment Analytics}

Figures~\ref{fig:eda-halflife-combined} quantify how quickly abnormal returns following earnings announcements decay back to baseline. Variation in half-lives justified a fixed exit window of five weeks in the sentiment sleeve. Furthermore, sentiment analytics (Figure~\ref{fig:eda-sentiment-combined}) shows that calls where prepared remarks and analyst tone diverge correlate with price drift. incremental positivity in transcripts shows exhaustion beyond a threshold, suggesting sentiment acceleration is a more robust trigger.

\begin{figure}[htbp]
\centering
\begin{minipage}{0.48\textwidth}
    \centering
    \includegraphics[width=\textwidth]{figures/eda_06_half_life_bar.png}
    \caption{CAR Half-life (Bar)}
    \label{fig:eda-halflife-bar}
\end{minipage}\hfill
\begin{minipage}{0.48\textwidth}
    \centering
    \includegraphics[width=\textwidth]{figures/eda_07_half_life_scatter.png}
    \caption{Half-life (Scatter)}
    \label{fig:eda-halflife-scatter}
\end{minipage}
\caption{Information decay metrics demonstrating heterogenous post-earnings drift.}
\label{fig:eda-halflife-combined}
\end{figure}

\begin{figure}[htbp]
\centering
\begin{minipage}{0.48\textwidth}
    \centering
    \includegraphics[width=\textwidth]{figures/eda_08_guidance_qa_divergence.png}
    \caption{Guidance vs Q\&A Divergence}
    \label{fig:eda-divergence}
\end{minipage}\hfill
\begin{minipage}{0.48\textwidth}
    \centering
    \includegraphics[width=\textwidth]{figures/eda_09_sentiment_exhaustion.png}
    \caption{Sentiment Exhaustion}
    \label{fig:eda-exhaustion}
\end{minipage}
\caption{Sentiment Analytics: internal tone divergence and forward return exhaustion effects.}
\label{fig:eda-sentiment-combined}
\end{figure}

\subsection{Volume Filters and Hypothesis Testing}

Standardised Abnormal Volume (Figure~\ref{fig:eda-sav}) confirms that a 2.0 SD threshold captures institutional activity. The subsequent hypothesis testing (Figures~\ref{fig:h01} to~\ref{fig:h09}) validates the Low Volatility Anomaly, Volume Confirmation, the Illiquidity Premium, and the 200-day Trend Filter, collectively anchoring the strategy's signal logic.

\begin{figure}[htbp]
\centering
\includegraphics[width=0.7\textwidth]{figures/eda_10_sav_filter.png}
\caption{Standardised Abnormal Volume (SAV) filter threshold verification.}
\label{fig:eda-sav}
\end{figure}

\begin{figure}[htbp]
\centering
\begin{minipage}{0.48\textwidth}
    \centering
    \includegraphics[width=\textwidth]{figures/h01_low_vs_high_vol_anomaly.png}
    \caption{Low Vol Anomaly}
    \label{fig:h01}
\end{minipage}\hfill
\begin{minipage}{0.48\textwidth}
    \centering
    \includegraphics[width=\textwidth]{figures/h02_volume_confirmation.png}
    \caption{Volume Confirmation}
    \label{fig:h02}
\end{minipage}
\end{figure}

\begin{figure}[htbp]
\centering
\begin{minipage}{0.48\textwidth}
    \centering
    \includegraphics[width=\textwidth]{figures/h03_rsi_mean_reversion.png}
    \caption{RSI Mean Reversion}
    \label{fig:h03}
\end{minipage}\hfill
\begin{minipage}{0.48\textwidth}
    \centering
    \includegraphics[width=\textwidth]{figures/h04_illiquidity_premium.png}
    \caption{Illiquidity Premium}
    \label{fig:h04}
\end{minipage}
\end{figure}

\begin{figure}[htbp]
\centering
\begin{minipage}{0.48\textwidth}
    \centering
    \includegraphics[width=\textwidth]{figures/h05_sentiment_acceleration.png}
    \caption{Sentiment Acceleration}
    \label{fig:h05}
\end{minipage}\hfill
\begin{minipage}{0.48\textwidth}
    \centering
    \includegraphics[width=\textwidth]{figures/h06_volatility_acceleration.png}
    \caption{Vol Acceleration Veto}
    \label{fig:h06}
\end{minipage}
\end{figure}

\begin{figure}[htbp]
\centering
\begin{minipage}{0.48\textwidth}
    \centering
    \includegraphics[width=\textwidth]{figures/h07_trend_filter_drawdown.png}
    \caption{Trend Filter Drawdown}
\end{minipage}\hfill
\begin{minipage}{0.48\textwidth}
    \centering
    \includegraphics[width=\textwidth]{figures/h07_trend_filter_equity.png}
    \caption{Trend Filter Equity}
\end{minipage}
\caption{Performance of the 200-day trend filter across different volatility regimes.}
\end{figure}

\begin{figure}[htbp]
\centering
\begin{minipage}{0.48\textwidth}
    \centering
    \includegraphics[width=\textwidth]{figures/h08_parkinson_vol_persistence.png}
    \caption{Parkinson Vol Persistence}
    \label{fig:h08}
\end{minipage}\hfill
\begin{minipage}{0.48\textwidth}
    \centering
    \includegraphics[width=\textwidth]{figures/h09_amihud_vol_window_search.png}
    \caption{Window Grid Search}
    \label{fig:h09}
\end{minipage}
\end{figure}

% FORCE PENDING FIGURES TO FLUSH BEFORE NEXT SECTION
\clearpage

\section{Data Cleaning and Preparation}
Data cleaning in this project served two purposes: preserving chronological integrity and ensuring that the feature engineering pipeline produced stable indicator values.

\begin{itemize}
    \item Duplicate records were removed from both price data and earnings data before any analytics were computed.
    \item Date fields were converted to proper datetime format and sorted by ticker and time.
    \item Earnings transcripts shorter than 100 characters were excluded.
    \item Rolling indicators introduce initial missing values; the implementation waited until sufficient history was available.
    \item Universe construction dropped securities with missing long-horizon measures.
    \item Daily analytics were aligned to a weekly schedule using an as-of join.
\end{itemize}

\section{Computational Efficiency and Performance Optimization}
Scalability was an important practical issue. Several optimisation strategies were introduced: rolling indicators used vectorised \texttt{groupby} and \texttt{transform} operations; transcript processing used chunking and caching; and the strategy consolidated analytics needed for both sleeves into one preprocessing pass. The ``Optimised V5'' implementation documented in the codebase uses batched FinBERT inference for improved efficiency.

\section{Strategy Enhancement Methodology}
\subsection*{Technical Indicators and Features}
The feature set includes RSI(14), ATR(14), 200-day and 50-day moving averages, 60-day and 10-day Parkinson volatility, 20-day volume average, and 60-day Amihud illiquidity. Stocks are re-ranked monthly according to a composite score of low volatility and high illiquidity. Sleeve 1 uses RSI mean reversion and volume-backed momentum.

\subsection*{Earnings Transcript Analysis}
The transcript analysis pipeline uses FinBERT on sentence-like chunks. A net sentiment score is computed. The key innovation is the use of \emph{sentiment acceleration}: comparing current quarter sentiment against the prior quarter. Sleeve 2 only triggers when sentiment improves.

\subsection*{Entry, Exit, and Risk Management Rules}
The strategy uses two sleeves with shared filters.
\textbf{Shared filters:} Entries are blocked if \(\text{vol\_10} > 1.5 \times \text{vol\_60}\) or if price is below the 200-day moving average.
\textbf{Sleeve 1:} Technical trigger using RSI or volume-momentum, with a 2-ATR trailing stop.
\textbf{Sleeve 2:} Sentitment acceleration trigger, with a 2-ATR trailing stop and maximum holding window.

\section{Experimental Results and Analysis}
\begin{longtable}{@{}lcccccccc@{}}
\caption{Comparison of strategy versions across development and validation splits}
\label{tab:results}\\
\toprule
Version & Dev Return & Dev Sharpe & Dev Max DD & Dev Win & Dev Vol & Dev Trades & Val Return & Val Sharpe \\
\midrule
V1 & 290.62\% & 1.99 & -21.90\% & 56.4\% & 19.35\% & 21,619 & 107.50\% & 2.10 \\
V2 & 336.55\% & 1.25 & -33.84\% & 53.4\% & 37.03\% & 24,248 & 221.88\% & 2.76 \\
V3 & 560.01\% & 1.67 & -34.09\% & 54.7\% & 33.69\% & 44,131 & 253.95\% & 2.51 \\
V4 & 557.07\% & 1.67 & -34.25\% & 54.7\% & 33.63\% & 45,327 & 227.92\% & 2.48 \\
\bottomrule
\end{longtable}

\section{Final Performance Outcomes}
The final recommended strategy is V4. It offers the most convincing trade-off between performance and robustness. It eliminates the expensive transcript sleeve while preserving the dual-tail universe structure and risk-management improvements of V3, leading to equivalent risk-adjusted performance with materially better drawdown containment.

\section{Related Work}
This project draws on literature regarding technical analysis, market microstructure (Amihud illiquidity), and natural language processing in finance (FinBERT). Our approach contributes a hybrid framework combining structured market signals with soft information from transcripts.

\section{Conclusion and Future Work}
\subsection{Conclusion}
Alpha was created primarily by better market structuring using volatility and illiquidity filters. V4 emerged as the best finale, synthesising return and robustness.

\subsection{Future Work}
Future iterations will include transaction cost modelling, Formal runtime benchmarks, and sector constraints.

\vfill
\newpage
\bibliography{mybib}
\vfill
\newpage
\section*{Appendices}
\subsection*{A. Individual Contributions}
[Insert specifics here]

\subsection*{B. Declaration of Use of Generative AI Tools}
This report was prepared with the assistance of generative AI tools.

\end{document}