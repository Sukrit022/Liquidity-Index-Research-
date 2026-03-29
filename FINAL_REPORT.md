# Liquidity Index Ablation Study Final Report

Generated from `results/leaderboard.csv`, `results/metrics_registry.csv`, and `results/MASTER_LEADERBOARD.csv`.

## 1. Abstract

This report closes the liquidity index ablation study by comparing 22 approved models across statistical, classical machine learning, recurrent neural network, advanced deep learning, and ensemble families on a fixed chronological holdout. The best model on the approved leaderboard is `Weighted Ensemble (Top-5 Overall)` with RMSE=0.1564, MAE=0.0937, MAPE=19.49%, and R^2=0.7506.
The runner-up `Ridge(alpha=1.0)` trails by only 0.000464 RMSE, and the third-place `LinearRegression` is only 0.000044 behind the runner-up, so the practical difference among the top three models is negligible even though the ranking is stable.
The dominant empirical pattern is that persistence-aware linear and statistical models beat both tree ensembles and deep neural architectures, implying that the liquidity index is largely driven by recent history and smooth local trend rather than high-order nonlinear structure.
A broader 42-model archived appendix is retained for reproducibility, but all main conclusions in this document are drawn from the approved 22-row leaderboard produced at the end of step-8.

## 2. Dataset & Problem Statement

The target is the daily `Market_Liquidity_Index` series stored in `Code/market_liquidity_index.csv`. The raw dataset contains 3365 rows, which became 3335 model-ready observations after feature warmup. The train window runs from 2011-07-13 through 2022-04-25 (2668 rows), and the test window runs from 2022-04-26 through 2024-12-31 (667 rows).
The forecasting task is one-step-ahead prediction of the liquidity index under a strict chronological split. This is a realistic market setting: the model sees only past observations, and all evaluation happens on a later period that includes post-2022 market conditions.
Missing values were handled with forward-fill then backfill. Outliers were treated with IQR clipping, which capped 17 extreme observations rather than dropping them, preserving rare stress regimes while limiting scale distortion for downstream learners.

## 3. Methodology

Feature engineering followed the approved preprocessing pipeline. Eight lag terms ([1, 2, 3, 5, 7, 14, 21, 30]), rolling means and rolling standard deviations over windows [7, 14, 30], and three calendar variables (['day_of_week', 'month', 'quarter']) were built, giving 17 features in total.
Leakage prevention was handled conservatively. The split remained chronological, all scalers were fit on the training partition only, `StandardScaler` was reserved for classical ML models, and `MinMaxScaler` was reserved for the deep-learning families. Statistical baselines were evaluated in walk-forward mode, while the supervised ML and DL families used the frozen train/test split from `artifacts/preprocessed_arrays.joblib`.
The approved model families were:
- Statistical baselines: Naive, 7-day Moving Average, ETS, ARIMA, and SARIMA.
- Classical ML: LinearRegression, Ridge, Lasso, linear SVR, RBF SVR, RandomForest, XGBoost, and LightGBM.
- Deep Learning RNN: LSTM, Bidirectional LSTM, and GRU on univariate lookback-30 sequences.
- Advanced DL: CNN-LSTM, Attention LSTM, and Temporal Transformer on multivariate lookback-30 sequences.
- Ensemble: weighted top-5 ensemble, top-3 ML simple average, and linear stacking.
All families were scored with MAE, RMSE, MAPE, R^2, and SMAPE. For the final ranking, RMSE is the primary criterion because it penalizes large forecast misses more strongly and cleanly separates the top cluster of models.
Key visual references for this section are `./plots/statistical/statistical_model_comparison.png`, `./plots/ml/rmse_comparison.png`, `./plots/leaderboard_comparison.png`, `./plots/dl/lstm_loss_curve.png`, `./plots/dl_advanced/cnn_lstm_forecast.png`, and `./plots/ensemble/ensemble_comparison.png`.

## 4. Results by Model Family

The family summary below shows the approved 22-model study. Ensemble models are best on average (mean RMSE=0.1573), but the gap to ML is small. Statistical models remain highly competitive, while both deep-learning families trail by a wide margin.

| Family | Models | Best Model | Best RMSE | Mean RMSE | Best R^2 |
|---|---:|---|---:|---:|---:|
| Ensemble | 3 | Weighted Ensemble (Top-5 Overall) | 0.1564 | 0.1573 | 0.7506 |
| ML | 8 | Ridge(alpha=1.0) | 0.1569 | 0.1951 | 0.7492 |
| Statistical | 5 | ARIMA(1, 1, 1) | 0.1584 | 0.1653 | 0.7444 |
| Advanced DL | 3 | CNN-LSTM | 0.1909 | 0.2383 | 0.6285 |
| Deep Learning RNN | 3 | Bidirectional LSTM (64 units, dropout=0.2) | 0.3532 | 0.3835 | -0.2711 |

Statistical baselines were unexpectedly strong. `ARIMA(1, 1, 1)` and its SARIMA counterpart tied at RMSE=0.1584, only about 1.23% worse than the overall winner. This indicates that the series retains substantial autocorrelation and can be forecast well with low-parameter temporal models.
Within ML, `Ridge(alpha=1.0)` and `LinearRegression` formed a near tie around RMSE=0.1569. The linear SVR and Lasso variants also stayed in the same cluster, while the nonlinear tree-based models deteriorated markedly: LightGBM finished at RMSE=0.2061, RandomForest at 0.2171, and XGBoost at 0.2207.
Advanced DL improved on the simple RNN step but still did not threaten the leading families. `CNN-LSTM` reached RMSE=0.1909 and R^2=0.6285, clearly better than the univariate RNNs but still well behind the linear, statistical, and ensemble leaders.
The RNN family was the weakest approved family by a large margin. `Bidirectional LSTM (64 units, dropout=0.2)` was still at RMSE=0.3532, and all three approved RNNs had negative R^2, which means they performed worse than a constant-mean baseline on the test set.

### Full Approved Leaderboard

| Rank | Family | Model | RMSE | MAE | MAPE (%) | SMAPE (%) | R^2 |
|---|---|---|---:|---:|---:|---:|---:|
| 1 | Ensemble | Weighted Ensemble (Top-5 Overall) | 0.1564 | 0.0937 | 19.4938 | 9.9258 | 0.7506 |
| 2 | ML | Ridge(alpha=1.0) | 0.1569 | 0.0951 | 19.6663 | 10.0263 | 0.7492 |
| 3 | ML | LinearRegression | 0.1569 | 0.0952 | 19.6784 | 10.0309 | 0.7490 |
| 4 | Ensemble | Simple Average Ensemble (Top-3 ML) | 0.1570 | 0.0948 | 19.6887 | 9.9961 | 0.7487 |
| 5 | ML | SVR(kernel='linear', C=1.0) | 0.1576 | 0.0941 | 19.7392 | 9.9506 | 0.7470 |
| 6 | ML | Lasso(alpha=0.01) | 0.1576 | 0.0951 | 19.3708 | 10.0835 | 0.7469 |
| 7 | Statistical | ARIMA(1, 1, 1) | 0.1584 | 0.0951 | 19.5804 | 10.1138 | 0.7444 |
| 8 | Statistical | SARIMA(1, 1, 1)x(0, 0, 0, 21) | 0.1584 | 0.0951 | 19.5804 | 10.1138 | 0.7444 |
| 9 | Ensemble | Stacking Ensemble (Linear Meta-Learner) | 0.1585 | 0.0961 | 19.8010 | 10.1775 | 0.7439 |
| 10 | Statistical | ETS (SimpleExpSmoothing) | 0.1593 | 0.0968 | 19.7817 | 10.2558 | 0.7416 |
| 11 | Statistical | Moving Average (7-day) | 0.1607 | 0.1040 | 20.2821 | 10.9142 | 0.7367 |
| 12 | Statistical | Naive/Persistence | 0.1895 | 0.1038 | 20.4320 | 11.2647 | 0.6339 |
| 13 | Advanced DL | CNN-LSTM | 0.1909 | 0.1366 | 24.9373 | 14.3030 | 0.6285 |
| 14 | ML | LightGBMRegressor(n_estimators=200) | 0.2061 | 0.1466 | 21.9276 | 14.3777 | 0.5673 |
| 15 | Advanced DL | Attention LSTM | 0.2125 | 0.1508 | 28.2776 | 15.7445 | 0.5400 |
| 16 | ML | RandomForestRegressor(n_estimators=200, max_depth=10) | 0.2171 | 0.1522 | 23.0157 | 14.6525 | 0.5195 |
| 17 | ML | XGBoostRegressor(n_estimators=200, learning_rate=0.05) | 0.2207 | 0.1558 | 22.8502 | 15.0533 | 0.5035 |
| 18 | ML | SVR(kernel='rbf', C=10, gamma='scale') | 0.2881 | 0.1847 | 24.1122 | 18.8419 | 0.1544 |
| 19 | Advanced DL | Temporal Transformer | 0.3116 | 0.2491 | 32.6103 | 24.7288 | 0.0106 |
| 20 | Deep Learning RNN | Bidirectional LSTM (64 units, dropout=0.2) | 0.3532 | 0.3003 | 33.5864 | 30.8463 | -0.2711 |
| 21 | Deep Learning RNN | GRU (64 units, dropout=0.2) | 0.3717 | 0.2986 | 48.5256 | 28.5107 | -0.4076 |
| 22 | Deep Learning RNN | LSTM (64 units, dropout=0.2) | 0.4256 | 0.3612 | 37.1785 | 37.7578 | -0.8457 |

## 5. Key Findings & Ablation Insights

1. Ensembles only barely improve the best base learner. The winning ensemble beats `Ridge(alpha=1.0)` by 0.000464 RMSE, which is a relative margin of only 0.295%. The ensemble is best, but the family-level conclusion is more important than the exact first-place row.
2. The most informative signals are recent persistence and short-horizon trend. A Ridge(alpha=1.0) refit on the saved standardized train split shows that `lag_1`, `rolling_mean_7`, `rolling_mean_30`, `lag_2`, and `lag_5` carry the largest coefficients. Calendar variables do not appear in the top features, which suggests that the study is mostly extracting autoregressive structure rather than seasonal calendar effects.

| Feature | Absolute standardized coefficient |
|---|---:|
| lag_1 | 0.3658 |
| rolling_mean_7 | 0.2718 |
| rolling_mean_30 | 0.1857 |
| lag_2 | 0.1323 |
| lag_5 | 0.0655 |
| lag_3 | 0.0510 |
| lag_21 | 0.0498 |
| lag_7 | 0.0444 |

3. Tree-based nonlinear models underperform because they struggle with smooth extrapolation on a trending holdout window. Their forecasts appear competitive in-sample, but out-of-sample they flatten relative to the linear and statistical families.
4. Multivariate advanced DL is better than univariate RNNs, which means the engineered feature set is useful even for neural models. However, the sample size is still too limited to justify the additional capacity relative to the strong low-parameter baselines.
5. The broad picture is robust even when the wider archived sweep is considered: simpler families keep winning, while depth, attention, and stacking only help when tightly constrained.

## 6. Statistical Analysis

The top three approved models are all very close to one another, so the correct interpretation is practical parity rather than a dramatic win. The table below compares their RMSE values against the naive persistence baseline and the best statistical baseline.

| Model | RMSE | Improvement vs Naive (%) | Improvement vs ARIMA (%) |
|---|---:|---:|---:|
| Weighted Ensemble (Top-5 Overall) | 0.156431 | 17.468 | 1.229 |
| Ridge(alpha=1.0) | 0.156894 | 17.223 | 0.936 |
| LinearRegression | 0.156938 | 17.200 | 0.909 |
| Top-3 mean RMSE | 0.156754 | 17.297 | 1.024 |

The winner improves RMSE over naive persistence by 17.47% and over the best statistical baseline by 1.23%. These are real but modest gains. The difference between first and second place is only 0.000464 RMSE, and the gap between second and third is only 0.000044.
Given that the study uses one chronological holdout instead of multiple rolling windows, and given that no Diebold-Mariano or bootstrap significance test was archived with the results, it would be overstated to claim that the winner is statistically superior to the runner-up in a formal inferential sense. What is clearly meaningful is the family-level pattern: the top cluster of ensemble, linear, and ARIMA-style models is materially better than naive persistence and far better than the approved RNN family.

## 7. Limitations & Future Work

- The final ranking is based on a single fixed holdout from 2022-04-26 to 2024-12-31 rather than repeated walk-forward backtests.
- No formal forecast-difference significance test was saved, so the top-3 ranking should be treated as practically close.
- The approved deep-learning steps were intentionally compact. Broader tuning, longer training schedules, or richer exogenous inputs could change the DL ordering.
- The target is forecast from lagged index behavior and calendar fields only; macro, sentiment, and order-book features were not included.
- Feature importance was derived from a step-9 Ridge refit on saved training arrays because trained step-5 model objects were not persisted as artifacts.
Future work should prioritize rolling-origin evaluation, ARIMAX or SARIMAX with exogenous regressors, regime-aware ensembles, probabilistic intervals, and stronger DL tuning only after the linear/statistical ceiling is genuinely challenged.

## 8. Appendix

The appendix inventories the broader 42-model archived sweep retained in the repository at `results/MASTER_LEADERBOARD.csv`. This appendix is included for reproducibility because the workspace still contains those experimental artifacts. The official conclusions above do not rank those archived rows against the approved 22-row step-8 leaderboard.
For the archived neural variants, the common training shorthand is: lookback=30, Adam(lr=0.001), MSE loss, batch_size=32, temporal validation split, and early stopping unless explicitly noted otherwise.

### Statistical Baselines

| Rank | Model | RMSE | Configuration summary |
|---|---|---:|---|
| 7 | arima | 0.1584 | ARIMA(order=(1,1,1), trend='n'); selected by auto_arima with train AIC=-1356.17. |
| 8 | sarima | 0.1584 | SARIMA(order=(1,1,1), seasonal_order=(0,0,0,21), trend='n'); seasonal period=21. |
| 11 | ets | 0.1592 | SimpleExpSmoothing; alpha=0.3724; initial_level=-1.4243; recursive walk-forward updates. |
| 12 | naive | 0.1895 | Persistence baseline; predicts the previous observed actual value. |
| 13 | moving_average | 0.1607 | Walk-forward moving average with window=7. |

### Classical ML

| Rank | Model | RMSE | Configuration summary |
|---|---|---:|---|
| 1 | SVR (Linear kernel) | 0.1576 | SVR with linear kernel; archive name omits C/epsilon, later approved run used C=1.0. |
| 4 | ElasticNet (a=0.01, l1=0.5) | 0.1565 | ElasticNet(alpha=0.01, l1_ratio=0.5) on standardized engineered features. |
| 5 | Ridge Regression (a=10.0) | 0.1567 | Ridge(alpha=10.0). |
| 6 | Ridge Regression (a=1.0) | 0.1569 | Ridge(alpha=1.0). |
| 9 | Linear Regression | 0.1569 | Ordinary least squares regression on the 17 standardized features. |
| 10 | Lasso Regression (a=0.01) | 0.1578 | Lasso(alpha=0.01); later approved run used max_iter=10000. |
| 20 | Random Forest (n=200) | 0.2095 | Random forest with n_estimators=200; full tree-depth kwargs were not preserved in archive metadata. |
| 21 | Gradient Boosting (n=200) | 0.2041 | Gradient boosting regressor with n_estimators=200; remaining booster kwargs not persisted. |
| 22 | LightGBM (n=300) | 0.2107 | LightGBM regressor with n_estimators=300; additional leaf and depth settings not retained. |
| 23 | Extra Trees (n=200) | 0.2141 | ExtraTrees regressor with n_estimators=200; remaining forest kwargs not retained. |
| 28 | XGBoost (n=300) | 0.2314 | XGBoost regressor with n_estimators=300; archive metadata does not retain learning-rate or depth. |
| 31 | SVR (RBF kernel) | 0.2769 | SVR with RBF kernel; later approved run used C=10 and gamma='scale'. |
| 36 | KNN (k=5) | 0.2675 | KNeighborsRegressor with n_neighbors=5. |

### RNN (LSTM/GRU)

| Rank | Model | RMSE | Configuration summary |
|---|---|---:|---|
| 14 | GRU (vanilla, 64u) | 0.1654 | Single-layer GRU with 64 units. |
| 16 | LSTM (vanilla, 64u) | 0.1682 | Single-layer LSTM with 64 units. |
| 25 | LSTM (dropout=0.2) | 0.2213 | Single-layer LSTM with dropout=0.2. |
| 27 | Stacked GRU (2-layer) | 0.2170 | Two-layer stacked GRU. |
| 30 | Bidirectional GRU | 0.2271 | Bidirectional GRU encoder. |
| 39 | Bidirectional LSTM | 0.3196 | Bidirectional LSTM encoder. |
| 40 | Stacked LSTM (2-layer) | 0.4156 | Two-layer stacked LSTM. |
| 41 | Deep LSTM (3-layer + BN) | 0.5512 | Three-layer LSTM stack with batch normalization. |
| 42 | LSTM (dropout=0.4) | 0.6926 | Single-layer LSTM with dropout=0.4. |

### Advanced DL

| Rank | Model | RMSE | Configuration summary |
|---|---|---:|---|
| 15 | Transformer (4-head) | 0.1691 | Temporal transformer with 4 attention heads and sinusoidal positional encoding; archived note indicates 150 epochs. |
| 18 | WaveNet Dilated CNN | 0.1904 | WaveNet-style dilated causal CNN with gated skip connections; exact dilation schedule not retained. |
| 24 | CNN-GRU | 0.2088 | Conv1D encoder followed by a GRU decoder. |
| 32 | Attention BiLSTM | 0.2369 | Bidirectional LSTM with attention pooling. |
| 34 | Attention LSTM | 0.2560 | LSTM with attention pooling. |
| 35 | TCN (Temporal Conv Net) | 0.2582 | Temporal convolutional network; exact residual-block depth was not persisted. |
| 37 | CNN-LSTM | 0.2671 | Conv1D encoder followed by an LSTM decoder. |
| 38 | CNN + Transformer | 0.3139 | Hybrid CNN encoder plus transformer block. |

### Ensemble/Hybrid

| Rank | Model | RMSE | Configuration summary |
|---|---|---:|---|
| 2 | Top-5 Average | 0.1563 | Uniform average over the top 5 archived models by ranking. |
| 3 | Top-3 Average | 0.1566 | Uniform average over the top 3 archived models by ranking. |
| 17 | Weighted Average (inv-MAE) | 0.1708 | Weighted average with inverse-MAE weights. |
| 19 | Simple Average (all models) | 0.1974 | Uniform average across the archived model pool. |
| 26 | Stacking (Ridge meta) | 0.2268 | Stacking ensemble with a Ridge meta-learner. |
| 29 | Stacking (Linear meta) | 0.2369 | Stacking ensemble with a LinearRegression meta-learner. |
| 33 | Stacking (XGBoost meta) | 0.2644 | Stacking ensemble with an XGBoost meta-learner. |
