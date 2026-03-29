# Full Ablation Leaderboard

Generated from `results/metrics_registry.csv` after step-8 ensemble evaluation.

Best overall model: `Weighted Ensemble (Top-5 Overall)` (Ensemble) with RMSE=0.1564.
Best ensemble: `Weighted Ensemble (Top-5 Overall)` with RMSE=0.1564.

| Rank | Family | Model | RMSE | MAE | MAPE (%) | SMAPE (%) | R^2 |
|---:|---|---|---:|---:|---:|---:|---:|
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

Comparison plot: `plots/leaderboard_comparison.png`
