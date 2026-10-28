# Model card

**Task:** Predict hourly aggregate arrival volume 24 hours ahead using only information known at the forecast origin.

**Estimator:** scikit-learn HistGradientBoostingRegressor(loss="poisson").

**Reference methods:** Daily and weekly seasonal-naive forecasts.

**Evaluation:** Chronological training/calibration/testing with 24-hour purging; MAE, RMSE, conformal empirical coverage.

**Cautions:** A shifted evaluation regime can make a simple baseline stronger than the ML model. Exchangeability assumptions for conformal prediction do not hold under general time-series drift. No clinical or operational deployment.
