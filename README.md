# Netflix Stock Return Prediction
This project analyzes Netflix (NFLX) stock data and predicts
5-day future returns using machine learning models.

# Dataset
- Historical daily NFLX stock data from yfinance.com
- Period 2015-02-16 to 2025-12-31

# Project Structure
- data/ : Raw stock price data
- src/ : Data wrangling, feature engineering and features/importance
- notebooks/ : EDA and model training

# Models Used
- Linear Regression
- Random Forest Regressor and Classifier

# Metrics
- MAE(Mean Absolute Error)
- RMSE(Mean Square Error)
- Accuracy Score

# Goal
Predict short-term(5 days) returns and evaluate whether models
outperform naive benchmarks.

# Key Findings
- Price-level prediction performed poorly due to non-stationarity.
- Return-based modeling significantly reduced prediction error.
- Models struggled to outperform a naive baseline.
- Directional accuracy hovered around 46–49%, close to random guessing.

## Conclusion
This highlight the difficulty of predicting stock prices using
historical data alone. While feature engineering improved stability,
models did not consistently outperform naive benchmarks.

## Future Suggested Improvement
- Incoporate macroeconomic indicator
- Include technical indicators like RSI, MACD ...
- Predict volatility instead of returns
