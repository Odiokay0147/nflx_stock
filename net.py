# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score
from src.wrangle_nflx import wrangle_nflx
from src.imp import plot_feature_importance

# %%
nflx = wrangle_nflx("data/NFLX_stock_data.csv")
nflx.info()
nflx.tail()

# %%
#Trend of closing price
plt.figure(figsize=(10, 5))
plt.plot(nflx.index, nflx["Close"])
plt.xlabel("Date")
plt.ylabel("Price")
plt.title("Netflix Closing Price From 2015 - 2026");

# %%
#Trade of Volume
plt.figure(figsize=(10, 5))
plt.plot(nflx.index, nflx["Volume"])
plt.xlabel("Date")
plt.ylabel("Volume")
plt.title("Netflix Trading Volume From 2015 - 2026");

# %%
#Daily Returns
plt.figure(figsize=(10, 5))
plt.plot(nflx.index, nflx["Returns"])
plt.xlabel("Date")
plt.ylabel("Returns")
plt.title("Netflix Daily Returns From 2015 - 2026");

# %%
#30 days rolling volatility
plt.figure(figsize=(10, 5))
plt.plot(nflx.index, nflx["Vol_30"])
plt.xlabel("Date")
plt.ylabel("Volatility_30")
plt.title("Netflix 30 Trading Days Rolling Volatility");

# %%
#21 days rolling volatility
plt.figure(figsize=(10, 5))
plt.plot(nflx.index, nflx["Vol_21"])
plt.xlabel("Date")
plt.ylabel("Volatility_21")
plt.title("Netflix 21 Trading Days Rolling Volatility");

# %%
#Trend showing moving averages 10 and 50
plt.figure(figsize=(10, 5))
plt.plot(nflx["Close"], label="Close")
plt.plot(nflx["MA_10"], label="MA_10")
plt.plot(nflx["MA_50"], label="MA_50")
plt.title("Netflix Moving Averages vs Price")
plt.legend();

# %%
#defining target and features variable
features = ["Vol_21", "Vol_30", "MA_10", "MA_50", "Lag_1", "Lag_5"]
X = nflx[features]
y5 = nflx["Returns_5d"]
y5c = nflx["Direction_5d"]

# %%
int(len(nflx) * 0.8)

# %%
#train and test split
def split(X, y):
    return train_test_split(X, y, test_size=0.2, shuffle=False)
X_train, X_test, y5_train, y5_test = split(X, y5)
X_train_c, X_test_c, y5c_train, y5c_test = split(X, y5c)

# %%
#Build and fit lr model and rf_reg
lr = LinearRegression()
rf_reg = RandomForestRegressor(
    n_estimators=200,
    max_depth=5,
    random_state=42
    )
lr.fit(X_train, y5_train)

# %%
y5_pred = lr.predict(X_test)

#mean absolute error and mean squared error
print("5-day MAE:", mean_absolute_error(y5_test, y5_pred))
print("5-day RMSE:", np.sqrt(mean_squared_error(y5_test, y5_pred)))

# %%
rf_reg.fit(X_train, y5_train)

# %%
y5_rf = rf_reg.predict(X_test)

#mean absolute error and mean squared error
print("5-day MAE:", mean_absolute_error(y5_test, y5_rf))
print("5-day RMSE:", np.sqrt(mean_squared_error(y5_test, y5_rf)))

# %%
naive_predict = np.zeros(len(y5_test))
naive_mae = mean_absolute_error(y5_test, naive_predict)

print("Naive MAE:", naive_mae)

# %%
#Building and fiting classifier models
lor = LogisticRegression(max_iter=1000)
rfc = RandomForestClassifier(
    n_estimators=200,
    max_depth=5,
    random_state=42
)
lor.fit(X_train_c, y5c_train)

# %%
y5c_pred = lor.predict(X_test_c)
print("5-day Direction Accuracy:", accuracy_score(y5c_test, y5c_pred))

# %%
rfc.fit(X_train_c, y5c_train)

# %%
y5c_rfc = rfc.predict(X_test_c)
print("5-day Direction Accuracy:", accuracy_score(y5c_test, y5c_rfc))

# %%
#Ploting features/Importances for the 5 days returns(Random forest Regression)
plot_feature_importance(
    rf_reg,
    features,
    "Feature Importance 5-Day Returns (Randon Forest Regression)"
)

# %%
#Ploting features/Importances for the 5 days returns(Random forest Classifier)
plot_feature_importance(
    rfc,
    features,
    "Feature Importance 5-Day Returns (Randon Forest Classifier)"
)

# %%