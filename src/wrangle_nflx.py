import pandas as pd

def wrangle_nflx(filepath):
    #load data
    df = pd.read_csv(filepath)

    #drop first 2 rows in data
    df = df.iloc[2:].reset_index(drop=True)

    #renaming columns
    df.columns = ["Date", "Close", "High", "Low", "Open", "Volume"]

    #converting datatypes for date and other columns
    df["Date"] = pd.to_datetime(df["Date"])
    num_cols = ["Close", "High", "Low", "Open", "Volume"]
    df[num_cols] = df[num_cols].astype(float)
    df.set_index("Date", inplace=True)

    #calculating returns
    df["Returns"] = df["Close"].pct_change()
    df["Returns_5d"] = df["Close"].pct_change(5).shift(-5)
    df["Returns_10d"] = df["Close"].pct_change(10).shift(-10)
    #directional targets
    df["Direction_5d"] = (df["Returns_5d"] > 0).astype(int)
    df["Direction_10d"] = (df["Returns_10d"] > 0).astype(int)
    #volatility 1month and 30 trading days(1.5month)
    df["Vol_21"] = df["Returns"].rolling(21).std()
    df["Vol_30"] = df["Returns"].rolling(30).std()
    #moving averages 10 and 50
    df["MA_10"] = df["Close"].rolling(10).mean()
    df["MA_50"] = df["Close"].rolling(50).mean()
    #lag features
    df["Lag_1"] = df["Close"].shift(1)
    df["Lag_5"] = df["Close"].shift(5)

    #drop NaN values
    df.dropna(inplace=True)

    return df