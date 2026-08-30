import numpy as np
import pandas as pd


def add_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add technical, volatility, and money-flow features
    to a market-data DataFrame.

    Expected input columns
    ----------------------
    Stock_High
    Stock_Low
    Stock_Close
    Stock_Adj_Close
    Stock_Volume
    Stock_Return_1D

    Returns
    -------
    pd.DataFrame
        Copy of the original DataFrame with additional features.
    """

    df = df.copy()

    # ==================================================
    # 1. MONEY FLOW
    # ==================================================

    # Average trading price during the day
    df["Typical_Price"] = (
        df["Stock_High"]
        + df["Stock_Low"]
        + df["Stock_Close"]
    ) / 3

    # Approximate dollar value traded
    df["Dollar_Volume"] = (
        df["Typical_Price"]
        * df["Stock_Volume"]
    )

    # Avoid division by zero when high == low
    price_range = (
        df["Stock_High"] - df["Stock_Low"]
    ).replace(0, np.nan)

    # Measures where the close occurred inside
    # the day's high-low range.
    #
    # Close near high  -> approximately +1
    # Close near low   -> approximately -1
    df["Money_Flow_Multiplier"] = (
        (
            (df["Stock_Close"] - df["Stock_Low"])
            -
            (df["Stock_High"] - df["Stock_Close"])
        )
        / price_range
    )

    # Volume weighted by directional price pressure
    df["Money_Flow_Volume"] = (
        df["Money_Flow_Multiplier"]
        * df["Stock_Volume"]
    )

    # Dollar flow weighted by buying/selling pressure
    df["Directional_Dollar_Flow"] = (
        df["Money_Flow_Multiplier"]
        * df["Dollar_Volume"]
    )

    # Rolling directional dollar flows
    for window in (5, 20, 60):
        df[f"Flow_{window}D"] = (
            df["Directional_Dollar_Flow"]
            .rolling(window=window)
            .sum()
        )

    # ==================================================
    # 2. MOVING AVERAGES
    # ==================================================

    for window in (20, 50, 200):
        df[f"MA_{window}"] = (
            df["Stock_Adj_Close"]
            .rolling(window=window)
            .mean()
        )

    # Relative distance between price and moving average
    df["Price_to_MA20"] = (
        df["Stock_Adj_Close"] / df["MA_20"] - 1
    )

    df["Price_to_MA50"] = (
        df["Stock_Adj_Close"] / df["MA_50"] - 1
    )

    df["Price_to_MA200"] = (
        df["Stock_Adj_Close"] / df["MA_200"] - 1
    )

    # Optional trend relationship
    df["MA20_to_MA200"] = (
        df["MA_20"] / df["MA_200"] - 1
    )

    # ==================================================
    # 3. RSI
    # ==================================================

    delta = df["Stock_Adj_Close"].diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)

    df["RSI_14"] = (
        100 - (100 / (1 + rs))
    )

    # ==================================================
    # 4. MACD
    # ==================================================

    ema_12 = (
        df["Stock_Adj_Close"]
        .ewm(span=12, adjust=False)
        .mean()
    )

    ema_26 = (
        df["Stock_Adj_Close"]
        .ewm(span=26, adjust=False)
        .mean()
    )

    df["MACD"] = ema_12 - ema_26

    df["MACD_Signal"] = (
        df["MACD"]
        .ewm(span=9, adjust=False)
        .mean()
    )

    # Useful additional MACD feature
    df["MACD_Histogram"] = (
        df["MACD"]
        - df["MACD_Signal"]
    )

    # ==================================================
    # 5. VOLATILITY
    # ==================================================

    df["Volatility_20D"] = (
        df["Stock_Return_1D"]
        .rolling(window=20)
        .std()
    )

    df["Volatility_60D"] = (
        df["Stock_Return_1D"]
        .rolling(window=60)
        .std()
    )

    return df