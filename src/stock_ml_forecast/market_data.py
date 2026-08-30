# src/stock_ml_forecast/market_data.py

from datetime import date

import pandas as pd
import yfinance as yf


def _get_ticker_column(
    data: pd.DataFrame,
    field: str,
    ticker: str,
) -> pd.Series:
    """
    Extract one field for one ticker from a yfinance DataFrame.
    """
    if isinstance(data.columns, pd.MultiIndex):
        return data[field][ticker]

    return data[field]


def download_market_data(
    ticker: str,
    benchmark_ticker: str = "SPY",
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    """
    Download stock and market data from Yahoo Finance.

    Parameters
    ----------
    ticker:
        Stock ticker to model, for example "HCA".

    benchmark_ticker:
        Benchmark ticker. Defaults to "SPY".

    start_date:
        Start date in YYYY-MM-DD format.
        If omitted, defaults to 10 years before end_date.

    end_date:
        End date in YYYY-MM-DD format.
        If omitted, defaults to today.

    Returns
    -------
    pd.DataFrame
        Clean daily market DataFrame containing stock,
        benchmark, volatility, and Treasury data.
    """

    if end_date is None:
        end = pd.Timestamp.today().normalize()
    else:
        end = pd.Timestamp(end_date)

    if start_date is None:
        start = end - pd.DateOffset(years=10)
    else:
        start = pd.Timestamp(start_date)

    market_tickers = list(
        dict.fromkeys(
            [
                ticker,
                benchmark_ticker,
                "^VIX",
                "^IRX",
                "^TNX",
            ]
        )
    )

    market = yf.download(
        market_tickers,
        start=start.strftime("%Y-%m-%d"),

        # yfinance treats end as exclusive,
        # so request one extra day.
        end=(end + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),

        interval="1d",
        auto_adjust=False,
        progress=False,
        group_by="column",
    )

    if market.empty:
        raise ValueError(
            f"No market data was downloaded for ticker {ticker}."
        )

    if isinstance(market.columns, pd.MultiIndex):
        market.columns.names = ["Price", "Ticker"]

    market = market.sort_index()

    df = pd.DataFrame(index=market.index.copy())
    df.index.name = "Date"

    # -----------------------------
    # Selected stock
    # -----------------------------

    df["Stock_Open"] = _get_ticker_column(
        market, "Open", ticker
    )

    df["Stock_High"] = _get_ticker_column(
        market, "High", ticker
    )

    df["Stock_Low"] = _get_ticker_column(
        market, "Low", ticker
    )

    df["Stock_Close"] = _get_ticker_column(
        market, "Close", ticker
    )

    df["Stock_Adj_Close"] = _get_ticker_column(
        market, "Adj Close", ticker
    )

    df["Stock_Volume"] = _get_ticker_column(
        market, "Volume", ticker
    )

    # -----------------------------
    # Benchmark / macro
    # -----------------------------

    df["Benchmark_Close"] = _get_ticker_column(
        market,
        "Close",
        benchmark_ticker,
    )

    df["VIX"] = _get_ticker_column(
        market,
        "Close",
        "^VIX",
    )

    df["Treasury_3M"] = _get_ticker_column(
        market,
        "Close",
        "^IRX",
    )

    df["Treasury_10Y"] = _get_ticker_column(
        market,
        "Close",
        "^TNX",
    )

    # -----------------------------
    # Basic market changes
    # -----------------------------

    df["Stock_Return_1D"] = (
        df["Stock_Adj_Close"].pct_change()
    )

    df["Benchmark_Return_1D"] = (
        df["Benchmark_Close"].pct_change()
    )

    df["VIX_Change"] = (
        df["VIX"].pct_change()
    )

    df["Treasury_3M_Change"] = (
        df["Treasury_3M"].diff()
    )

    df["Treasury_10Y_Change"] = (
        df["Treasury_10Y"].diff()
    )

    df["Yield_Curve_10Y_3M"] = (
        df["Treasury_10Y"]
        - df["Treasury_3M"]
    )

    return df