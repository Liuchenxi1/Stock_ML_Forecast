from stock_ml_forecast.market_data import download_market_data
from stock_ml_forecast.features import add_technical_features
from stock_ml_forecast.sec_data import add_sec_fundamentals
from stock_ml_forecast.dataset import add_forward_return_targets


TICKER = "HCA"

SEC_USER_AGENT = (
    "Stock ML Forecast shinnkiryu@gmail.com"
)


df = download_market_data(
    ticker=TICKER,
    benchmark_ticker="SPY",
)

df = add_technical_features(df)

df = add_sec_fundamentals(
    df=df,
    ticker=TICKER,
    user_agent=SEC_USER_AGENT,
)

df = add_forward_return_targets(df)

print(
    df[
        [
            "Stock_Adj_Close",
            "Forward_Return_63D",
            "Forward_Return_126D",
            "Forward_Return_252D",
        ]
    ].tail(270)
)

print(df.columns.to_list())