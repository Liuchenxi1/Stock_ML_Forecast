from stock_ml_forecast.market_data import download_market_data
from stock_ml_forecast.features import add_technical_features


df = download_market_data(
    ticker= "HCA",
    # input("Enter the stock ticker to analyze (example: AAPL, MSFT, HCA): ").strip().upper(),
    benchmark_ticker="SPY"
)

df = add_technical_features(df)
print(df.tail())
print(df.shape)