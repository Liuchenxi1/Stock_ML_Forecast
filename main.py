from stock_ml_forecast.market_data import download_market_data


df = download_market_data(
    ticker= "HCA",
    # input("Enter the stock ticker to analyze (example: AAPL, MSFT, HCA): ").strip().upper(),
    benchmark_ticker="SPY"
)

print(df.tail())
print(df.shape)
print(df.columns.to_list())