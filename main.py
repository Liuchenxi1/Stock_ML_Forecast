from stock_ml_forecast.market_data import (
    download_market_data,
)

from stock_ml_forecast.features import (
    add_technical_features,
)

from stock_ml_forecast.sec_data import (
    add_sec_fundamentals,
)

from stock_ml_forecast.dataset import (
    add_forward_return_targets,
    prepare_all_horizons,
)


TICKER = "HCA"

SEC_USER_AGENT = (
    "Stock ML Forecast your_email@example.com"
)


# ==================================================
# Market data
# ==================================================

df = download_market_data(
    ticker=TICKER,
    benchmark_ticker="SPY",
)


# ==================================================
# Technical features
# ==================================================

df = add_technical_features(df)


# ==================================================
# SEC fundamentals
# ==================================================

df = add_sec_fundamentals(
    df=df,
    ticker=TICKER,
    user_agent=SEC_USER_AGENT,
)


# ==================================================
# Forecast targets
# ==================================================

df = add_forward_return_targets(df)


# ==================================================
# ML datasets
# ==================================================

datasets = prepare_all_horizons(df)


# Example: 63-day model dataset
dataset_63 = datasets[63]

print("\n63-day X:")
print(dataset_63.X.tail())

print("\n63-day y:")
print(dataset_63.y.tail())


# 126-day
dataset_126 = datasets[126]


# 252-day
dataset_252 = datasets[252]