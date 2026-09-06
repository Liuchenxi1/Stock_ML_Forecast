from stock_ml_forecast.market_data import download_market_data

from stock_ml_forecast.features import add_technical_features

from stock_ml_forecast.sec_data import add_sec_fundamentals

from stock_ml_forecast.dataset import add_forward_return_targets,prepare_all_horizons,split_all_horizons

from stock_ml_forecast.preprocessing import preprocess_split

from stock_ml_forecast.models import train_linear_model,train_ridge_model,evaluate_regression,evaluate_naive_zero,evaluate_naive_mean

TICKER = "HCA"

SEC_USER_AGENT = (
    "Stock ML Forecast your_email@example.com"
)

# 1. Market data//

df = download_market_data(
    ticker=TICKER,
    benchmark_ticker="SPY",
)

# 2. Technical features//

df = add_technical_features(df)

# 3. SEC fundamentals//

df = add_sec_fundamentals(
    df=df,
    ticker=TICKER,
    user_agent=SEC_USER_AGENT,
)

# 4. Forecast targets//

df = add_forward_return_targets(df)

# 5. ML datasets//

datasets = prepare_all_horizons(df)

# 6. Purged chronological split

splits = split_all_horizons(
    datasets=datasets,
    train_ratio=0.70,
    val_ratio=0.15,
)

# 7. Preprocessing

prepared = {
    horizon: preprocess_split(split)
    for horizon, split in splits.items()
}

# 8. Display dataset sizes

for horizon in (63, 126, 252):

    split = splits[horizon]

    print("\n" + "=" * 60)
    print(f"{horizon}D dataset sizes")

    print(
        "Train:",
        split.X_train.shape,
    )

    print(
        "Validation:",
        split.X_val.shape,
    )

    print(
        "Test:",
        split.X_test.shape,
    )

# 9. Train baseline models

for horizon, data in prepared.items():

    print("\n" + "=" * 60)
    print(f"{horizon}-day models")


    # --------------------------------------------------------
    # Naive baselines
    # --------------------------------------------------------

    print("\nNaive baselines")

    print(
        "Zero-return baseline:",
        evaluate_naive_zero(
            data.y_test,
        ),
    )

    print(
        "Training-mean baseline:",
        evaluate_naive_mean(
            data.y_train,
            data.y_test,
        ),
    )


    # --------------------------------------------------------
    # Linear Regression
    # --------------------------------------------------------

    linear = train_linear_model(
        data
    )

    print("\nLinear Regression")

    print(
        "Validation:",
        evaluate_regression(
            linear,
            data.X_val,
            data.y_val,
        ),
    )

    print(
        "Test:",
        evaluate_regression(
            linear,
            data.X_test,
            data.y_test,
        ),
    )


    # --------------------------------------------------------
    # Ridge Regression
    # --------------------------------------------------------

    ridge = train_ridge_model(
        data,
        alpha=1.0,
    )

    print("\nRidge")

    print(
        "Validation:",
        evaluate_regression(
            ridge,
            data.X_val,
            data.y_val,
        ),
    )

    print(
        "Test:",
        evaluate_regression(
            ridge,
            data.X_test,
            data.y_test,
        ),
    )