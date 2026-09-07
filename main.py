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
    split_all_horizons,
)

from stock_ml_forecast.preprocessing import (
    preprocess_split,
)

from stock_ml_forecast.models import (
    # New architecture
    train_direction_magnitude_models,
    predict_expected_return,
    evaluate_direction_magnitude,
    evaluate_conditional_magnitude,

    # Baselines / old models
    evaluate_naive_zero,
    evaluate_naive_mean,
    train_linear_model,
    train_ridge_model,
    evaluate_regression,
)


# ============================================================
# Configuration
# ============================================================

TICKER = "AAPL"

SEC_USER_AGENT = (
    "Stock ML Forecast shinnkiryu@gmail.com"
)


# ============================================================
# 1. Download market data
# ============================================================

df = download_market_data(
    ticker=TICKER,
    benchmark_ticker="SPY",
)


# ============================================================
# 2. Add technical features
# ============================================================

df = add_technical_features(
    df
)


# ============================================================
# 3. Add SEC fundamentals
# ============================================================

df = add_sec_fundamentals(
    df=df,
    ticker=TICKER,
    user_agent=SEC_USER_AGENT,
)


# ============================================================
# 4. Add forward-return targets
# ============================================================

df = add_forward_return_targets(
    df
)


# ============================================================
# 5. Build horizon datasets
# ============================================================

datasets = prepare_all_horizons(
    df
)


# ============================================================
# 6. Purged chronological splits
# ============================================================

splits = split_all_horizons(
    datasets=datasets,
    train_ratio=0.70,
    val_ratio=0.15,
)


# ============================================================
# 7. Preprocess each horizon
# ============================================================

prepared = {
    horizon: preprocess_split(split)
    for horizon, split in splits.items()
}


# ============================================================
# 8. Train Direction + Upside + Downside models
# ============================================================

direction_magnitude_models = {}


for horizon, data in prepared.items():

    print(
        f"\n{horizon}-day "
        "Direction + Magnitude model"
    )

    model = (
        train_direction_magnitude_models(
            data
        )
    )

    direction_magnitude_models[
        horizon
    ] = model

    validation_metrics = (
        evaluate_direction_magnitude(
            models=model,
            X=data.X_val,
            y_return=data.y_return_val,
            y_direction=data.y_direction_val,
        )
    )

    test_metrics = (
        evaluate_direction_magnitude(
            models=model,
            X=data.X_test,
            y_return=data.y_return_test,
            y_direction=data.y_direction_test,
        )
    )

    print(
        "Validation:",
        validation_metrics,
    )

    print(
        "Test:",
        test_metrics,
    )

    # --------------------------------------------------------
    # Conditional upside / downside performance
    # --------------------------------------------------------

    conditional_validation = (
        evaluate_conditional_magnitude(
            models=model,
            X=data.X_val,
            y_upside=data.y_upside_val,
            y_downside=data.y_downside_val,
        )
    )

    conditional_test = (
        evaluate_conditional_magnitude(
            models=model,
            X=data.X_test,
            y_upside=data.y_upside_test,
            y_downside=data.y_downside_test,
        )
    )

    print(
        "Validation magnitude:",
        conditional_validation,
    )

    print(
        "Test magnitude:",
        conditional_test,
    )

    print("=" * 60)


# ============================================================
# 9. Dataset sizes
# ============================================================

for horizon in (
    63,
    126,
    252,
):

    split = splits[
        horizon
    ]

    print(
        f"\n{horizon}D dataset sizes"
    )

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

    print("=" * 60)


# ============================================================
# 10. Baseline / Linear / Ridge comparison
# ============================================================

for horizon, data in prepared.items():

    print(
        f"\n{horizon}-day baseline models"
    )

    # --------------------------------------------------------
    # Naive baselines
    # --------------------------------------------------------

    print(
        "Zero-return baseline:",
        evaluate_naive_zero(
            data.y_return_test,
        ),
    )

    print(
        "Training-mean baseline:",
        evaluate_naive_mean(
            data.y_return_train,
            data.y_return_test,
        ),
    )

    # --------------------------------------------------------
    # Linear Regression
    # --------------------------------------------------------

    linear = (
        train_linear_model(
            data
        )
    )

    linear_validation = (
        evaluate_regression(
            linear,
            data.X_val,
            data.y_return_val,
        )
    )

    linear_test = (
        evaluate_regression(
            linear,
            data.X_test,
            data.y_return_test,
        )
    )

    print(
        "Linear Validation:",
        linear_validation,
    )

    print(
        "Linear Test:",
        linear_test,
    )

    # --------------------------------------------------------
    # Ridge Regression
    # --------------------------------------------------------

    ridge = (
        train_ridge_model(
            data,
            alpha=1.0,
        )
    )

    ridge_validation = (
        evaluate_regression(
            ridge,
            data.X_val,
            data.y_return_val,
        )
    )

    ridge_test = (
        evaluate_regression(
            ridge,
            data.X_test,
            data.y_return_test,
        )
    )

    print(
        "Ridge Validation:",
        ridge_validation,
    )

    print(
        "Ridge Test:",
        ridge_test,
    )

    print("=" * 60)


# ============================================================
# 11. Inspect the latest test forecasts
# ============================================================

for horizon in (
    63,
    126,
    252,
):

    model = (
        direction_magnitude_models[
            horizon
        ]
    )

    data = prepared[
        horizon
    ]

    forecasts = (
        predict_expected_return(
            models=model,
            X=data.X_test,
        )
    )

    print(
        f"\n{horizon}D latest forecasts"
    )

    print(
        forecasts.tail(10)
    )

    print("=" * 60)