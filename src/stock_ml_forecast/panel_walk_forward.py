from dataclasses import dataclass

import pandas as pd

from stock_ml_forecast.panel_features import (
    MODEL_FEATURES_252D,
)

from stock_ml_forecast.panel_models import (
    train_panel_models,
)

from stock_ml_forecast.panel_backtest import (
    run_monthly_ranking_backtest,
)


@dataclass
class WalkForwardResult:
    folds: pd.DataFrame
    monthly: pd.DataFrame


def run_walk_forward_validation(
    data: pd.DataFrame,
    final_test_start: str,
    purge_horizon: int = 252,
    validation_days: int = 126,
    step_days: int = 126,
    min_train_days: int = 756,
) -> WalkForwardResult:
    """
    Expanding-window walk-forward validation.

    final_test_start:
        Beginning of the locked final test period.

    purge_horizon:
        Number of trading dates removed between training
        and validation.

    validation_days:
        Approx. 6 months = 126 trading days.

    min_train_days:
        Require roughly 3 years before first fold.
    """

    dates = pd.DatetimeIndex(
        data.index
        .get_level_values("Date")
        .unique()
    ).sort_values()

    final_test_start = pd.Timestamp(
        final_test_start
    )

    development_dates = dates[
        dates < final_test_start
    ]

    fold_rows = []
    monthly_results = []

    validation_start_position = (
        min_train_days
        + purge_horizon
    )

    fold_number = 1

    while (
        validation_start_position
        + validation_days
        <= len(development_dates)
    ):

        validation_start = (
            validation_start_position
        )

        validation_end = (
            validation_start
            + validation_days
        )

        train_end = (
            validation_start
            - purge_horizon
        )

        train_dates = (
            development_dates[
                :train_end
            ]
        )

        validation_dates = (
            development_dates[
                validation_start:
                validation_end
            ]
        )

        if (
            len(train_dates) < min_train_days
            or len(validation_dates) == 0
        ):
            validation_start_position += (
                step_days
            )

            continue

        row_dates = (
            data.index
            .get_level_values("Date")
        )

        train_mask = (
            row_dates.isin(
                train_dates
            )
        )

        validation_mask = (
            row_dates.isin(
                validation_dates
            )
        )

        X_train = (
            data.loc[
                train_mask,
                MODEL_FEATURES_252D,
            ]
            .copy()
        )

        X_validation = (
            data.loc[
                validation_mask,
                MODEL_FEATURES_252D,
            ]
            .copy()
        )

        y_top10_train = (
            data.loc[
                train_mask,
                "Top_10pct_252D",
            ]
            .astype(int)
        )

        y_return_train = (
            data.loc[
                train_mask,
                "Forward_Excess_Return_252D",
            ]
            .astype(float)
        )

        y_top10_validation = (
            data.loc[
                validation_mask,
                "Top_10pct_252D",
            ]
            .astype(int)
        )

        y_return_validation = (
            data.loc[
                validation_mask,
                "Forward_Excess_Return_252D",
            ]
            .astype(float)
        )

        print(
            f"\nWalk-forward fold {fold_number}"
        )

        print(
            "Train:",
            train_dates.min(),
            "->",
            train_dates.max(),
        )

        print(
            "Validation:",
            validation_dates.min(),
            "->",
            validation_dates.max(),
        )

        models = train_panel_models(
            X_train=X_train,
            y_top10_train=y_top10_train,
            y_return_train=y_return_train,
        )

        backtest = (
            run_monthly_ranking_backtest(
                classifier=models.classifier,
                X=X_validation,
                y_top10=y_top10_validation,
                y_return=y_return_validation,
                label=f"Fold {fold_number}",
            )
        )

        summary = (
            backtest.summary.copy()
        )

        summary[
            "Fold"
        ] = fold_number

        summary[
            "Train_Start"
        ] = train_dates.min()

        summary[
            "Train_End"
        ] = train_dates.max()

        summary[
            "Validation_Start"
        ] = validation_dates.min()

        summary[
            "Validation_End"
        ] = validation_dates.max()

        fold_rows.append(
            summary
        )

        monthly = (
            backtest.monthly.copy()
        )

        monthly[
            "Fold"
        ] = fold_number

        monthly_results.append(
            monthly
        )

        fold_number += 1

        validation_start_position += (
            step_days
        )

    folds = pd.concat(
        fold_rows,
        ignore_index=True,
    )

    monthly = pd.concat(
        monthly_results,
        ignore_index=True,
    )

    return WalkForwardResult(
        folds=folds,
        monthly=monthly,
    )


def print_walk_forward_summary(
    result: WalkForwardResult,
) -> None:

    print(
        "\nWalk-forward summary:"
    )

    ml = (
        result.folds[
            result.folds["Method"]
            == "ML Top-10 Probability"
        ]
        .copy()
    )

    momentum = (
        result.folds[
            result.folds["Method"]
            == "63D Momentum"
        ]
        .copy()
    )

    print(
        "\nML Top-10 Probability"
    )

    print(
        "Average precision @ 10%:     "
        f"{ml['Precision_at_10pct'].mean():.2%}"
    )

    print(
        "Average precision lift:      "
        f"{ml['Precision_Lift_vs_Random'].mean():.2f}x"
    )

    print(
        "Average return lift:         "
        f"{ml['Return_Lift'].mean():.2%}"
    )

    print(
        "\n63D Momentum"
    )

    print(
        "Average precision @ 10%:     "
        f"{momentum['Precision_at_10pct'].mean():.2%}"
    )

    print(
        "Average precision lift:      "
        f"{momentum['Precision_Lift_vs_Random'].mean():.2f}x"
    )

    print(
        "Average return lift:         "
        f"{momentum['Return_Lift'].mean():.2%}"
    )

    print(
        "\nML fold precision:"
    )

    for _, row in ml.iterrows():

        print(
            f"Fold {int(row['Fold'])}: "
            f"{row['Precision_at_10pct']:.2%}"
        )

    print("=" * 60)

def print_regime_summary(
    panel: pd.DataFrame,
    start_date: str,
    end_date: str,
    label: str,
) -> None:

    dates = (
        panel.index
        .get_level_values("Date")
    )

    mask = (
        (dates >= pd.Timestamp(start_date))
        &
        (dates <= pd.Timestamp(end_date))
    )

    period = (
        panel.loc[mask]
        .reset_index()
        .groupby("Date")
        .first()
    )

    columns = [
        "VIX",
        "VIX_Change_20D",
        "Treasury_3M",
        "Treasury_3M_Change_63D",
        "Treasury_10Y",
        "Treasury_10Y_Change_63D",
        "Yield_Curve_10Y_3M",
        "Yield_Curve_Change_63D",
        "SPY_Return_63D",
        "SPY_Volatility_20D",
    ]

    print(
        f"\n{label} regime summary:"
    )

    print(
        period[
            columns
        ]
        .describe()
        .loc[
            [
                "mean",
                "std",
                "min",
                "max",
            ]
        ]
        .T
    )

    print("=" * 60)