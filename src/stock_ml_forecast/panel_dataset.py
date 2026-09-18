from dataclasses import dataclass

import pandas as pd

from stock_ml_forecast.panel_features import (
    MODEL_FEATURES_252D,
)


@dataclass
class PanelDatasetSplit:
    X_train: pd.DataFrame
    X_validation: pd.DataFrame
    X_test: pd.DataFrame

    y_return_train: pd.Series
    y_return_validation: pd.Series
    y_return_test: pd.Series

    y_top10_train: pd.Series
    y_top10_validation: pd.Series
    y_top10_test: pd.Series

    train_dates: pd.DatetimeIndex
    validation_dates: pd.DatetimeIndex
    test_dates: pd.DatetimeIndex

    purge_horizon: int


def prepare_panel_training_data(
    panel: pd.DataFrame,
) -> pd.DataFrame:
    """
    Keep only rows with known 252-day targets.

    NaN feature values are intentionally preserved.
    HistGradientBoosting can handle them directly.
    """

    required_targets = [
        "Forward_Excess_Return_252D",
        "Top_10pct_252D",
    ]

    missing_targets = [
        column
        for column in required_targets
        if column not in panel.columns
    ]

    if missing_targets:
        raise ValueError(
            "Missing target columns: "
            f"{missing_targets}"
        )

    missing_features = [
        column
        for column in MODEL_FEATURES_252D
        if column not in panel.columns
    ]

    if missing_features:
        raise ValueError(
            "Missing model features: "
            f"{missing_features}"
        )

    data = panel.copy()

    # --------------------------------------------------------
    # Keep only rows where the future outcome is known.
    # --------------------------------------------------------

    data = data[
        data["Forward_Excess_Return_252D"].notna()
        &
        data["Top_10pct_252D"].notna()
    ].copy()

    data[
        "Top_10pct_252D"
    ] = (
        data[
            "Top_10pct_252D"
        ]
        .astype(int)
    )

    return data.sort_index()


def purged_chronological_split(
    data: pd.DataFrame,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
    purge_horizon: int = 252,
) -> PanelDatasetSplit:
    """
    Global chronological split across all companies.

    Every company on the same date belongs to the same split.

    The final `purge_horizon` trading dates are removed from
    train and validation so forward 252-day labels cannot
    overlap the following split.
    """

    if (
        train_fraction <= 0
        or validation_fraction <= 0
    ):
        raise ValueError(
            "Split fractions must be positive."
        )

    if (
        train_fraction
        + validation_fraction
        >= 1
    ):
        raise ValueError(
            "Train + validation fractions must be < 1."
        )

    # ========================================================
    # Unique global trading dates
    # ========================================================

    all_dates = pd.DatetimeIndex(
        data.index
        .get_level_values("Date")
        .unique()
    ).sort_values()

    number_of_dates = len(
        all_dates
    )

    train_boundary = int(
        number_of_dates
        * train_fraction
    )

    validation_boundary = int(
        number_of_dates
        * (
            train_fraction
            + validation_fraction
        )
    )

    # --------------------------------------------------------
    # Initial chronological blocks
    # --------------------------------------------------------

    raw_train_dates = all_dates[
        :train_boundary
    ]

    raw_validation_dates = all_dates[
        train_boundary:
        validation_boundary
    ]

    test_dates = all_dates[
        validation_boundary:
    ]

    # ========================================================
    # Purge overlapping labels
    # ========================================================

    if len(raw_train_dates) <= purge_horizon:
        raise ValueError(
            "Training period is too short "
            "for requested purge horizon."
        )

    if len(raw_validation_dates) <= purge_horizon:
        raise ValueError(
            "Validation period is too short "
            "for requested purge horizon."
        )

    train_dates = raw_train_dates[
        :-purge_horizon
    ]

    validation_dates = raw_validation_dates[
        :-purge_horizon
    ]

    # ========================================================
    # Create masks
    # ========================================================

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

    test_mask = (
        row_dates.isin(
            test_dates
        )
    )

    # ========================================================
    # Features
    # ========================================================

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

    X_test = (
        data.loc[
            test_mask,
            MODEL_FEATURES_252D,
        ]
        .copy()
    )

    # ========================================================
    # Regression target
    # ========================================================

    y_return_train = (
        data.loc[
            train_mask,
            "Forward_Excess_Return_252D",
        ]
        .astype(float)
        .copy()
    )

    y_return_validation = (
        data.loc[
            validation_mask,
            "Forward_Excess_Return_252D",
        ]
        .astype(float)
        .copy()
    )

    y_return_test = (
        data.loc[
            test_mask,
            "Forward_Excess_Return_252D",
        ]
        .astype(float)
        .copy()
    )

    # ========================================================
    # Classification target
    # ========================================================

    y_top10_train = (
        data.loc[
            train_mask,
            "Top_10pct_252D",
        ]
        .astype(int)
        .copy()
    )

    y_top10_validation = (
        data.loc[
            validation_mask,
            "Top_10pct_252D",
        ]
        .astype(int)
        .copy()
    )

    y_top10_test = (
        data.loc[
            test_mask,
            "Top_10pct_252D",
        ]
        .astype(int)
        .copy()
    )

    return PanelDatasetSplit(
        X_train=X_train,
        X_validation=X_validation,
        X_test=X_test,

        y_return_train=y_return_train,
        y_return_validation=y_return_validation,
        y_return_test=y_return_test,

        y_top10_train=y_top10_train,
        y_top10_validation=y_top10_validation,
        y_top10_test=y_top10_test,

        train_dates=train_dates,
        validation_dates=validation_dates,
        test_dates=test_dates,

        purge_horizon=purge_horizon,
    )


def print_split_summary(
    split: PanelDatasetSplit,
) -> None:
    """
    Show sizes, dates, and target distributions.
    """

    print(
        "\nPurged chronological split:"
    )

    print(
        f"Train:      "
        f"{len(split.X_train):>10,} rows"
    )

    print(
        f"Validation: "
        f"{len(split.X_validation):>10,} rows"
    )

    print(
        f"Test:       "
        f"{len(split.X_test):>10,} rows"
    )

    print()

    print(
        "Train dates:"
    )

    print(
        split.train_dates.min(),
        "->",
        split.train_dates.max(),
    )

    print(
        "Validation dates:"
    )

    print(
        split.validation_dates.min(),
        "->",
        split.validation_dates.max(),
    )

    print(
        "Test dates:"
    )

    print(
        split.test_dates.min(),
        "->",
        split.test_dates.max(),
    )

    print(
        "\nTop-10% rates:"
    )

    print(
        "Train:      "
        f"{split.y_top10_train.mean():.1%}"
    )

    print(
        "Validation: "
        f"{split.y_top10_validation.mean():.1%}"
    )

    print(
        "Test:       "
        f"{split.y_top10_test.mean():.1%}"
    )

    print(
        "\nMean excess returns:"
    )

    print(
        "Train:      "
        f"{split.y_return_train.mean():.2%}"
    )

    print(
        "Validation: "
        f"{split.y_return_validation.mean():.2%}"
    )

    print(
        "Test:       "
        f"{split.y_return_test.mean():.2%}"
    )

    print(
        "\nPurge horizon:",
        split.purge_horizon,
        "trading days",
    )

    print("=" * 60)