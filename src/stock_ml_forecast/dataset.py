from dataclasses import dataclass

import numpy as np
import pandas as pd

from stock_ml_forecast.config import (
    MODEL_FEATURES,
    FORECAST_HORIZONS,
)


@dataclass
class HorizonDataset:
    """
    Container for one forecasting horizon.
    """

    horizon: int
    target_column: str

    data: pd.DataFrame

    X: pd.DataFrame
    y: pd.Series


def add_forward_return_targets(
    df: pd.DataFrame,
    price_column: str = "Stock_Adj_Close",
    horizons: tuple[int, ...] = FORECAST_HORIZONS,
) -> pd.DataFrame:
    """
    Create forward-return targets.

    Example:

        Forward_Return_63D =
            Price[t + 63] / Price[t] - 1
    """

    result = df.copy()

    for horizon in horizons:

        target_column = (
            f"Forward_Return_{horizon}D"
        )

        result[target_column] = (
            result[price_column].shift(-horizon)
            / result[price_column]
            - 1
        )

    return result


def get_available_features(
    df: pd.DataFrame,
    requested_features: list[str] | None = None,
) -> list[str]:
    """
    Return only features that actually exist
    in the DataFrame.

    This is useful because some companies may not
    provide every SEC concept.
    """

    if requested_features is None:
        requested_features = MODEL_FEATURES

    available = [
        feature
        for feature in requested_features
        if feature in df.columns
    ]

    missing = [
        feature
        for feature in requested_features
        if feature not in df.columns
    ]

    if missing:
        print("\nMissing features:")

        for feature in missing:
            print(f"  - {feature}")

    return available


def prepare_horizon_dataset(
    df: pd.DataFrame,
    horizon: int,
    features: list[str] | None = None,
    min_feature_coverage: float = 0.60,
) -> HorizonDataset:
    """
    Create a clean machine-learning dataset for one
    forward-return horizon.

    Steps:
    1. Select the target.
    2. Keep only rows where target is known.
    3. Remove features with insufficient coverage.
    4. Keep X and y aligned.

    Missing feature values are NOT imputed here.
    Imputation should happen later using training data only.
    """

    if features is None:
        features = MODEL_FEATURES

    target_column = (
        f"Forward_Return_{horizon}D"
    )

    if target_column not in df.columns:
        raise ValueError(
            f"{target_column} does not exist. "
            "Run add_forward_return_targets() first."
        )

    # -------------------------------------------------
    # Only use features that exist
    # -------------------------------------------------

    available_features = get_available_features(
        df=df,
        requested_features=features,
    )

    if not available_features:
        raise ValueError(
            "No model features are available."
        )

    # -------------------------------------------------
    # Start with features + target
    # -------------------------------------------------

    columns = (
        available_features
        + [target_column]
    )

    dataset = df[columns].copy()

    # Replace infinity produced by ratios/calculations
    dataset = dataset.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    # -------------------------------------------------
    # Remove rows without known future target
    # -------------------------------------------------

    dataset = dataset.dropna(
        subset=[target_column]
    )

    if dataset.empty:
        raise ValueError(
            f"No usable rows for horizon {horizon}."
        )

    # -------------------------------------------------
    # Feature coverage
    # -------------------------------------------------

    coverage = (
        dataset[available_features]
        .notna()
        .mean()
    )

    selected_features = (
        coverage[
            coverage >= min_feature_coverage
        ]
        .index
        .tolist()
    )

    removed_features = (
        coverage[
            coverage < min_feature_coverage
        ]
        .sort_values()
    )

    if not removed_features.empty:

        print(
            f"\n{horizon}D features removed "
            f"for low coverage:"
        )

        for feature, ratio in removed_features.items():

            print(
                f"  {feature}: "
                f"{ratio:.1%}"
            )

    if not selected_features:
        raise ValueError(
            f"No features meet "
            f"{min_feature_coverage:.0%} coverage."
        )

    # -------------------------------------------------
    # Final dataset
    # -------------------------------------------------

    dataset = dataset[
        selected_features
        + [target_column]
    ].copy()

    X = dataset[selected_features].copy()

    y = dataset[target_column].copy()

    return HorizonDataset(
        horizon=horizon,
        target_column=target_column,
        data=dataset,
        X=X,
        y=y,
    )


def prepare_all_horizons(
    df: pd.DataFrame,
    horizons: tuple[int, ...] = FORECAST_HORIZONS,
    features: list[str] | None = None,
    min_feature_coverage: float = 0.60,
) -> dict[int, HorizonDataset]:
    """
    Build one dataset for every forecasting horizon.
    """

    datasets = {}

    for horizon in horizons:

        print(
            "\n"
            + "=" * 60
        )

        print(
            f"Preparing {horizon}-day dataset"
        )

        dataset = prepare_horizon_dataset(
            df=df,
            horizon=horizon,
            features=features,
            min_feature_coverage=min_feature_coverage,
        )

        datasets[horizon] = dataset

        print(
            f"Rows: {len(dataset.data):,}"
        )

        print(
            f"Features: {dataset.X.shape[1]}"
        )

        print(
            f"Date range: "
            f"{dataset.data.index.min().date()} "
            f"to "
            f"{dataset.data.index.max().date()}"
        )

    return datasets