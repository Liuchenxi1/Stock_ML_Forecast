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

@dataclass
class DatasetSplit:
    """
    Chronological train / validation / test split
    for one forecasting horizon.
    """

    horizon: int
    target_column: str
    features: list[str]

    X_train: pd.DataFrame
    y_train: pd.Series

    X_val: pd.DataFrame
    y_val: pd.Series

    X_test: pd.DataFrame
    y_test: pd.Series

    purge_size: int

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

def purged_chronological_split(
    dataset: HorizonDataset,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    purge_size: int | None = None,
) -> DatasetSplit:
    """
    Split one horizon dataset chronologically into:

        train -> validation -> test

    and purge observations near split boundaries whose
    forward-return targets overlap the next period.

    Parameters
    ----------
    dataset:
        HorizonDataset created by prepare_horizon_dataset().

    train_ratio:
        Fraction of observations assigned to the initial
        training region.

    val_ratio:
        Fraction assigned to the initial validation region.

    purge_size:
        Number of observations removed from the END of
        train and validation.

        If None, defaults to the forecast horizon.

        Example:
            horizon = 252
            purge_size = 252

    Returns
    -------
    DatasetSplit
    """

    if not 0 < train_ratio < 1:
        raise ValueError(
            "train_ratio must be between 0 and 1."
        )

    if not 0 < val_ratio < 1:
        raise ValueError(
            "val_ratio must be between 0 and 1."
        )

    if train_ratio + val_ratio >= 1:
        raise ValueError(
            "train_ratio + val_ratio must be less than 1."
        )

    # -------------------------------------------------
    # Make sure everything is chronological
    # -------------------------------------------------

    X = dataset.X.sort_index()
    y = dataset.y.loc[X.index]

    n = len(X)

    if n < 3:
        raise ValueError(
            "Dataset is too small to split."
        )

    # -------------------------------------------------
    # Raw split boundaries
    # -------------------------------------------------

    train_end = int(
        n * train_ratio
    )

    val_end = int(
        n * (train_ratio + val_ratio)
    )

    if purge_size is None:
        purge_size = dataset.horizon

    if purge_size < 0:
        raise ValueError(
            "purge_size cannot be negative."
        )

    # -------------------------------------------------
    # Purged boundaries
    #
    # Raw:
    #
    # TRAIN      | VALIDATION | TEST
    #
    # Purged:
    #
    # TRAIN |gap| VALIDATION |gap| TEST
    #       H                  H
    # -------------------------------------------------

    train_purged_end = (
        train_end - purge_size
    )

    val_purged_end = (
        val_end - purge_size
    )

    if train_purged_end <= 0:
        raise ValueError(
            f"Purge size {purge_size} is too large "
            "for the training split."
        )

    if val_purged_end <= train_end:
        raise ValueError(
            f"Purge size {purge_size} is too large "
            "for the validation split."
        )

    # -------------------------------------------------
    # Create splits
    # -------------------------------------------------

    X_train = X.iloc[
        :train_purged_end
    ].copy()

    y_train = y.iloc[
        :train_purged_end
    ].copy()

    # Validation STARTS at the original train boundary.
    # The rows between train_purged_end and train_end
    # are intentionally unused.
    X_val = X.iloc[
        train_end:val_purged_end
    ].copy()

    y_val = y.iloc[
        train_end:val_purged_end
    ].copy()

    # Test begins at the original validation boundary.
    X_test = X.iloc[
        val_end:
    ].copy()

    y_test = y.iloc[
        val_end:
    ].copy()

    # -------------------------------------------------
    # Safety checks
    # -------------------------------------------------

    if X_train.empty:
        raise ValueError(
            "Training split is empty."
        )

    if X_val.empty:
        raise ValueError(
            "Validation split is empty."
        )

    if X_test.empty:
        raise ValueError(
            "Test split is empty."
        )

    return DatasetSplit(
        horizon=dataset.horizon,
        target_column=dataset.target_column,
        features=list(X.columns),

        X_train=X_train,
        y_train=y_train,

        X_val=X_val,
        y_val=y_val,

        X_test=X_test,
        y_test=y_test,

        purge_size=purge_size,
    )

def split_all_horizons(
    datasets: dict[int, HorizonDataset],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> dict[int, DatasetSplit]:
    """
    Apply purged chronological splitting to every
    forecasting horizon.
    """

    splits = {}

    for horizon, dataset in datasets.items():

        split = purged_chronological_split(
            dataset=dataset,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            purge_size=horizon,
        )

        splits[horizon] = split

        print("\n" + "=" * 60)
        print(
            f"{horizon}-day purged split"
        )

        print(
            f"Purge size: {split.purge_size} rows"
        )

        print(
            f"Train: {len(split.X_train):,} rows"
        )

        print(
            f"Validation: {len(split.X_val):,} rows"
        )

        print(
            f"Test: {len(split.X_test):,} rows"
        )

        print(
            "\nDates:"
        )

        print(
            f"Train: "
            f"{split.X_train.index.min().date()} "
            f"-> "
            f"{split.X_train.index.max().date()}"
        )

        print(
            f"Validation: "
            f"{split.X_val.index.min().date()} "
            f"-> "
            f"{split.X_val.index.max().date()}"
        )

        print(
            f"Test: "
            f"{split.X_test.index.min().date()} "
            f"-> "
            f"{split.X_test.index.max().date()}"
        )

    return splits