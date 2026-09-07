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
    Dataset for one forecast horizon.

    Targets:
        y_return:
            Actual forward return.

        y_direction:
            1 if forward return > 0
            0 otherwise.

        y_upside:
            Positive forward return.
            NaN when the actual return <= 0.

        y_downside:
            Negative forward return.
            NaN when the actual return >= 0.
    """

    horizon: int

    data: pd.DataFrame

    X: pd.DataFrame

    y_return: pd.Series
    y_direction: pd.Series
    y_upside: pd.Series
    y_downside: pd.Series

    features: list[str]


@dataclass
class DatasetSplit:
    """
    Purged chronological train / validation / test split.
    """

    horizon: int
    features: list[str]

    X_train: pd.DataFrame
    X_val: pd.DataFrame
    X_test: pd.DataFrame

    y_return_train: pd.Series
    y_return_val: pd.Series
    y_return_test: pd.Series

    y_direction_train: pd.Series
    y_direction_val: pd.Series
    y_direction_test: pd.Series

    y_upside_train: pd.Series
    y_upside_val: pd.Series
    y_upside_test: pd.Series

    y_downside_train: pd.Series
    y_downside_val: pd.Series
    y_downside_test: pd.Series

    purge_size: int


def add_forward_return_targets(
    df: pd.DataFrame,
    price_column: str = "Stock_Adj_Close",
    horizons: tuple[int, ...] = FORECAST_HORIZONS,
) -> pd.DataFrame:
    """
    Create four targets for each horizon:

        Forward_Return_63D
        Direction_63D
        Upside_Return_63D
        Downside_Return_63D

    Example:

        Forward_Return_63D =
            Price[t + 63] / Price[t] - 1
    """

    result = df.copy()

    for horizon in horizons:

        return_column = (
            f"Forward_Return_{horizon}D"
        )

        direction_column = (
            f"Direction_{horizon}D"
        )

        upside_column = (
            f"Upside_Return_{horizon}D"
        )

        downside_column = (
            f"Downside_Return_{horizon}D"
        )

        forward_return = (
            result[price_column].shift(-horizon)
            / result[price_column]
            - 1
        )

        result[return_column] = forward_return

        # Keep NaN where the future return is unknown.
        result[direction_column] = np.where(
            forward_return.notna(),
            (forward_return > 0).astype(float),
            np.nan,
        )

        # Conditional magnitude targets
        result[upside_column] = forward_return.where(
            forward_return > 0
        )

        result[downside_column] = forward_return.where(
            forward_return < 0
        )

    return result


def get_available_features(
    df: pd.DataFrame,
    requested_features: list[str] | None = None,
) -> list[str]:

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

        print("Missing features:")

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
    Build one clean dataset for one forecasting horizon.
    """

    if features is None:
        features = MODEL_FEATURES

    return_column = (
        f"Forward_Return_{horizon}D"
    )

    direction_column = (
        f"Direction_{horizon}D"
    )

    upside_column = (
        f"Upside_Return_{horizon}D"
    )

    downside_column = (
        f"Downside_Return_{horizon}D"
    )

    required_targets = [
        return_column,
        direction_column,
        upside_column,
        downside_column,
    ]

    for target in required_targets:

        if target not in df.columns:
            raise ValueError(
                f"{target} does not exist. "
                "Run add_forward_return_targets() first."
            )

    available_features = get_available_features(
        df=df,
        requested_features=features,
    )

    if not available_features:
        raise ValueError(
            "No model features are available."
        )

    columns = (
        available_features
        + required_targets
    )

    dataset = df[columns].copy()

    dataset = dataset.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    # Only remove rows where the actual future return
    # is not known.
    dataset = dataset.dropna(
        subset=[return_column]
    )

    if dataset.empty:
        raise ValueError(
            f"No usable rows for horizon {horizon}."
        )

    # Feature coverage check
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
            f"{horizon}D features removed "
            "for low coverage:"
        )

        for feature, ratio in removed_features.items():

            print(
                f"  {feature}: "
                f"{ratio:.1%}"
            )

    if not selected_features:
        raise ValueError(
            "No usable model features remain."
        )

    data = dataset[
        selected_features
        + required_targets
    ].copy()

    return HorizonDataset(
        horizon=horizon,
        data=data,
        X=data[selected_features].copy(),

        y_return=data[return_column].copy(),

        y_direction=(
            data[direction_column]
            .astype(int)
            .copy()
        ),

        y_upside=data[upside_column].copy(),
        y_downside=data[downside_column].copy(),

        features=selected_features,
    )


def prepare_all_horizons(
    df: pd.DataFrame,
    horizons: tuple[int, ...] = FORECAST_HORIZONS,
    features: list[str] | None = None,
    min_feature_coverage: float = 0.60,
) -> dict[int, HorizonDataset]:

    datasets = {}

    for horizon in horizons:

        print(
            f"\nPreparing {horizon}-day dataset"
        )

        dataset = prepare_horizon_dataset(
            df=df,
            horizon=horizon,
            features=features,
            min_feature_coverage=min_feature_coverage,
        )

        datasets[horizon] = dataset

        positive_rate = (
            dataset.y_direction.mean()
        )

        print(
            f"Rows: {len(dataset.data):,}"
        )

        print(
            f"Features: {dataset.X.shape[1]}"
        )

        print(
            f"Positive returns: "
            f"{positive_rate:.1%}"
        )

        print(
            "Date range: "
            f"{dataset.data.index.min().date()} "
            "-> "
            f"{dataset.data.index.max().date()}"
        )

        print("=" * 60)

    return datasets


def purged_chronological_split(
    dataset: HorizonDataset,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    purge_size: int | None = None,
) -> DatasetSplit:

    if train_ratio <= 0 or val_ratio <= 0:
        raise ValueError(
            "Ratios must be positive."
        )

    if train_ratio + val_ratio >= 1:
        raise ValueError(
            "train_ratio + val_ratio must be < 1."
        )

    X = dataset.X.sort_index()

    n = len(X)

    train_end = int(
        n * train_ratio
    )

    val_end = int(
        n * (train_ratio + val_ratio)
    )

    if purge_size is None:
        purge_size = dataset.horizon

    train_purged_end = (
        train_end - purge_size
    )

    val_purged_end = (
        val_end - purge_size
    )

    if train_purged_end <= 0:
        raise ValueError(
            "Purge removes entire training set."
        )

    if val_purged_end <= train_end:
        raise ValueError(
            "Purge removes entire validation set."
        )

    train_index = X.index[
        :train_purged_end
    ]

    val_index = X.index[
        train_end:val_purged_end
    ]

    test_index = X.index[
        val_end:
    ]

    return DatasetSplit(
        horizon=dataset.horizon,
        features=dataset.features,

        X_train=X.loc[train_index].copy(),
        X_val=X.loc[val_index].copy(),
        X_test=X.loc[test_index].copy(),

        y_return_train=(
            dataset.y_return
            .loc[train_index]
            .copy()
        ),

        y_return_val=(
            dataset.y_return
            .loc[val_index]
            .copy()
        ),

        y_return_test=(
            dataset.y_return
            .loc[test_index]
            .copy()
        ),

        y_direction_train=(
            dataset.y_direction
            .loc[train_index]
            .copy()
        ),

        y_direction_val=(
            dataset.y_direction
            .loc[val_index]
            .copy()
        ),

        y_direction_test=(
            dataset.y_direction
            .loc[test_index]
            .copy()
        ),

        y_upside_train=(
            dataset.y_upside
            .loc[train_index]
            .copy()
        ),

        y_upside_val=(
            dataset.y_upside
            .loc[val_index]
            .copy()
        ),

        y_upside_test=(
            dataset.y_upside
            .loc[test_index]
            .copy()
        ),

        y_downside_train=(
            dataset.y_downside
            .loc[train_index]
            .copy()
        ),

        y_downside_val=(
            dataset.y_downside
            .loc[val_index]
            .copy()
        ),

        y_downside_test=(
            dataset.y_downside
            .loc[test_index]
            .copy()
        ),

        purge_size=purge_size,
    )


def split_all_horizons(
    datasets: dict[int, HorizonDataset],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> dict[int, DatasetSplit]:

    splits = {}

    for horizon, dataset in datasets.items():

        split = purged_chronological_split(
            dataset=dataset,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            purge_size=horizon,
        )

        splits[horizon] = split

        print(
            f"\n{horizon}-day purged split"
        )

        print(
            f"Purge size: "
            f"{split.purge_size} rows"
        )

        print(
            f"Train: "
            f"{len(split.X_train):,} rows"
        )

        print(
            f"Validation: "
            f"{len(split.X_val):,} rows"
        )

        print(
            f"Test: "
            f"{len(split.X_test):,} rows"
        )

        print("Dates:")

        print(
            "  Train: "
            f"{split.X_train.index.min().date()} "
            "-> "
            f"{split.X_train.index.max().date()}"
        )

        print(
            "  Validation: "
            f"{split.X_val.index.min().date()} "
            "-> "
            f"{split.X_val.index.max().date()}"
        )

        print(
            "  Test: "
            f"{split.X_test.index.min().date()} "
            "-> "
            f"{split.X_test.index.max().date()}"
        )

        print("=" * 60)

    return splits