from dataclasses import dataclass

import pandas as pd

from sklearn.preprocessing import StandardScaler

from stock_ml_forecast.dataset import DatasetSplit


@dataclass
class PreparedSplit:

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

    medians: pd.Series
    scaler: StandardScaler


def preprocess_split(
    split: DatasetSplit,
) -> PreparedSplit:
    """
    Training-only preprocessing.

    1. Calculate medians from X_train only.
    2. Fill train / validation / test with those medians.
    3. Fit scaler on X_train only.
    4. Transform validation and test with same scaler.
    """

    X_train = split.X_train.copy()
    X_val = split.X_val.copy()
    X_test = split.X_test.copy()

    # --------------------------------------------------
    # Training-only imputation
    # --------------------------------------------------

    medians = X_train.median()

    X_train = X_train.fillna(
        medians
    )

    X_val = X_val.fillna(
        medians
    )

    X_test = X_test.fillna(
        medians
    )

    # --------------------------------------------------
    # Training-only scaling
    # --------------------------------------------------

    scaler = StandardScaler()

    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(
            X_train
        ),
        index=X_train.index,
        columns=X_train.columns,
    )

    X_val_scaled = pd.DataFrame(
        scaler.transform(
            X_val
        ),
        index=X_val.index,
        columns=X_val.columns,
    )

    X_test_scaled = pd.DataFrame(
        scaler.transform(
            X_test
        ),
        index=X_test.index,
        columns=X_test.columns,
    )

    return PreparedSplit(
        horizon=split.horizon,
        features=split.features,

        X_train=X_train_scaled,
        X_val=X_val_scaled,
        X_test=X_test_scaled,

        y_return_train=split.y_return_train,
        y_return_val=split.y_return_val,
        y_return_test=split.y_return_test,

        y_direction_train=split.y_direction_train,
        y_direction_val=split.y_direction_val,
        y_direction_test=split.y_direction_test,

        y_upside_train=split.y_upside_train,
        y_upside_val=split.y_upside_val,
        y_upside_test=split.y_upside_test,

        y_downside_train=split.y_downside_train,
        y_downside_val=split.y_downside_val,
        y_downside_test=split.y_downside_test,

        medians=medians,
        scaler=scaler,
    )