import pandas as pd
from sklearn.preprocessing import StandardScaler


class PreparedSplit:
    def __init__(
        self,
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        medians,
        scaler,
        features,
    ):
        self.X_train = X_train
        self.X_val = X_val
        self.X_test = X_test

        self.y_train = y_train
        self.y_val = y_val
        self.y_test = y_test

        self.medians = medians
        self.scaler = scaler
        self.features = features


def preprocess_split(split) -> PreparedSplit:
    """
    Impute missing values using TRAIN medians only,
    then fit StandardScaler on TRAIN only.
    """

    X_train = split.X_train.copy()
    X_val = split.X_val.copy()
    X_test = split.X_test.copy()

    # 1. Training-only median
    medians = X_train.median()

    X_train = X_train.fillna(medians)
    X_val = X_val.fillna(medians)
    X_test = X_test.fillna(medians)

    # 2. Fit scaler on training only
    scaler = StandardScaler()

    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        index=X_train.index,
        columns=X_train.columns,
    )

    X_val_scaled = pd.DataFrame(
        scaler.transform(X_val),
        index=X_val.index,
        columns=X_val.columns,
    )

    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        index=X_test.index,
        columns=X_test.columns,
    )

    return PreparedSplit(
        X_train=X_train_scaled,
        X_val=X_val_scaled,
        X_test=X_test_scaled,
        y_train=split.y_train.copy(),
        y_val=split.y_val.copy(),
        y_test=split.y_test.copy(),
        medians=medians,
        scaler=scaler,
        features=list(X_train.columns),
    )