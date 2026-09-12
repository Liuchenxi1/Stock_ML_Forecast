from dataclasses import dataclass

import numpy as np


@dataclass
class SequenceSplit:
    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray

    y_direction_train: np.ndarray
    y_direction_val: np.ndarray
    y_direction_test: np.ndarray

    y_upside_train: np.ndarray
    y_upside_val: np.ndarray
    y_upside_test: np.ndarray

    y_downside_train: np.ndarray
    y_downside_val: np.ndarray
    y_downside_test: np.ndarray

    upside_weight_train: np.ndarray
    upside_weight_val: np.ndarray
    upside_weight_test: np.ndarray

    downside_weight_train: np.ndarray
    downside_weight_val: np.ndarray
    downside_weight_test: np.ndarray

    y_return_train: np.ndarray
    y_return_val: np.ndarray
    y_return_test: np.ndarray

    train_index: np.ndarray
    val_index: np.ndarray
    test_index: np.ndarray

    lookback: int
    num_features: int


def _make_sequences(
    X,
    y_direction,
    y_upside,
    y_downside,
    y_return,
    lookback: int,
):
    X_seq = []

    direction_seq = []
    upside_seq = []
    downside_seq = []
    return_seq = []

    upside_weights = []
    downside_weights = []

    indices = []

    for i in range(
        lookback - 1,
        len(X),
    ):

        start = (
            i - lookback + 1
        )

        X_seq.append(
            X.iloc[
                start:i + 1
            ].to_numpy(
                dtype=np.float32
            )
        )

        direction_seq.append(
            float(
                y_direction.iloc[i]
            )
        )

        return_value = float(
            y_return.iloc[i]
        )

        return_seq.append(
            return_value
        )

        upside_value = (
            y_upside.iloc[i]
        )

        downside_value = (
            y_downside.iloc[i]
        )

        # Keras should not receive NaN targets.
        # Fill irrelevant branch with zero and use
        # sample weight 0 so that loss is ignored.

        if np.isnan(upside_value):
            upside_seq.append(0.0)
            upside_weights.append(0.0)
        else:
            upside_seq.append(
                float(upside_value)
            )
            upside_weights.append(1.0)

        if np.isnan(downside_value):
            downside_seq.append(0.0)
            downside_weights.append(0.0)
        else:
            downside_seq.append(
                float(downside_value)
            )
            downside_weights.append(1.0)

        indices.append(
            X.index[i]
        )

    return (
        np.asarray(
            X_seq,
            dtype=np.float32,
        ),
        np.asarray(
            direction_seq,
            dtype=np.float32,
        ),
        np.asarray(
            upside_seq,
            dtype=np.float32,
        ),
        np.asarray(
            downside_seq,
            dtype=np.float32,
        ),
        np.asarray(
            return_seq,
            dtype=np.float32,
        ),
        np.asarray(
            upside_weights,
            dtype=np.float32,
        ),
        np.asarray(
            downside_weights,
            dtype=np.float32,
        ),
        np.asarray(indices),
    )


def build_sequence_split(
    prepared_split,
    lookback: int = 60,
) -> SequenceSplit:
    """
    Convert existing preprocessed tabular splits
    into GRU sequences.

    Each sequence contains the previous `lookback`
    trading observations including the current row.

    For this first experiment, train/validation/test
    sequences are constructed independently so no
    sequence crosses a split boundary.
    """

    train = _make_sequences(
        X=prepared_split.X_train,
        y_direction=prepared_split.y_direction_train,
        y_upside=prepared_split.y_upside_train,
        y_downside=prepared_split.y_downside_train,
        y_return=prepared_split.y_return_train,
        lookback=lookback,
    )

    val = _make_sequences(
        X=prepared_split.X_val,
        y_direction=prepared_split.y_direction_val,
        y_upside=prepared_split.y_upside_val,
        y_downside=prepared_split.y_downside_val,
        y_return=prepared_split.y_return_val,
        lookback=lookback,
    )

    test = _make_sequences(
        X=prepared_split.X_test,
        y_direction=prepared_split.y_direction_test,
        y_upside=prepared_split.y_upside_test,
        y_downside=prepared_split.y_downside_test,
        y_return=prepared_split.y_return_test,
        lookback=lookback,
    )

    return SequenceSplit(
        X_train=train[0],
        X_val=val[0],
        X_test=test[0],

        y_direction_train=train[1],
        y_direction_val=val[1],
        y_direction_test=test[1],

        y_upside_train=train[2],
        y_upside_val=val[2],
        y_upside_test=test[2],

        y_downside_train=train[3],
        y_downside_val=val[3],
        y_downside_test=test[3],

        y_return_train=train[4],
        y_return_val=val[4],
        y_return_test=test[4],

        upside_weight_train=train[5],
        upside_weight_val=val[5],
        upside_weight_test=test[5],

        downside_weight_train=train[6],
        downside_weight_val=val[6],
        downside_weight_test=test[6],

        train_index=train[7],
        val_index=val[7],
        test_index=test[7],

        lookback=lookback,
        num_features=prepared_split.X_train.shape[1],
    )