from dataclasses import dataclass

import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)


@dataclass
class TensorFlowForecast:
    model: keras.Model
    history: keras.callbacks.History


def build_gru_model(
    lookback: int,
    num_features: int,
) -> keras.Model:
    """
    Shared GRU with three output heads:

        direction
        upside
        downside
    """

    inputs = keras.Input(
        shape=(
            lookback,
            num_features,
        ),
        name="market_sequence",
    )

    x = layers.GRU(
        32,
        return_sequences=False,
        name="gru",
    )(inputs)

    x = layers.Dropout(
        0.20
    )(x)

    x = layers.Dense(
        32,
        activation="relu",
    )(x)

    x = layers.Dropout(
        0.10
    )(x)

    shared = layers.Dense(
        16,
        activation="relu",
        name="shared_state",
    )(x)

    # ------------------------------------------
    # Direction
    # ------------------------------------------

    direction = layers.Dense(
        1,
        activation="sigmoid",
        name="direction",
    )(shared)

    # ------------------------------------------
    # Upside
    #
    # softplus guarantees >= 0
    # ------------------------------------------

    upside = layers.Dense(
        1,
        activation="softplus",
        name="upside",
    )(shared)

    # ------------------------------------------
    # Downside
    #
    # softplus >= 0, then negate it.
    # ------------------------------------------

    downside_raw = layers.Dense(
        1,
        activation="softplus",
        name="downside_magnitude",
    )(shared)

    downside = layers.Lambda(
        lambda value: -value,
        name="downside",
    )(downside_raw)

    model = keras.Model(
        inputs=inputs,
        outputs={
            "direction": direction,
            "upside": upside,
            "downside": downside,
        },
    )

    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=0.001,
        ),
        loss={
            "direction":
                keras.losses.BinaryCrossentropy(),

            "upside":
                keras.losses.MeanSquaredError(),

            "downside":
                keras.losses.MeanSquaredError(),
        },
        loss_weights={
            "direction": 1.0,
            "upside": 1.0,
            "downside": 1.0,
        },
        metrics={
            "direction": [
                keras.metrics.BinaryAccuracy(
                    name="accuracy"
                ),
                keras.metrics.AUC(
                    name="auc"
                ),
            ],
        },
    )

    return model

def train_gru_model(
    data,
    epochs: int = 200,
    batch_size: int = 32,
) -> TensorFlowForecast:

    tf.random.set_seed(42)
    np.random.seed(42)

    model = build_gru_model(
        lookback=data.lookback,
        num_features=data.num_features,
    )

    early_stopping = (
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=20,
            restore_best_weights=True,
            min_delta=1e-4,
        )
    )

    reduce_lr = (
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=8,
            min_lr=1e-6,
        )
    )

    history = model.fit(
        data.X_train,

        {
            "direction":
                data.y_direction_train,

            "upside":
                data.y_upside_train,

            "downside":
                data.y_downside_train,
        },

        sample_weight={
            "direction":
                np.ones_like(
                    data.y_direction_train
                ),

            "upside":
                data.upside_weight_train,

            "downside":
                data.downside_weight_train,
        },

        validation_data=(
            data.X_val,

            {
                "direction":
                    data.y_direction_val,

                "upside":
                    data.y_upside_val,

                "downside":
                    data.y_downside_val,
            },

            {
                "direction":
                    np.ones_like(
                        data.y_direction_val
                    ),

                "upside":
                    data.upside_weight_val,

                "downside":
                    data.downside_weight_val,
            },
        ),

        epochs=epochs,
        batch_size=batch_size,
        callbacks=[
            early_stopping,
            reduce_lr,
        ],
        verbose=1,
    )

    return TensorFlowForecast(
        model=model,
        history=history,
    )

def predict_gru_expected_return(
    forecast: TensorFlowForecast,
    X,
    index=None,
) -> pd.DataFrame:

    prediction = forecast.model.predict(
        X,
        verbose=0,
    )

    p_up = (
        prediction["direction"]
        .reshape(-1)
    )

    expected_upside = (
        prediction["upside"]
        .reshape(-1)
    )

    expected_downside = (
        prediction["downside"]
        .reshape(-1)
    )

    p_down = (
        1.0 - p_up
    )

    expected_return = (
        p_up * expected_upside
        +
        p_down * expected_downside
    )

    return pd.DataFrame(
        {
            "P_Up": p_up,
            "P_Down": p_down,
            "Expected_Upside":
                expected_upside,
            "Expected_Downside":
                expected_downside,
            "Expected_Return":
                expected_return,
        },
        index=index,
    )

def evaluate_gru(
    forecast,
    X,
    y_return,
    y_direction,
    index=None,
) -> dict:

    predictions = (
        predict_gru_expected_return(
            forecast=forecast,
            X=X,
            index=index,
        )
    )

    expected_return = (
        predictions[
            "Expected_Return"
        ].to_numpy()
    )

    p_up = (
        predictions[
            "P_Up"
        ].to_numpy()
    )

    predicted_direction = (
        p_up >= 0.50
    ).astype(int)

    correlation = np.corrcoef(
        y_return,
        expected_return,
    )[0, 1]

    result = {
        "MAE":
            mean_absolute_error(
                y_return,
                expected_return,
            ),

        "RMSE":
            np.sqrt(
                mean_squared_error(
                    y_return,
                    expected_return,
                )
            ),

        "R2":
            r2_score(
                y_return,
                expected_return,
            ),

        "Directional_Accuracy":
            accuracy_score(
                y_direction,
                predicted_direction,
            ),

        "Balanced_Accuracy":
            balanced_accuracy_score(
                y_direction,
                predicted_direction,
            ),

        "Brier":
            brier_score_loss(
                y_direction,
                p_up,
            ),

        "Prediction_Correlation":
            correlation,
    }

    if len(
        np.unique(y_direction)
    ) >= 2:

        result["ROC_AUC"] = (
            roc_auc_score(
                y_direction,
                p_up,
            )
        )

    else:
        result["ROC_AUC"] = np.nan

    return result

