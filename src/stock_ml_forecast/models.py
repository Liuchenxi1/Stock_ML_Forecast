from dataclasses import dataclass

import numpy as np
import pandas as pd

from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)

from sklearn.linear_model import (
    LinearRegression,
    Ridge,
)

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)


# ============================================================
# Generic regression evaluation
# ============================================================


def evaluate_regression(
    model,
    X,
    y,
) -> dict:
    """
    Evaluate a regression model.
    """

    predictions = model.predict(X)

    return {
        "MAE": mean_absolute_error(
            y,
            predictions,
        ),
        "RMSE": np.sqrt(
            mean_squared_error(
                y,
                predictions,
            )
        ),
        "R2": r2_score(
            y,
            predictions,
        ),
        "Directional_Accuracy": np.mean(
            np.sign(y)
            == np.sign(predictions)
        ),
    }


# ============================================================
# Naive baselines
# ============================================================


def evaluate_naive_zero(
    y,
) -> dict:
    """
    Predict zero return for every observation.
    """

    predictions = np.zeros(
        len(y)
    )

    return {
        "MAE": mean_absolute_error(
            y,
            predictions,
        ),
        "RMSE": np.sqrt(
            mean_squared_error(
                y,
                predictions,
            )
        ),
        "R2": r2_score(
            y,
            predictions,
        ),
        "Directional_Accuracy": np.mean(
            np.sign(y)
            == np.sign(predictions)
        ),
    }


def evaluate_naive_mean(
    y_train,
    y,
) -> dict:
    """
    Predict the training-set mean return
    for every observation.
    """

    prediction_value = (
        y_train.mean()
    )

    predictions = np.full(
        len(y),
        prediction_value,
    )

    return {
        "MAE": mean_absolute_error(
            y,
            predictions,
        ),
        "RMSE": np.sqrt(
            mean_squared_error(
                y,
                predictions,
            )
        ),
        "R2": r2_score(
            y,
            predictions,
        ),
        "Directional_Accuracy": np.mean(
            np.sign(y)
            == np.sign(predictions)
        ),
    }


def evaluate_majority_direction(
    y_train,
    y,
) -> dict:
    """
    Predict whichever direction is more common
    in the training set.
    """

    majority_positive = (
        (y_train > 0).mean()
        >= 0.5
    )

    if majority_positive:
        predictions = np.ones(
            len(y)
        )
    else:
        predictions = -np.ones(
            len(y)
        )

    return {
        "Directional_Accuracy": np.mean(
            np.sign(y)
            == np.sign(predictions)
        ),
    }


# ============================================================
# Linear Regression
# ============================================================


def train_linear_model(
    data,
):
    """
    Train Linear Regression.
    """

    model = LinearRegression()

    model.fit(
        data.X_train,
        data.y_return_train,
    )

    return model


# ============================================================
# Ridge Regression
# ============================================================


def train_ridge_model(
    data,
    alpha: float = 1.0,
):
    """
    Train Ridge Regression.
    """

    model = Ridge(
        alpha=alpha,
    )

    model.fit(
        data.X_train,
        data.y_return_train,
    )

    return model


def tune_ridge(
    data,
    alphas: list[float] | None = None,
):
    """
    Tune Ridge alpha using validation MAE only.
    """

    if alphas is None:
        alphas = [
            0.001,
            0.01,
            0.1,
            1.0,
            10.0,
            100.0,
            1000.0,
        ]

    results = []

    for alpha in alphas:

        model = Ridge(
            alpha=alpha,
        )

        model.fit(
            data.X_train,
            data.y_return_train,
        )

        predictions = model.predict(
            data.X_val
        )

        mae = mean_absolute_error(
            data.y_return_val,
            predictions,
        )

        rmse = np.sqrt(
            mean_squared_error(
                data.y_return_val,
                predictions,
            )
        )

        r2 = r2_score(
            data.y_return_val,
            predictions,
        )

        results.append(
            {
                "alpha": alpha,
                "MAE": mae,
                "RMSE": rmse,
                "R2": r2,
            }
        )

    results = sorted(
        results,
        key=lambda item: item["MAE"],
    )

    best_alpha = (
        results[0]["alpha"]
    )

    best_model = Ridge(
        alpha=best_alpha,
    )

    best_model.fit(
        data.X_train,
        data.y_return_train,
    )

    return (
        best_model,
        results,
    )


# ============================================================
# Direction + Upside + Downside architecture
# ============================================================


@dataclass
class DirectionMagnitudeModels:
    """
    Three-model architecture:

    1. Direction model:
       predicts P(up)

    2. Upside model:
       predicts return magnitude when return > 0

    3. Downside model:
       predicts return magnitude when return < 0
    """

    direction_model: HistGradientBoostingClassifier
    upside_model: HistGradientBoostingRegressor
    downside_model: HistGradientBoostingRegressor


def train_direction_magnitude_models(
    data,
) -> DirectionMagnitudeModels:
    """
    Train:

        Direction classifier
        Upside regressor
        Downside regressor
    """

    # ========================================================
    # Direction model
    # ========================================================

    direction_model = (
        HistGradientBoostingClassifier(
            max_iter=200,
            learning_rate=0.05,
            max_leaf_nodes=15,
            l2_regularization=1.0,
            random_state=42,
        )
    )

    direction_model.fit(
        data.X_train,
        data.y_direction_train,
    )

    # ========================================================
    # Upside magnitude model
    # ========================================================

    upside_mask = (
        data.y_upside_train.notna()
    )

    if upside_mask.sum() == 0:
        raise ValueError(
            "No positive-return training observations "
            "available for the upside model."
        )

    upside_model = (
        HistGradientBoostingRegressor(
            max_iter=200,
            learning_rate=0.05,
            max_leaf_nodes=15,
            l2_regularization=1.0,
            random_state=42,
        )
    )

    upside_model.fit(
        data.X_train.loc[
            upside_mask
        ],
        data.y_upside_train.loc[
            upside_mask
        ],
    )

    # ========================================================
    # Downside magnitude model
    # ========================================================

    downside_mask = (
        data.y_downside_train.notna()
    )

    if downside_mask.sum() == 0:
        raise ValueError(
            "No negative-return training observations "
            "available for the downside model."
        )

    downside_model = (
        HistGradientBoostingRegressor(
            max_iter=200,
            learning_rate=0.05,
            max_leaf_nodes=15,
            l2_regularization=1.0,
            random_state=42,
        )
    )

    downside_model.fit(
        data.X_train.loc[
            downside_mask
        ],
        data.y_downside_train.loc[
            downside_mask
        ],
    )

    return DirectionMagnitudeModels(
        direction_model=direction_model,
        upside_model=upside_model,
        downside_model=downside_model,
    )


# ============================================================
# Direction + magnitude prediction
# ============================================================


def predict_expected_return(
    models: DirectionMagnitudeModels,
    X: pd.DataFrame,
) -> pd.DataFrame:
    """
    Produce:

        P(up)
        P(down)
        expected upside
        expected downside
        expected return

    Formula:

        E[R]
        =
        P(up) * E[R | up]
        +
        P(down) * E[R | down]
    """

    # Probability of positive return
    p_up = (
        models
        .direction_model
        .predict_proba(X)[:, 1]
    )

    p_down = (
        1.0 - p_up
    )

    # Conditional upside prediction
    expected_upside = (
        models
        .upside_model
        .predict(X)
    )

    # Conditional downside prediction
    expected_downside = (
        models
        .downside_model
        .predict(X)
    )

    # Make sure each branch keeps
    # its intended economic meaning.
    expected_upside = np.maximum(
        expected_upside,
        0.0,
    )

    expected_downside = np.minimum(
        expected_downside,
        0.0,
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
        index=X.index,
    )


# ============================================================
# Direction + magnitude evaluation
# ============================================================


def evaluate_direction_magnitude(
    models: DirectionMagnitudeModels,
    X: pd.DataFrame,
    y_return: pd.Series,
    y_direction: pd.Series,
) -> dict:
    """
    Evaluate both:

        return prediction quality
        direction probability quality
    """

    forecast = (
        predict_expected_return(
            models=models,
            X=X,
        )
    )

    expected_return = (
        forecast[
            "Expected_Return"
        ]
    )

    p_up = (
        forecast[
            "P_Up"
        ]
    )

    predicted_direction = (
        p_up >= 0.50
    ).astype(int)

    rmse = np.sqrt(
        mean_squared_error(
            y_return,
            expected_return,
        )
    )

    # Correlation can fail if one side is constant.
    if (
        np.std(y_return) == 0
        or
        np.std(expected_return) == 0
    ):
        correlation = np.nan

    else:
        correlation = np.corrcoef(
            y_return,
            expected_return,
        )[0, 1]

    metrics = {
        "MAE": mean_absolute_error(
            y_return,
            expected_return,
        ),

        "RMSE": rmse,

        "R2": r2_score(
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

    # ROC-AUC requires both classes
    # to exist in the evaluation dataset.
    if y_direction.nunique() >= 2:

        metrics["ROC_AUC"] = (
            roc_auc_score(
                y_direction,
                p_up,
            )
        )

    else:

        metrics["ROC_AUC"] = (
            np.nan
        )

    return metrics


# ============================================================
# Conditional magnitude evaluation
# ============================================================


def evaluate_conditional_magnitude(
    models: DirectionMagnitudeModels,
    X: pd.DataFrame,
    y_upside: pd.Series,
    y_downside: pd.Series,
) -> dict:
    """
    Evaluate upside and downside magnitude models separately.

    This helps answer:

        When the stock actually goes up,
        how accurate is the upside model?

        When it actually goes down,
        how accurate is the downside model?
    """

    results = {}

    # ========================================================
    # Upside
    # ========================================================

    upside_mask = (
        y_upside.notna()
    )

    if upside_mask.sum() > 0:

        upside_predictions = (
            models
            .upside_model
            .predict(
                X.loc[
                    upside_mask
                ]
            )
        )

        upside_predictions = (
            np.maximum(
                upside_predictions,
                0.0,
            )
        )

        results[
            "Upside_Count"
        ] = int(
            upside_mask.sum()
        )

        results[
            "Upside_MAE"
        ] = mean_absolute_error(
            y_upside.loc[
                upside_mask
            ],
            upside_predictions,
        )

        results[
            "Upside_RMSE"
        ] = np.sqrt(
            mean_squared_error(
                y_upside.loc[
                    upside_mask
                ],
                upside_predictions,
            )
        )

    else:

        results[
            "Upside_Count"
        ] = 0

        results[
            "Upside_MAE"
        ] = np.nan

        results[
            "Upside_RMSE"
        ] = np.nan

    # ========================================================
    # Downside
    # ========================================================

    downside_mask = (
        y_downside.notna()
    )

    if downside_mask.sum() > 0:

        downside_predictions = (
            models
            .downside_model
            .predict(
                X.loc[
                    downside_mask
                ]
            )
        )

        downside_predictions = (
            np.minimum(
                downside_predictions,
                0.0,
            )
        )

        results[
            "Downside_Count"
        ] = int(
            downside_mask.sum()
        )

        results[
            "Downside_MAE"
        ] = mean_absolute_error(
            y_downside.loc[
                downside_mask
            ],
            downside_predictions,
        )

        results[
            "Downside_RMSE"
        ] = np.sqrt(
            mean_squared_error(
                y_downside.loc[
                    downside_mask
                ],
                downside_predictions,
            )
        )

    else:

        results[
            "Downside_Count"
        ] = 0

        results[
            "Downside_MAE"
        ] = np.nan

        results[
            "Downside_RMSE"
        ] = np.nan

    return results