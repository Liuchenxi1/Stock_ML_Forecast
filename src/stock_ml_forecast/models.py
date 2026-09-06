from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score

import numpy as np


def evaluate_regression(
    model,
    X,
    y,
) -> dict:
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


def train_linear_model(data):
    model = LinearRegression()

    model.fit(
        data.X_train,
        data.y_train,
    )

    return model


def train_ridge_model(
    data,
    alpha: float = 1.0,
):
    model = Ridge(
        alpha=alpha,
    )

    model.fit(
        data.X_train,
        data.y_train,
    )

    return model

def evaluate_naive_zero(y) -> dict:
    predictions = np.zeros(len(y))

    return {
        "MAE": mean_absolute_error(y, predictions),
        "RMSE": np.sqrt(
            mean_squared_error(y, predictions)
        ),
        "R2": r2_score(y, predictions),
        "Directional_Accuracy": np.mean(
            np.sign(y) == np.sign(predictions)
        ),
    }


def evaluate_naive_mean(
    y_train,
    y,
) -> dict:
    prediction_value = y_train.mean()

    predictions = np.full(
        len(y),
        prediction_value,
    )

    return {
        "MAE": mean_absolute_error(y, predictions),
        "RMSE": np.sqrt(
            mean_squared_error(y, predictions)
        ),
        "R2": r2_score(y, predictions),
        "Directional_Accuracy": np.mean(
            np.sign(y) == np.sign(predictions)
        ),
    }