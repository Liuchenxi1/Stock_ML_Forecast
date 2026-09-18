from dataclasses import dataclass

import numpy as np
import pandas as pd

from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
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


@dataclass
class PanelModels:
    classifier: HistGradientBoostingClassifier
    regressor: HistGradientBoostingRegressor


def train_panel_models(
    X_train: pd.DataFrame,
    y_top10_train: pd.Series,
    y_return_train: pd.Series,
) -> PanelModels:
    """
    Train first baseline panel models.

    Classifier:
        Predict probability of being in future top 10%.

    Regressor:
        Predict future 252D excess return vs SPY.
    """

    classifier = (
        HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_iter=200,
            max_leaf_nodes=31,
            l2_regularization=1.0,
            random_state=42,
        )
    )

    regressor = (
        HistGradientBoostingRegressor(
            learning_rate=0.05,
            max_iter=200,
            max_leaf_nodes=31,
            l2_regularization=1.0,
            random_state=42,
        )
    )

    print(
        "Training Top-10% classifier..."
    )

    classifier.fit(
        X_train,
        y_top10_train,
    )

    print(
        "Training excess-return regressor..."
    )

    regressor.fit(
        X_train,
        y_return_train,
    )

    print("=" * 60)

    return PanelModels(
        classifier=classifier,
        regressor=regressor,
    )


def evaluate_classifier(
    model: HistGradientBoostingClassifier,
    X: pd.DataFrame,
    y: pd.Series,
    label: str,
) -> dict:
    """
    Evaluate Top-10% classification performance.
    """

    probability = (
        model.predict_proba(X)[:, 1]
    )

    prediction = (
        probability >= 0.50
    ).astype(int)

    metrics = {
        "Accuracy": accuracy_score(
            y,
            prediction,
        ),
        "Balanced_Accuracy":
            balanced_accuracy_score(
                y,
                prediction,
            ),
        "ROC_AUC":
            roc_auc_score(
                y,
                probability,
            ),
        "Brier":
            brier_score_loss(
                y,
                probability,
            ),
    }

    print(
        f"\n{label} classifier:"
    )

    for name, value in metrics.items():

        print(
            f"{name:<20} "
            f"{value:.4f}"
        )

    return metrics


def evaluate_regressor(
    model: HistGradientBoostingRegressor,
    X: pd.DataFrame,
    y: pd.Series,
    label: str,
) -> dict:
    """
    Evaluate 252D excess-return regression.
    """

    prediction = model.predict(
        X
    )

    correlation = (
        np.corrcoef(
            y,
            prediction,
        )[0, 1]
    )

    metrics = {
        "MAE":
            mean_absolute_error(
                y,
                prediction,
            ),
        "RMSE":
            np.sqrt(
                mean_squared_error(
                    y,
                    prediction,
                )
            ),
        "R2":
            r2_score(
                y,
                prediction,
            ),
        "Correlation":
            correlation,
    }

    print(
        f"\n{label} regressor:"
    )

    for name, value in metrics.items():

        print(
            f"{name:<20} "
            f"{value:.4f}"
        )

    return metrics


def evaluate_mean_baseline(
    y_train: pd.Series,
    y: pd.Series,
    label: str,
) -> dict:
    """
    Regression baseline:
    always predict the training-set mean.
    """

    prediction = np.full(
        len(y),
        y_train.mean(),
    )

    metrics = {
        "MAE":
            mean_absolute_error(
                y,
                prediction,
            ),
        "RMSE":
            np.sqrt(
                mean_squared_error(
                    y,
                    prediction,
                )
            ),
        "R2":
            r2_score(
                y,
                prediction,
            ),
    }

    print(
        f"\n{label} mean baseline:"
    )

    for name, value in metrics.items():

        print(
            f"{name:<20} "
            f"{value:.4f}"
        )

    return metrics