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

def evaluate_top_decile_ranking(
    classifier,
    regressor,
    X: pd.DataFrame,
    y_top10: pd.Series,
    y_return: pd.Series,
    label: str,
    top_fraction: float = 0.10,
) -> dict:
    """
    Evaluate the model as a cross-sectional stock ranker.

    For every Date:
        1. predict P(Top 10%)
        2. rank stocks
        3. select highest-scoring 10%
        4. compare with actual future outcomes
    """

    result = pd.DataFrame(
        index=X.index
    )

    result["Probability_Top10"] = (
        classifier.predict_proba(X)[:, 1]
    )

    result["Predicted_Excess_Return"] = (
        regressor.predict(X)
    )

    result["Actual_Top10"] = (
        y_top10.astype(int)
    )

    result["Actual_Excess_Return"] = (
        y_return.astype(float)
    )

    daily_results = []

    for date, group in result.groupby(
        level="Date"
    ):

        if len(group) < 10:
            continue

        number_selected = max(
            1,
            int(
                len(group)
                * top_fraction
            ),
        )

        selected = (
            group
            .nlargest(
                number_selected,
                "Probability_Top10",
            )
        )

        actual_positives = (
            group["Actual_Top10"]
            .sum()
        )

        true_positives = (
            selected["Actual_Top10"]
            .sum()
        )

        precision = (
            selected[
                "Actual_Top10"
            ]
            .mean()
        )

        if actual_positives > 0:

            recall = (
                true_positives
                / actual_positives
            )

        else:

            recall = np.nan

        selected_return = (
            selected[
                "Actual_Excess_Return"
            ]
            .mean()
        )

        universe_return = (
            group[
                "Actual_Excess_Return"
            ]
            .mean()
        )

        daily_results.append(
            {
                "Date": date,
                "Precision_at_10pct":
                    precision,

                "Recall_at_10pct":
                    recall,

                "Selected_Excess_Return":
                    selected_return,

                "Universe_Excess_Return":
                    universe_return,

                "Return_Lift":
                    selected_return
                    - universe_return,
            }
        )

    daily = pd.DataFrame(
        daily_results
    )

    metrics = {
        "Precision_at_10pct":
            daily[
                "Precision_at_10pct"
            ].mean(),

        "Recall_at_10pct":
            daily[
                "Recall_at_10pct"
            ].mean(),

        "Selected_Excess_Return":
            daily[
                "Selected_Excess_Return"
            ].mean(),

        "Universe_Excess_Return":
            daily[
                "Universe_Excess_Return"
            ].mean(),

        "Return_Lift":
            daily[
                "Return_Lift"
            ].mean(),
    }

    print(
        f"\n{label} ranking:"
    )

    print(
        f"{'Precision @ 10%':<30}"
        f"{metrics['Precision_at_10pct']:.2%}"
    )

    print(
        f"{'Recall @ 10%':<30}"
        f"{metrics['Recall_at_10pct']:.2%}"
    )

    print(
        f"{'Selected actual excess return':<30}"
        f"{metrics['Selected_Excess_Return']:.2%}"
    )

    print(
        f"{'Universe actual excess return':<30}"
        f"{metrics['Universe_Excess_Return']:.2%}"
    )

    print(
        f"{'Excess-return lift':<30}"
        f"{metrics['Return_Lift']:.2%}"
    )

    return metrics