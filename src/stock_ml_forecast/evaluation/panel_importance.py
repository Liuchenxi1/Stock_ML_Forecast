import pandas as pd

from sklearn.inspection import (
    permutation_importance,
)


def calculate_feature_importance(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    n_repeats: int = 5,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Measure classifier feature importance using
    permutation importance and ROC-AUC.
    """

    result = permutation_importance(
        estimator=model,
        X=X,
        y=y,
        scoring="roc_auc",
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=-1,
    )

    importance = pd.DataFrame(
        {
            "Feature": X.columns,
            "Importance_Mean":
                result.importances_mean,
            "Importance_Std":
                result.importances_std,
        }
    )

    return (
        importance
        .sort_values(
            "Importance_Mean",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )


def print_feature_importance(
    importance: pd.DataFrame,
    top_n: int = 22,
) -> None:

    print(
        "\nClassifier feature importance:"
    )

    print(
        importance
        .head(top_n)
        .to_string(
            index=False
        )
    )

    print("=" * 60)