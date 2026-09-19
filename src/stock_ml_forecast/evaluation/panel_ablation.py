import pandas as pd

from stock_ml_forecast.macro.panel_features import (
    MODEL_FEATURES_252D,
    MARKET_FEATURES,
    FUNDAMENTAL_FEATURES,
)

from stock_ml_forecast.modeling.panel_models import (
    train_panel_models,
)

from stock_ml_forecast.evaluation.panel_backtest import (
    run_monthly_ranking_backtest,
)


def run_feature_ablation(
    split,
) -> pd.DataFrame:
    """
    Compare three feature groups:

    1. ALL
    2. MARKET + MACRO
    3. FUNDAMENTALS ONLY

    Uses the existing validation and test split.
    Model parameters remain unchanged.
    """

    experiments = {
        "ALL": MODEL_FEATURES_252D,
        "MARKET_MACRO": MARKET_FEATURES,
        "FUNDAMENTALS": FUNDAMENTAL_FEATURES,
    }

    rows = []

    for experiment_name, features in experiments.items():

        print(
            f"\nAblation model: {experiment_name}"
        )

        print(
            f"Feature count: {len(features)}"
        )

        # ====================================================
        # Select feature group
        # ====================================================

        X_train = (
            split.X_train[
                features
            ]
            .copy()
        )

        X_validation = (
            split.X_validation[
                features
            ]
            .copy()
        )

        X_test = (
            split.X_test[
                features
            ]
            .copy()
        )

        # ====================================================
        # Train
        # ====================================================

        models = train_panel_models(
            X_train=X_train,
            y_top10_train=split.y_top10_train,
            y_return_train=split.y_return_train,
        )

        # ====================================================
        # Validation monthly ranking
        # ====================================================

        validation_result = (
            run_monthly_ranking_backtest(
                classifier=models.classifier,
                X=X_validation,
                y_top10=split.y_top10_validation,
                y_return=split.y_return_validation,
                label="Validation",
                include_momentum_baselines=False,
            )
        )

        validation_ml = (
            validation_result.summary[
                validation_result.summary[
                    "Method"
                ]
                == "ML Top-10 Probability"
            ]
            .iloc[0]
        )

        # ====================================================
        # Test monthly ranking
        # ====================================================

        test_result = (
            run_monthly_ranking_backtest(
                classifier=models.classifier,
                X=X_test,
                y_top10=split.y_top10_test,
                y_return=split.y_return_test,
                label="Test",
            )
        )

        test_ml = (
            test_result.summary[
                test_result.summary[
                    "Method"
                ]
                == "ML Top-10 Probability"
            ]
            .iloc[0]
        )

        # ====================================================
        # Save summary
        # ====================================================

        rows.append(
            {
                "Model":
                    experiment_name,

                "Feature_Count":
                    len(features),

                "Validation_Precision_10":
                    validation_ml[
                        "Precision_at_10pct"
                    ],

                "Validation_Return_Lift":
                    validation_ml[
                        "Return_Lift"
                    ],

                "Test_Precision_10":
                    test_ml[
                        "Precision_at_10pct"
                    ],

                "Test_Return_Lift":
                    test_ml[
                        "Return_Lift"
                    ],

                "Test_Mean_Selected_Return":
                    test_ml[
                        "Mean_Selected_Excess_Return"
                    ],

                "Test_Median_Selected_Return":
                    test_ml[
                        "Median_Selected_Excess_Return"
                    ],
            }
        )

    return pd.DataFrame(
        rows
    )


def print_ablation_summary(
    result: pd.DataFrame,
) -> None:

    print(
        "\nFeature ablation summary:"
    )

    for _, row in result.iterrows():

        print(
            f"\n{row['Model']}"
        )

        print(
            f"{'Feature count':<32}"
            f"{int(row['Feature_Count'])}"
        )

        print(
            f"{'Validation precision @ 10%':<32}"
            f"{row['Validation_Precision_10']:.2%}"
        )

        print(
            f"{'Validation return lift':<32}"
            f"{row['Validation_Return_Lift']:.2%}"
        )

        print(
            f"{'Test precision @ 10%':<32}"
            f"{row['Test_Precision_10']:.2%}"
        )

        print(
            f"{'Test return lift':<32}"
            f"{row['Test_Return_Lift']:.2%}"
        )

        print(
            f"{'Test mean selected return':<32}"
            f"{row['Test_Mean_Selected_Return']:.2%}"
        )

        print(
            f"{'Test median selected return':<32}"
            f"{row['Test_Median_Selected_Return']:.2%}"
        )

    print("=" * 60)