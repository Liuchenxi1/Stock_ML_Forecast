from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

from stock_ml_forecast.macro.panel_features import MARKET_FEATURES, MODEL_FEATURES_252D
from stock_ml_forecast.modeling.panel_models import train_panel_models
from stock_ml_forecast.modeling.panel_targets import add_252d_targets

TOP_FRACTION = 0.10

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LONG_HISTORY_PANEL_PATH = (
    PROJECT_ROOT / "data" / "processed" / "sp500_panel_2006_experiment.parquet"
)
RESULTS_DIR = (
    PROJECT_ROOT / "long_history_model" / "results" / "all_features_experiment"
)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SUMMARY_PATH = RESULTS_DIR / "all_features_long_history_walk_forward.csv"
PREDICTIONS_PATH = RESULTS_DIR / "all_features_long_history_predictions.parquet"

# Previous 2006-history MARKET_MACRO (24-feature) results.
MARKET_MACRO_2006_PRECISION = {
    1: 0.3499,
    2: 0.2857,
    3: 0.2303,
    4: 0.2180,
    5: 0.1686,
    6: 0.2971,
    7: 0.2886,
    8: 0.2771,
}

# Exactly the same validation periods as the previous experiment.
FOLDS = [
    {"fold": 1, "train_end": "2019-01-03", "validation_start": "2020-01-06", "validation_end": "2020-07-06"},
    {"fold": 2, "train_end": "2019-07-05", "validation_start": "2020-07-07", "validation_end": "2021-01-04"},
    {"fold": 3, "train_end": "2020-01-03", "validation_start": "2021-01-05", "validation_end": "2021-07-06"},
    {"fold": 4, "train_end": "2020-07-06", "validation_start": "2021-07-07", "validation_end": "2022-01-03"},
    {"fold": 5, "train_end": "2021-01-04", "validation_start": "2022-01-04", "validation_end": "2022-07-06"},
    {"fold": 6, "train_end": "2021-07-06", "validation_start": "2022-07-07", "validation_end": "2023-01-04"},
    {"fold": 7, "train_end": "2022-01-03", "validation_start": "2023-01-05", "validation_end": "2023-07-07"},
    {"fold": 8, "train_end": "2022-07-06", "validation_start": "2023-07-10", "validation_end": "2024-01-05"},
]


def sep() -> None:
    print("=" * 60)


def load_panel() -> pd.DataFrame:
    if not LONG_HISTORY_PANEL_PATH.exists():
        raise FileNotFoundError(
            f"Long-history panel not found:\n{LONG_HISTORY_PANEL_PATH}"
        )

    panel = pd.read_parquet(LONG_HISTORY_PANEL_PATH)
    dates = pd.DatetimeIndex(panel.index.get_level_values("Date"))

    print("\nLoading long-history panel:")
    print(LONG_HISTORY_PANEL_PATH)
    print("Rows:", f"{len(panel):,}")
    print("Companies:", panel.index.get_level_values("Ticker").nunique())
    print("Date range:", dates.min().date(), "->", dates.max().date())
    sep()
    return panel


def ensure_targets(panel: pd.DataFrame) -> pd.DataFrame:
    required = ["Forward_Excess_Return_252D", "Top_10pct_252D"]

    if any(column not in panel.columns for column in required):
        print("\nCalculating 252D targets in memory...")
        panel = add_252d_targets(panel)
    else:
        print("\nUsing existing 252D targets.")

    sep()
    return panel


def validate_features(panel: pd.DataFrame) -> list[str]:
    missing_all = [f for f in MODEL_FEATURES_252D if f not in panel.columns]
    if missing_all:
        raise ValueError(f"Missing ALL-model features:\n{missing_all}")

    fundamental_features = [
        f for f in MODEL_FEATURES_252D if f not in MARKET_FEATURES
    ]

    print("\nFEATURE SETS")
    print("MARKET_MACRO features:", len(MARKET_FEATURES))
    print("ALL features:", len(MODEL_FEATURES_252D))
    print("Fundamentals added:", len(fundamental_features))
    print()

    for feature in fundamental_features:
        coverage = panel[feature].notna().mean()
        print(f"{feature:<30} coverage {coverage:>7.2%}")

    sep()
    return fundamental_features


def select_period(
    panel: pd.DataFrame,
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    dates = pd.DatetimeIndex(panel.index.get_level_values("Date"))
    mask = pd.Series(True, index=panel.index)

    if start is not None:
        mask &= dates >= pd.Timestamp(start)
    if end is not None:
        mask &= dates <= pd.Timestamp(end)

    return panel.loc[mask.to_numpy()].copy()


def monthly_rank_evaluation(
    validation: pd.DataFrame,
    classifier,
    fold_number: int,
) -> tuple[pd.DataFrame, dict]:
    scored = validation.copy()
    scored["Predicted_Probability"] = classifier.predict_proba(
        scored[MODEL_FEATURES_252D]
    )[:, 1]

    scored = scored.reset_index()
    scored["Month"] = scored["Date"].dt.to_period("M")
    first_dates = scored.groupby("Month")["Date"].min()

    frames = []
    precision_values = []
    selected_returns = []
    universe_returns = []
    return_lifts = []
    universe_sizes = []

    for snapshot_date in first_dates:
        snapshot = scored[scored["Date"] == snapshot_date].copy()
        snapshot = snapshot.sort_values(
            "Predicted_Probability", ascending=False
        ).reset_index(drop=True)

        n_select = max(1, math.ceil(len(snapshot) * TOP_FRACTION))
        snapshot["Predicted_Top10"] = 0
        snapshot.loc[: n_select - 1, "Predicted_Top10"] = 1

        selected = snapshot[snapshot["Predicted_Top10"] == 1]

        precision = selected["Top_10pct_252D"].astype(float).mean()
        selected_return = selected["Forward_Excess_Return_252D"].mean()
        universe_return = snapshot["Forward_Excess_Return_252D"].mean()

        precision_values.append(precision)
        selected_returns.append(selected_return)
        universe_returns.append(universe_return)
        return_lifts.append(selected_return - universe_return)
        universe_sizes.append(len(snapshot))

        snapshot["Fold"] = fold_number
        snapshot["Snapshot_Date"] = snapshot_date
        snapshot["Actual_Top10"] = snapshot["Top_10pct_252D"].astype(int)
        snapshot["Actual_Excess_Return"] = snapshot["Forward_Excess_Return_252D"]

        keep = [
            "Fold",
            "Snapshot_Date",
            "Ticker",
            "Company",
            "Sector",
            "Sub_Industry",
            "Predicted_Probability",
            "Predicted_Top10",
            "Actual_Top10",
            "Actual_Excess_Return",
        ]
        frames.append(snapshot[[c for c in keep if c in snapshot.columns]])

    predictions = pd.concat(frames, ignore_index=True)

    metrics = {
        "Evaluation_Months": len(precision_values),
        "Average_Universe_Size": sum(universe_sizes) / len(universe_sizes),
        "Precision_at_10pct": sum(precision_values) / len(precision_values),
        "Mean_Selected_Excess_Return": sum(selected_returns) / len(selected_returns),
        "Mean_Universe_Excess_Return": sum(universe_returns) / len(universe_returns),
        "Return_Lift": sum(return_lifts) / len(return_lifts),
    }
    return predictions, metrics


def run_fold(panel: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, dict]:
    fold = config["fold"]

    train = select_period(panel, end=config["train_end"])
    validation = select_period(
        panel,
        start=config["validation_start"],
        end=config["validation_end"],
    )

    train = train[
        train["Forward_Excess_Return_252D"].notna()
        & train["Top_10pct_252D"].notna()
    ].copy()

    validation = validation[
        validation["Forward_Excess_Return_252D"].notna()
        & validation["Top_10pct_252D"].notna()
    ].copy()

    train_dates = pd.DatetimeIndex(train.index.get_level_values("Date"))
    val_dates = pd.DatetimeIndex(validation.index.get_level_values("Date"))

    print(f"\nFold {fold}")
    print("Train:", train_dates.min().date(), "->", train_dates.max().date())
    print("Validation:", val_dates.min().date(), "->", val_dates.max().date())
    print("Training rows:", f"{len(train):,}")
    print("Validation rows:", f"{len(validation):,}")
    print("Training ALL 33-feature model...")

    models = train_panel_models(
        X_train=train[MODEL_FEATURES_252D],
        y_top10_train=train["Top_10pct_252D"].astype(int),
        y_return_train=train["Forward_Excess_Return_252D"].astype(float),
    )

    predictions, metrics = monthly_rank_evaluation(
        validation,
        models.classifier,
        fold,
    )

    market_precision = MARKET_MACRO_2006_PRECISION[fold]
    metrics.update(
        {
            "Fold": fold,
            "Train_Start": train_dates.min().date(),
            "Train_End": train_dates.max().date(),
            "Validation_Start": val_dates.min().date(),
            "Validation_End": val_dates.max().date(),
            "Training_Rows": len(train),
            "Validation_Rows": len(validation),
            "Market_Macro_2006_Precision": market_precision,
        }
    )
    metrics["Precision_Change_vs_Market_Macro"] = (
        metrics["Precision_at_10pct"] - market_precision
    )

    print("ALL 33 Precision @ 10%:", f"{metrics['Precision_at_10pct']:.2%}")
    print("MARKET_MACRO 24 Precision @ 10%:", f"{market_precision:.2%}")
    print("Change:", f"{metrics['Precision_Change_vs_Market_Macro']:+.2%}")
    print("Average return lift:", f"{metrics['Return_Lift']:.2%}")
    sep()

    return predictions, metrics


def main() -> None:
    print("\nLONG-HISTORY ALL-FEATURE EXPERIMENT")
    print("24 MARKET_MACRO + 9 FUNDAMENTALS")
    print("Same 2006-history panel and same 8 validation folds")
    sep()

    panel = load_panel()
    panel = ensure_targets(panel)
    validate_features(panel)

    all_predictions = []
    summaries = []

    for config in FOLDS:
        predictions, metrics = run_fold(panel, config)
        all_predictions.append(predictions)
        summaries.append(metrics)

    summary = pd.DataFrame(summaries).sort_values("Fold").reset_index(drop=True)
    predictions = pd.concat(all_predictions, ignore_index=True)

    summary.to_csv(SUMMARY_PATH, index=False)
    predictions.to_parquet(PREDICTIONS_PATH, index=False)

    all_average = summary["Precision_at_10pct"].mean()
    market_average = summary["Market_Macro_2006_Precision"].mean()
    average_change = all_average - market_average

    improved = (summary["Precision_Change_vs_Market_Macro"] > 0).sum()
    worse = (summary["Precision_Change_vs_Market_Macro"] < 0).sum()
    unchanged = (summary["Precision_Change_vs_Market_Macro"] == 0).sum()

    print("\nFINAL COMPARISON")
    for row in summary.itertuples(index=False):
        print(
            f"Fold {row.Fold}: MARKET_MACRO {row.Market_Macro_2006_Precision:.2%} "
            f"-> ALL 33 {row.Precision_at_10pct:.2%} "
            f"({row.Precision_Change_vs_Market_Macro:+.2%})"
        )

    print()
    print("Average MARKET_MACRO Precision @ 10%:", f"{market_average:.2%}")
    print("Average ALL 33 Precision @ 10%:", f"{all_average:.2%}")
    print("Average change:", f"{average_change:+.2%}")
    print("Folds improved:", improved)
    print("Folds worse:", worse)
    print("Folds unchanged:", unchanged)

    print("\nFocus on old weak folds:")
    for fold in (3, 4):
        row = summary[summary["Fold"] == fold].iloc[0]
        print(
            f"Fold {fold}: MARKET_MACRO {row['Market_Macro_2006_Precision']:.2%} "
            f"-> ALL 33 {row['Precision_at_10pct']:.2%} "
            f"({row['Precision_Change_vs_Market_Macro']:+.2%})"
        )

    print("\nSaved summary:")
    print(SUMMARY_PATH)
    print("\nSaved monthly predictions:")
    print(PREDICTIONS_PATH)
    sep()


if __name__ == "__main__":
    main()
