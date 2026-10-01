from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

from stock_ml_forecast.macro.panel_features import MARKET_FEATURES
from stock_ml_forecast.modeling.panel_models import train_panel_models
from stock_ml_forecast.modeling.panel_targets import add_252d_targets

TOP_FRACTION = 0.10
EXTRA_FEATURE = "FCF_Growth_YoY"
EXPERIMENT_FEATURES = list(MARKET_FEATURES) + [EXTRA_FEATURE]

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LONG_HISTORY_PANEL_PATH = (
    PROJECT_ROOT / "data" / "processed" / "sp500_panel_2006_experiment.parquet"
)
RESULTS_DIR = (
    PROJECT_ROOT / "long_history_model" / "results" / "market_macro_fcf_growth"
)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
SUMMARY_PATH = RESULTS_DIR / "market_macro_fcf_growth_walk_forward.csv"
PREDICTIONS_PATH = RESULTS_DIR / "market_macro_fcf_growth_predictions.parquet"

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


def validate_features(panel: pd.DataFrame) -> None:
    missing = [feature for feature in EXPERIMENT_FEATURES if feature not in panel.columns]
    if missing:
        raise ValueError(f"Missing experiment features:\n{missing}")

    coverage = panel[EXTRA_FEATURE].notna().mean()
    print("\nFEATURE SET")
    print("MARKET_MACRO:", len(MARKET_FEATURES))
    print("Extra feature:", EXTRA_FEATURE)
    print("Total:", len(EXPERIMENT_FEATURES))
    print(f"{EXTRA_FEATURE:<25} coverage {coverage:>7.2%}")
    sep()


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


def evaluate_monthly(
    validation: pd.DataFrame,
    classifier,
    fold_number: int,
) -> tuple[pd.DataFrame, dict]:
    scored = validation.copy()
    scored["Predicted_Probability"] = classifier.predict_proba(
        scored[EXPERIMENT_FEATURES]
    )[:, 1]

    scored = scored.reset_index()
    scored["Month"] = scored["Date"].dt.to_period("M")
    first_dates = scored.groupby("Month")["Date"].min()

    prediction_frames = []
    precisions = []
    return_lifts = []
    selected_returns = []
    universe_returns = []
    universe_sizes = []

    for snapshot_date in first_dates:
        snapshot = scored[scored["Date"] == snapshot_date].copy()
        if snapshot.empty:
            continue

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

        precisions.append(precision)
        selected_returns.append(selected_return)
        universe_returns.append(universe_return)
        return_lifts.append(selected_return - universe_return)
        universe_sizes.append(len(snapshot))

        snapshot["Fold"] = fold_number
        snapshot["Snapshot_Date"] = snapshot_date
        snapshot["Actual_Top10"] = snapshot["Top_10pct_252D"].astype(int)
        snapshot["Actual_Excess_Return"] = snapshot["Forward_Excess_Return_252D"]

        keep_columns = [
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
        available_columns = [c for c in keep_columns if c in snapshot.columns]
        prediction_frames.append(snapshot[available_columns])

    if not prediction_frames:
        raise RuntimeError(f"Fold {fold_number} produced no snapshots.")

    predictions = pd.concat(prediction_frames, ignore_index=True)
    metrics = {
        "Evaluation_Months": len(precisions),
        "Average_Universe_Size": sum(universe_sizes) / len(universe_sizes),
        "Precision_at_10pct": sum(precisions) / len(precisions),
        "Mean_Selected_Excess_Return": sum(selected_returns) / len(selected_returns),
        "Mean_Universe_Excess_Return": sum(universe_returns) / len(universe_returns),
        "Return_Lift": sum(return_lifts) / len(return_lifts),
    }
    return predictions, metrics


def run_fold(panel: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, dict]:
    fold_number = config["fold"]

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
    validation_dates = pd.DatetimeIndex(validation.index.get_level_values("Date"))

    print(f"\nFold {fold_number}")
    print("Train:", train_dates.min().date(), "->", train_dates.max().date())
    print(
        "Validation:",
        validation_dates.min().date(),
        "->",
        validation_dates.max().date(),
    )
    print("Training rows:", f"{len(train):,}")
    print("Validation rows:", f"{len(validation):,}")
    print("Training MARKET_MACRO + FCF_Growth_YoY model...")

    models = train_panel_models(
        X_train=train[EXPERIMENT_FEATURES],
        y_top10_train=train["Top_10pct_252D"].astype(int),
        y_return_train=train["Forward_Excess_Return_252D"].astype(float),
    )

    predictions, metrics = evaluate_monthly(
        validation=validation,
        classifier=models.classifier,
        fold_number=fold_number,
    )

    baseline = MARKET_MACRO_2006_PRECISION[fold_number]
    metrics.update(
        {
            "Fold": fold_number,
            "Train_Start": train_dates.min().date(),
            "Train_End": train_dates.max().date(),
            "Validation_Start": validation_dates.min().date(),
            "Validation_End": validation_dates.max().date(),
            "Training_Rows": len(train),
            "Validation_Rows": len(validation),
            "Market_Macro_2006_Precision": baseline,
        }
    )
    metrics["Precision_Change_vs_Market_Macro"] = (
        metrics["Precision_at_10pct"] - baseline
    )

    print(
        "MARKET_MACRO + FCF_Growth_YoY Precision @ 10%:",
        f"{metrics['Precision_at_10pct']:.2%}",
    )
    print("MARKET_MACRO baseline Precision @ 10%:", f"{baseline:.2%}")
    print("Change:", f"{metrics['Precision_Change_vs_Market_Macro']:+.2%}")
    print(
        "Average selected excess return:",
        f"{metrics['Mean_Selected_Excess_Return']:.2%}",
    )
    print(
        "Average universe excess return:",
        f"{metrics['Mean_Universe_Excess_Return']:.2%}",
    )
    print("Average return lift:", f"{metrics['Return_Lift']:.2%}")
    sep()

    return predictions, metrics


def main() -> None:
    print("\nMARKET_MACRO + FCF_GROWTH_YOY EXPERIMENT")
    print("24 MARKET_MACRO + FCF_Growth_YoY")
    print("Same 2006-history panel and same 8 validation folds")
    sep()

    panel = load_panel()
    panel = ensure_targets(panel)
    validate_features(panel)

    all_predictions = []
    summaries = []

    for config in FOLDS:
        predictions, metrics = run_fold(panel=panel, config=config)
        all_predictions.append(predictions)
        summaries.append(metrics)

    summary = pd.DataFrame(summaries).sort_values("Fold").reset_index(drop=True)
    predictions = pd.concat(all_predictions, ignore_index=True)

    summary.to_csv(SUMMARY_PATH, index=False)
    predictions.to_parquet(PREDICTIONS_PATH, index=False)

    experiment_average = summary["Precision_at_10pct"].mean()
    baseline_average = summary["Market_Macro_2006_Precision"].mean()

    improved_folds = (summary["Precision_Change_vs_Market_Macro"] > 0).sum()
    worse_folds = (summary["Precision_Change_vs_Market_Macro"] < 0).sum()
    unchanged_folds = (summary["Precision_Change_vs_Market_Macro"] == 0).sum()

    print("\nFINAL COMPARISON")
    for row in summary.itertuples(index=False):
        print(
            f"Fold {row.Fold}: "
            f"MARKET_MACRO {row.Market_Macro_2006_Precision:.2%} "
            f"-> + FCF_Growth_YoY {row.Precision_at_10pct:.2%} "
            f"({row.Precision_Change_vs_Market_Macro:+.2%})"
        )

    print()
    print(
        "Average MARKET_MACRO Precision @ 10%:",
        f"{baseline_average:.2%}",
    )
    print(
        "Average + FCF_Growth_YoY Precision @ 10%:",
        f"{experiment_average:.2%}",
    )
    print("Average change:", f"{experiment_average - baseline_average:+.2%}")
    print("Folds improved:", improved_folds)
    print("Folds worse:", worse_folds)
    print("Folds unchanged:", unchanged_folds)

    print("\nFocus on Fold 3 / Fold 4:")
    for fold_number in [3, 4]:
        row = summary[summary["Fold"] == fold_number].iloc[0]
        print(
            f"Fold {fold_number}: "
            f"{row['Market_Macro_2006_Precision']:.2%} "
            f"-> {row['Precision_at_10pct']:.2%} "
            f"({row['Precision_Change_vs_Market_Macro']:+.2%})"
        )

    print("\nSaved summary:")
    print(SUMMARY_PATH)
    print("\nSaved monthly predictions:")
    print(PREDICTIONS_PATH)
    sep()


if __name__ == "__main__":
    main()
