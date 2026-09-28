from __future__ import annotations

import json
import pickle
from pathlib import Path

import pandas as pd

from stock_ml_forecast.macro.panel_features import MARKET_FEATURES
from stock_ml_forecast.modeling.panel_models import train_panel_models
from stock_ml_forecast.modeling.panel_targets import add_252d_targets


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

LONG_HISTORY_PANEL_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "sp500_panel_2006_experiment.parquet"
)

MODEL_DIR = (
    PROJECT_ROOT
    / "long_history_model"
    / "results"
    / "models"
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

CLASSIFIER_PATH = (
    MODEL_DIR
    / "market_macro_classifier_2006_2026.pkl"
)

REGRESSOR_PATH = (
    MODEL_DIR
    / "market_macro_regressor_2006_2026.pkl"
)

METADATA_PATH = (
    MODEL_DIR
    / "training_metadata.json"
)


def print_separator() -> None:
    print("=" * 60)


def load_long_history_panel() -> pd.DataFrame:
    if not LONG_HISTORY_PANEL_PATH.exists():
        raise FileNotFoundError(
            "Long-history panel was not found:\n"
            f"{LONG_HISTORY_PANEL_PATH}\n\n"
            "Run this first:\n"
            "uv run python long_history_model/build_long_history_panel.py"
        )

    print("\nLoading long-history panel:")
    print(LONG_HISTORY_PANEL_PATH)

    panel = pd.read_parquet(
        LONG_HISTORY_PANEL_PATH
    )

    dates = pd.DatetimeIndex(
        panel.index.get_level_values("Date")
    )

    print(
        "Rows:",
        f"{len(panel):,}",
    )

    print(
        "Companies:",
        panel.index
        .get_level_values("Ticker")
        .nunique(),
    )

    print(
        "Date range:",
        dates.min().date(),
        "->",
        dates.max().date(),
    )

    print_separator()

    return panel


def ensure_targets(
    panel: pd.DataFrame,
) -> pd.DataFrame:
    required = [
        "Forward_Excess_Return_252D",
        "Top_10pct_252D",
    ]

    missing = [
        column
        for column in required
        if column not in panel.columns
    ]

    if missing:
        print(
            "\n252D targets are not saved in the long-history panel."
        )

        print(
            "Calculating targets in memory..."
        )

        panel = add_252d_targets(
            panel
        )

    else:
        print(
            "\nUsing existing 252D targets."
        )

    print_separator()

    return panel


def prepare_training_data(
    panel: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.Series,
    pd.Series,
    pd.DataFrame,
]:
    missing_features = [
        feature
        for feature in MARKET_FEATURES
        if feature not in panel.columns
    ]

    if missing_features:
        raise ValueError(
            "Long-history panel is missing MARKET_MACRO features:\n"
            f"{missing_features}"
        )

    known = panel[
        panel[
            "Forward_Excess_Return_252D"
        ].notna()
        &
        panel[
            "Top_10pct_252D"
        ].notna()
    ].copy()

    X_train = known[
        MARKET_FEATURES
    ]

    y_top10 = (
        known[
            "Top_10pct_252D"
        ]
        .astype(int)
    )

    y_return = (
        known[
            "Forward_Excess_Return_252D"
        ]
        .astype(float)
    )

    dates = pd.DatetimeIndex(
        known.index.get_level_values("Date")
    )

    print(
        "\nLONG-HISTORY TRAINING DATA"
    )

    print(
        "Rows:",
        f"{len(known):,}",
    )

    print(
        "Companies:",
        known.index
        .get_level_values("Ticker")
        .nunique(),
    )

    print(
        "Training dates:",
        dates.min().date(),
        "->",
        dates.max().date(),
    )

    print(
        "Features:",
        len(MARKET_FEATURES),
    )

    print(
        "Top-10% positive rate:",
        f"{y_top10.mean():.2%}",
    )

    print(
        "Mean future excess return:",
        f"{y_return.mean():.2%}",
    )

    print_separator()

    return (
        X_train,
        y_top10,
        y_return,
        known,
    )


def train_long_history_model(
    X_train: pd.DataFrame,
    y_top10: pd.Series,
    y_return: pd.Series,
):
    print(
        "\nTraining 2006-2026 MARKET_MACRO model..."
    )

    models = train_panel_models(
        X_train=X_train,
        y_top10_train=y_top10,
        y_return_train=y_return,
    )

    print_separator()

    return models


def save_models(
    models,
    known: pd.DataFrame,
) -> None:
    with open(
        CLASSIFIER_PATH,
        "wb",
    ) as file:
        pickle.dump(
            models.classifier,
            file,
        )

    with open(
        REGRESSOR_PATH,
        "wb",
    ) as file:
        pickle.dump(
            models.regressor,
            file,
        )

    dates = pd.DatetimeIndex(
        known.index.get_level_values("Date")
    )

    metadata = {
        "panel_path":
            str(
                LONG_HISTORY_PANEL_PATH
            ),

        "training_start":
            str(
                dates.min().date()
            ),

        "training_end":
            str(
                dates.max().date()
            ),

        "training_rows":
            int(
                len(known)
            ),

        "companies":
            int(
                known.index
                .get_level_values("Ticker")
                .nunique()
            ),

        "feature_count":
            len(
                MARKET_FEATURES
            ),

        "features":
            list(
                MARKET_FEATURES
            ),

        "target":
            "Top_10pct_252D",

        "return_target":
            "Forward_Excess_Return_252D",
    }

    with open(
        METADATA_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            indent=2,
        )

    print(
        "\nSaved classifier:"
    )

    print(
        CLASSIFIER_PATH
    )

    print(
        "\nSaved regressor:"
    )

    print(
        REGRESSOR_PATH
    )

    print(
        "\nSaved training metadata:"
    )

    print(
        METADATA_PATH
    )

    print_separator()


def main() -> None:
    print(
        "\nLONG-HISTORY MODEL TRAINING"
    )

    print(
        "Model: MARKET_MACRO"
    )

    print(
        "Historical panel: 2006 -> latest"
    )

    print_separator()

    panel = load_long_history_panel()

    panel = ensure_targets(
        panel
    )

    (
        X_train,
        y_top10,
        y_return,
        known,
    ) = prepare_training_data(
        panel
    )

    models = train_long_history_model(
        X_train=X_train,
        y_top10=y_top10,
        y_return=y_return,
    )

    save_models(
        models=models,
        known=known,
    )

    print(
        "\nTraining complete."
    )

    print(
        "Next step: run walk_forward_long_history.py"
    )

    print_separator()


if __name__ == "__main__":
    main()
