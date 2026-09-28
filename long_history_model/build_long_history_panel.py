from __future__ import annotations

import time
from pathlib import Path

import pandas as pd


# ============================================================
# Import compatibility
# ============================================================
#
# Your project was reorganized into subfolders. These fallbacks
# support either the newer layout or the older flat layout.
# ============================================================

try:
    from stock_ml_forecast.data.universe import get_sp500_universe
except ImportError:
    from stock_ml_forecast.universe import get_sp500_universe

try:
    from stock_ml_forecast.data.panel import build_company_dataset
except ImportError:
    from stock_ml_forecast.panel import build_company_dataset

try:
    from stock_ml_forecast.macro.panel_features import (
        add_panel_features,
        MARKET_FEATURES,
    )
except ImportError:
    from stock_ml_forecast.panel_features import (
        add_panel_features,
        MARKET_FEATURES,
    )


# ============================================================
# Configuration
# ============================================================

START_DATE = "2006-01-01"
END_DATE = None

BENCHMARK_TICKER = "SPY"

# IMPORTANT:
# Replace this with the same SEC user-agent you use in main.py.
SEC_USER_AGENT = "Stock ML Forecast shinnkiryu@gmail.com"

# First long-history build should refresh market data back to 2006.
FORCE_REFRESH = True

# Set to 5 for a quick test.
# Set to None for the full current S&P 500 universe.
MAX_COMPANIES = None

# Save progress periodically so a long run is not lost.
CHECKPOINT_EVERY = 25

# If a partial checkpoint exists, reuse it and continue.
RESUME_FROM_CHECKPOINT = True


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROCESSED_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
)

PROCESSED_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

LONG_HISTORY_PANEL_PATH = (
    PROCESSED_DIR
    / "sp500_panel_2006_experiment.parquet"
)

CHECKPOINT_PATH = (
    PROCESSED_DIR
    / "sp500_panel_2006_experiment_partial.parquet"
)

FAILURES_PATH = (
    PROJECT_ROOT
    / "long_history_model"
    / "results"
    / "build_failures.csv"
)

FAILURES_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)


def print_separator() -> None:
    print("=" * 60)


def validate_configuration() -> None:
    if "your_email@example.com" in SEC_USER_AGENT:
        raise ValueError(
            "Please update SEC_USER_AGENT in "
            "build_long_history_panel.py to the same "
            "SEC user-agent you use in main.py."
        )




def normalize_company_frame(
    company: pd.DataFrame,
    ticker: str,
) -> pd.DataFrame:
    """Normalize every company frame to MultiIndex(Date, Ticker)."""

    df = company.copy()

    if (
        isinstance(df.index, pd.MultiIndex)
        and "Date" in df.index.names
        and "Ticker" in df.index.names
    ):
        return df.sort_index()

    if "Date" not in df.columns:
        index_name = df.index.name
        df = df.reset_index()

        if index_name == "Date":
            pass
        elif "index" in df.columns:
            df = df.rename(columns={"index": "Date"})
        elif index_name is not None and index_name in df.columns:
            df = df.rename(columns={index_name: "Date"})

    if "Ticker" not in df.columns:
        df["Ticker"] = ticker

    if "Date" not in df.columns:
        raise ValueError(
            f"{ticker}: could not find/create Date column. "
            f"Columns: {list(df.columns)}"
        )

    df["Date"] = pd.to_datetime(df["Date"])
    df["Ticker"] = df["Ticker"].fillna(ticker).astype(str)

    return (
        df
        .sort_values(["Date", "Ticker"])
        .drop_duplicates(subset=["Date", "Ticker"], keep="last")
        .set_index(["Date", "Ticker"])
        .sort_index()
    )


def combine_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Combine only canonical MultiIndex(Date, Ticker) frames."""

    if not frames:
        raise RuntimeError("No data frames are available to combine.")

    for i, frame in enumerate(frames, start=1):
        if not (
            isinstance(frame.index, pd.MultiIndex)
            and "Date" in frame.index.names
            and "Ticker" in frame.index.names
        ):
            raise ValueError(
                f"Frame {i} is not normalized to MultiIndex(Date, Ticker)."
            )

    panel = pd.concat(frames, axis=0)

    return (
        panel
        .reset_index()
        .sort_values(["Date", "Ticker"])
        .drop_duplicates(subset=["Date", "Ticker"], keep="last")
        .set_index(["Date", "Ticker"])
        .sort_index()
    )


def load_checkpoint() -> tuple[list[pd.DataFrame], set[str]]:
    """
    Load previously completed company rows, if available.
    """

    if (
        not RESUME_FROM_CHECKPOINT
        or not CHECKPOINT_PATH.exists()
    ):
        return [], set()

    print("\nLoading existing checkpoint:")
    print(CHECKPOINT_PATH)

    partial = pd.read_parquet(
        CHECKPOINT_PATH
    )

    if not (
        isinstance(partial.index, pd.MultiIndex)
        and "Date" in partial.index.names
        and "Ticker" in partial.index.names
    ):
        raise ValueError(
            "Existing checkpoint is not MultiIndex(Date, Ticker). "
            "Do not delete it; send me its structure if this appears."
        )

    partial = partial.sort_index()

    completed = set(
        partial.index
        .get_level_values("Ticker")
        .unique()
        .tolist()
    )

    print(
        "Completed tickers in checkpoint:",
        len(completed),
    )

    print_separator()

    return [partial], completed


def save_checkpoint(
    frames: list[pd.DataFrame],
) -> None:
    """
    Save raw company-level data before panel-level features.
    """

    if not frames:
        return

    partial = combine_frames(
        frames
    )

    partial.to_parquet(
        CHECKPOINT_PATH
    )

    print(
        "\nCheckpoint saved:"
    )

    print(
        CHECKPOINT_PATH
    )

    print(
        "Checkpoint companies:",
        partial.index
        .get_level_values("Ticker")
        .nunique(),
    )

    print_separator()


def build_long_history_panel() -> pd.DataFrame:
    validate_configuration()

    print("\nLONG-HISTORY PANEL BUILD")
    print("Start date:", START_DATE)
    print(
        "End date:",
        END_DATE or "latest available",
    )
    print("Benchmark:", BENCHMARK_TICKER)
    print("Force refresh:", FORCE_REFRESH)
    print(
        "Max companies:",
        MAX_COMPANIES
        if MAX_COMPANIES is not None
        else "ALL",
    )
    print(
        "Final output:",
        LONG_HISTORY_PANEL_PATH,
    )
    print_separator()

    universe = get_sp500_universe()

    if MAX_COMPANIES is not None:
        universe = (
            universe
            .head(MAX_COMPANIES)
            .copy()
        )

    print(
        "Universe securities:",
        len(universe),
    )

    print_separator()

    frames, completed = load_checkpoint()

    failures: list[dict[str, str]] = []

    total = len(universe)
    newly_completed = 0

    for number, row in enumerate(
        universe.itertuples(index=False),
        start=1,
    ):
        ticker = row.Ticker

        if ticker in completed:
            print(
                f"[{number}/{total}] "
                f"{ticker} - checkpoint, skip"
            )
            continue

        print(
            f"\n[{number}/{total}] {ticker}"
        )

        try:
            company = build_company_dataset(
                ticker=ticker,
                user_agent=SEC_USER_AGENT,
                benchmark_ticker=BENCHMARK_TICKER,
                start_date=START_DATE,
                end_date=END_DATE,
                force_refresh=FORCE_REFRESH,
            )

            if company.empty:
                raise ValueError(
                    "Company dataset is empty."
                )

            company["Company"] = row.Company
            company["Sector"] = row.Sector
            company["Sub_Industry"] = row.Sub_Industry

            # build_company_dataset normally already adds Ticker.
            # This keeps the file robust if that implementation changes.
            if "Ticker" not in company.columns:
                company["Ticker"] = ticker

            company = normalize_company_frame(
                company=company,
                ticker=ticker,
            )

            frames.append(
                company
            )

            completed.add(
                ticker
            )

            newly_completed += 1

            company_dates = pd.DatetimeIndex(
                company.index.get_level_values("Date")
            )

            print(
                "Rows:",
                f"{len(company):,}",
            )

            print(
                "Dates:",
                company_dates.min().date(),
                "->",
                company_dates.max().date(),
            )

            if (
                newly_completed
                % CHECKPOINT_EVERY
                == 0
            ):
                save_checkpoint(
                    frames
                )

                # Reload checkpoint as a single frame so
                # memory does not keep many duplicate objects.
                checkpoint = pd.read_parquet(
                    CHECKPOINT_PATH
                )

                frames = [
                    checkpoint
                ]

        except Exception as exc:
            message = str(exc)

            print(
                f"{ticker}: FAILED - "
                f"{message}"
            )

            failures.append(
                {
                    "Ticker": ticker,
                    "Error": message,
                }
            )

        # Polite pause for SEC-related requests.
        time.sleep(0.12)

    if failures:
        pd.DataFrame(
            failures
        ).to_csv(
            FAILURES_PATH,
            index=False,
        )

        print(
            "\nFailure log saved:"
        )

        print(
            FAILURES_PATH
        )

    if not frames:
        raise RuntimeError(
            "No company datasets were built "
            "and no checkpoint was available."
        )

    # ========================================================
    # Final raw panel
    # ========================================================

    print(
        "\nCombining company datasets..."
    )

    panel = combine_frames(
        frames
    )

    print(
        "Raw combined rows:",
        f"{len(panel):,}",
    )

    print(
        "Raw combined companies:",
        panel.index
        .get_level_values("Ticker")
        .nunique(),
    )

    print_separator()

    # ========================================================
    # Panel-level features
    # ========================================================

    print(
        "\nAdding panel-level MARKET + MACRO features..."
    )

    panel = add_panel_features(
        panel
    )

    # ========================================================
    # Validation
    # ========================================================

    dates = pd.DatetimeIndex(
        panel.index
        .get_level_values("Date")
    )

    companies = (
        panel.index
        .get_level_values("Ticker")
        .nunique()
    )

    missing_market_features = [
        feature
        for feature in MARKET_FEATURES
        if feature not in panel.columns
    ]

    print(
        "\nLONG-HISTORY PANEL SUMMARY"
    )

    print(
        "Rows:",
        f"{len(panel):,}",
    )

    print(
        "Companies:",
        companies,
    )

    print(
        "Date range:",
        dates.min().date(),
        "->",
        dates.max().date(),
    )

    print(
        "MARKET_MACRO features:",
        len(MARKET_FEATURES),
    )

    if dates.min() > pd.Timestamp(
        "2007-01-01"
    ):
        print(
            "\nWARNING:"
        )

        print(
            "The earliest panel date is later than expected."
        )

        print(
            "The raw market cache may still be using "
            "shorter historical data."
        )

    if missing_market_features:
        print(
            "\nMissing MARKET_MACRO features:"
        )

        for feature in missing_market_features:
            print(
                f"  - {feature}"
            )

        raise ValueError(
            "Cannot save final long-history panel "
            "because required MARKET_MACRO features "
            "are missing."
        )

    print(
        "All MARKET_MACRO features are present."
    )

    print(
        "Build failures:",
        len(failures),
    )

    print_separator()

    # ========================================================
    # Save final panel
    # ========================================================

    panel.to_parquet(
        LONG_HISTORY_PANEL_PATH
    )

    print(
        "\nSaved final long-history panel:"
    )

    print(
        LONG_HISTORY_PANEL_PATH
    )

    print_separator()

    # Remove partial checkpoint only after successful final save.
    if CHECKPOINT_PATH.exists():
        CHECKPOINT_PATH.unlink()

        print(
            "Removed completed checkpoint:"
        )

        print(
            CHECKPOINT_PATH
        )

        print_separator()

    return panel


def main() -> None:
    build_long_history_panel()


if __name__ == "__main__":
    main()
