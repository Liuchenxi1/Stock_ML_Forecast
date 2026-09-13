import time

import pandas as pd

from stock_ml_forecast.cache import (
    load_or_download_market_data,
    load_or_download_sec_facts,
)

from stock_ml_forecast.features import (
    add_technical_features,
)

from stock_ml_forecast.sec_data import (
    add_sec_fundamentals,
)

from stock_ml_forecast.paths import (
    SP500_PANEL_PATH,
    ensure_data_directories,
)


def build_company_dataset(
    ticker: str,
    user_agent: str,
    benchmark_ticker: str = "SPY",
    start_date: str | None = None,
    end_date: str | None = None,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Build one company's complete point-in-time dataset.
    """

    ticker = ticker.upper()

    # ------------------------------------------
    # Market
    # ------------------------------------------

    df = load_or_download_market_data(
        ticker=ticker,
        benchmark_ticker=benchmark_ticker,
        start_date=start_date,
        end_date=end_date,
        force_refresh=force_refresh,
    )

    # ------------------------------------------
    # Technical features
    # ------------------------------------------

    df = add_technical_features(
        df
    )

    # ------------------------------------------
    # SEC facts
    # ------------------------------------------

    _, facts = (
        load_or_download_sec_facts(
            ticker=ticker,
            user_agent=user_agent,
            force_refresh=force_refresh,
        )
    )

    df = add_sec_fundamentals(
        df=df,
        ticker=ticker,
        ser_agent=user_agent,
        facts=facts,
    )

    # ------------------------------------------
    # Company identity
    #
    # Keep this for tracking.
    # Do NOT necessarily feed Ticker into model.
    # ------------------------------------------

    df["Ticker"] = ticker

    return df


def build_sp500_panel(
    universe: pd.DataFrame,
    user_agent: str,
    benchmark_ticker: str = "SPY",
    start_date: str | None = None,
    end_date: str | None = None,
    force_refresh: bool = False,
    max_companies: int | None = None,
) -> pd.DataFrame:
    """
    Build a combined multi-company S&P 500 panel.
    """

    ensure_data_directories()

    companies = universe.copy()

    if max_companies is not None:

        companies = companies.head(
            max_companies
        )

    frames = []

    failures = []

    total = len(
        companies
    )

    for number, row in enumerate(
        companies.itertuples(
            index=False
        ),
        start=1,
    ):

        ticker = row.Ticker

        print(
            f"\n[{number}/{total}] "
            f"{ticker}"
        )

        try:

            company_df = (
                build_company_dataset(
                    ticker=ticker,
                    user_agent=user_agent,
                    benchmark_ticker=benchmark_ticker,
                    start_date=start_date,
                    end_date=end_date,
                    force_refresh=force_refresh,
                )
            )

            # Add metadata from universe
            company_df[
                "Company"
            ] = row.Company

            company_df[
                "Sector"
            ] = row.Sector

            company_df[
                "Sub_Industry"
            ] = row.Sub_Industry

            frames.append(
                company_df
            )

        except Exception as exc:

            print(
                f"{ticker}: FAILED - "
                f"{exc}"
            )

            failures.append(
                {
                    "Ticker": ticker,
                    "Error": str(exc),
                }
            )

        # Be polite to SEC.
        time.sleep(
            0.12
        )

    if not frames:

        raise RuntimeError(
            "No company datasets were built."
        )

    panel = pd.concat(
        frames,
        axis=0,
    )

    panel = panel.reset_index()

    panel = panel.sort_values(
        [
            "Date",
            "Ticker",
        ]
    )

    panel = panel.set_index(
        [
            "Date",
            "Ticker",
        ]
    )

    panel.to_parquet(
        SP500_PANEL_PATH
    )

    print(
        f"\nSaved panel:"
        f"\n{SP500_PANEL_PATH}"
    )

    print(
        f"Rows: {len(panel):,}"
    )

    print(
        "Companies: "
        f"{panel.index.get_level_values('Ticker').nunique():,}"
    )

    if failures:

        print(
            f"Failures: "
            f"{len(failures)}"
        )

    print("=" * 60)

    return panel


def load_sp500_panel() -> pd.DataFrame:
    """
    Load the already-built processed panel.
    """

    if not SP500_PANEL_PATH.exists():

        raise FileNotFoundError(
            "S&P 500 panel does not exist. "
            "Build it first."
        )

    return pd.read_parquet(
        SP500_PANEL_PATH
    )