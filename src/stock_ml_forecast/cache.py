import json

import pandas as pd

from datetime import datetime, timedelta, timezone
from pathlib import Path


from stock_ml_forecast.market_data import (
    download_market_data,
)

from stock_ml_forecast.sec_data import (
    lookup_sec_company,
    download_company_facts,
)

from stock_ml_forecast.paths import (
    MARKET_DATA_DIR,
    SEC_DATA_DIR,
    ensure_data_directories,
)


def market_cache_path(
    ticker: str,
):
    return (
        MARKET_DATA_DIR
        / f"{ticker.upper()}.parquet"
    )


def sec_cache_path(
    ticker: str,
):
    return (
        SEC_DATA_DIR
        / f"{ticker.upper()}.json"
    )


def load_or_download_market_data(
    ticker: str,
    benchmark_ticker: str = "SPY",
    start_date: str | None = None,
    end_date: str | None = None,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Read market data from the local cache.

    If the cache does not exist, download from Yahoo
    and save it as Parquet.
    """

    ensure_data_directories()

    path = market_cache_path(
        ticker
    )

    if (
            not force_refresh
            and is_cache_fresh(
        path,
        max_age_hours=24,
        )
    ):
        print(
            f"{ticker}: market cache"
        )

        return pd.read_parquet(
            path
        )

    print(
        f"{ticker}: downloading market data"
    )

    df = download_market_data(
        ticker=ticker,
        benchmark_ticker=benchmark_ticker,
        start_date=start_date,
        end_date=end_date,
    )

    df.to_parquet(
        path
    )

    return df


def load_or_download_sec_facts(
    ticker: str,
    user_agent: str,
    force_refresh: bool = False,
) -> tuple[dict, dict]:
    """
    Return:

        company_info
        company_facts

    Read Company Facts from the local JSON cache when
    possible.

    Otherwise download once from SEC and save locally.
    """

    ensure_data_directories()

    ticker = ticker.upper()

    path = sec_cache_path(
        ticker
    )

    company = lookup_sec_company(
        ticker=ticker,
        user_agent=user_agent,
    )

    if company is None:

        raise ValueError(
            f"SEC company not found: {ticker}"
        )

    if (
        path.exists()
        and not force_refresh
    ):

        print(
            f"{ticker}: SEC cache"
        )

        with open(
            path,
            "r",
            encoding="utf-8",
        ) as file:

            facts = json.load(
                file
            )

        return (
            company,
            facts,
        )

    print(
        f"{ticker}: downloading SEC facts"
    )

    facts = download_company_facts(
        cik=company["cik"],
        user_agent=user_agent,
    )

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            facts,
            file,
        )

    return (
        company,
        facts,
    )

def is_cache_fresh(
    path: Path,
    max_age_hours: int,
) -> bool:

    if not path.exists():
        return False

    modified_time = datetime.fromtimestamp(
        path.stat().st_mtime,
        tz=timezone.utc,
    )

    age = (
        datetime.now(timezone.utc)
        - modified_time
    )

    return age < timedelta(
        hours=max_age_hours
    )