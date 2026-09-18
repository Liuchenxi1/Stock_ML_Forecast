import pandas as pd


SP500_URL = (
    "https://en.wikipedia.org/wiki/"
    "List_of_S%26P_500_companies"
)


def get_sp500_universe() -> pd.DataFrame:
    """
    Download the current S&P 500 constituent list.
    """

    headers = {
        "User-Agent": (
            "Mozilla/5.0 "
            "(Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 "
            "(KHTML, like Gecko) "
            "Chrome/152.0 Safari/537.36"
        )
    }

    tables = pd.read_html(
        SP500_URL,
        storage_options=headers,
    )

    df = tables[0].copy()

    df = df.rename(
        columns={
            "Symbol": "Ticker",
            "Security": "Company",
            "GICS Sector": "Sector",
            "GICS Sub-Industry": "Sub_Industry",
        }
    )

    df["Ticker"] = (
        df["Ticker"]
        .astype(str)
        .str.replace(
            ".",
            "-",
            regex=False,
        )
    )

    columns_to_keep = [
        "Ticker",
        "Company",
        "Sector",
        "Sub_Industry",
        "CIK",
        "Founded",
    ]

    return df[
        columns_to_keep
    ].copy()