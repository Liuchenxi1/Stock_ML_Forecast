import pandas as pd


SP500_URL = (
    "https://en.wikipedia.org/wiki/"
    "List_of_S%26P_500_companies"
)


def get_sp500_universe() -> pd.DataFrame:
    """
    Download the current S&P 500 constituent list.

    Returns:
        Symbol
        Security
        GICS_Sector
        GICS_Sub_Industry
    """

    tables = pd.read_html(
        SP500_URL
    )

    df = tables[0].copy()

    df = df.rename(
        columns={
            "Symbol": "Ticker",
            "Security": "Company",
            "GICS Sector": "Sector",
            "GICS Sub-Industry":
                "Sub_Industry",
        }
    )

    # Yahoo uses BRK-B instead of BRK.B, etc.
    df["Ticker"] = (
        df["Ticker"]
        .str.replace(
            ".",
            "-",
            regex=False,
        )
    )

    return df[
        [
            "Ticker",
            "Company",
            "Sector",
            "Sub_Industry",
        ]
    ]