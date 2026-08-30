import pandas as pd


DEFAULT_HORIZONS = (63, 126, 252)


def add_forward_return_targets(
    df: pd.DataFrame,
    price_column: str = "Stock_Adj_Close",
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
) -> pd.DataFrame:
    """
    Create forward-return targets for fixed trading-day horizons.

    Example:
        Forward_Return_63D =
            Price[t + 63] / Price[t] - 1
    """

    result = df.copy()

    for horizon in horizons:
        result[f"Forward_Return_{horizon}D"] = (
            result[price_column].shift(-horizon)
            / result[price_column]
            - 1
        )

    return result