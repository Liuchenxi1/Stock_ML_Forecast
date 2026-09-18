import numpy as np
import pandas as pd


TARGET_HORIZON = 252


TARGET_COLUMNS = [
    "Forward_Stock_Return_252D",
    "Forward_SPY_Return_252D",
    "Forward_Excess_Return_252D",
    "Cross_Sectional_Percentile_252D",
    "Top_10pct_252D",
]


def add_252d_targets(
    panel: pd.DataFrame,
    horizon: int = TARGET_HORIZON,
    top_percentile: float = 0.90,
    min_companies_for_rank: int = 50,
) -> pd.DataFrame:
    """
    Add approximately one-year forward-return targets.

    Targets
    -------
    Forward_Stock_Return_252D
        Company's future 252-trading-day return.

    Forward_SPY_Return_252D
        SPY future 252-trading-day return.

    Forward_Excess_Return_252D
        Stock future return minus SPY future return.

    Cross_Sectional_Percentile_252D
        Stock's future excess-return percentile relative
        to other companies on the same starting date.

    Top_10pct_252D
        1 if the company ends up in the top 10% of
        future excess returns for that starting date.

        0 otherwise.

        Missing when the future is not yet known.
    """

    df = panel.copy()

    # --------------------------------------------------------
    # Validate panel structure
    # --------------------------------------------------------

    if not isinstance(
        df.index,
        pd.MultiIndex,
    ):
        raise ValueError(
            "Panel must use a MultiIndex "
            "with Date and Ticker."
        )

    if (
        "Date" not in df.index.names
        or
        "Ticker" not in df.index.names
    ):
        raise ValueError(
            "Panel index must contain "
            "'Date' and 'Ticker'."
        )

    required_columns = [
        "Stock_Adj_Close",
        "Benchmark_Close",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing required columns: "
            f"{missing_columns}"
        )

    df = df.sort_index()

    # ========================================================
    # Stock forward return
    # ========================================================

    by_ticker = df.groupby(
        level="Ticker",
        sort=False,
    )

    future_stock_price = (
        by_ticker[
            "Stock_Adj_Close"
        ]
        .shift(-horizon)
    )

    df[
        "Forward_Stock_Return_252D"
    ] = (
        future_stock_price
        / df["Stock_Adj_Close"]
        - 1.0
    )

    # ========================================================
    # SPY forward return
    # ========================================================
    #
    # Benchmark_Close is duplicated across companies.
    #
    # Calculate the SPY return once per Date rather than
    # independently for every company.
    # ========================================================

    benchmark_by_date = (
        df[
            "Benchmark_Close"
        ]
        .groupby(
            level="Date"
        )
        .first()
        .sort_index()
    )

    future_benchmark = (
        benchmark_by_date
        .shift(-horizon)
    )

    benchmark_forward_return = (
        future_benchmark
        / benchmark_by_date
        - 1.0
    )

    dates = (
        df.index
        .get_level_values("Date")
    )

    df[
        "Forward_SPY_Return_252D"
    ] = dates.map(
        benchmark_forward_return
    ).to_numpy()

    # ========================================================
    # Excess return
    # ========================================================

    df[
        "Forward_Excess_Return_252D"
    ] = (
        df[
            "Forward_Stock_Return_252D"
        ]
        -
        df[
            "Forward_SPY_Return_252D"
        ]
    )

    # ========================================================
    # Cross-sectional percentile
    # ========================================================
    #
    # Compare companies that existed on the SAME starting
    # date.
    #
    # Example:
    #
    # AAPL       +30%
    # MSFT       +20%
    # XYZ        +50%
    #
    # ranking is based on forward excess return.
    # ========================================================

    percentile = (
        df
        .groupby(
            level="Date"
        )[
            "Forward_Excess_Return_252D"
        ]
        .rank(
            pct=True,
            method="average",
        )
    )

    # --------------------------------------------------------
    # Count companies with known future targets per date.
    #
    # Avoid producing a "top 10%" label on dates where only
    # a tiny number of companies have usable future returns.
    # --------------------------------------------------------

    valid_count = (
        df[
            "Forward_Excess_Return_252D"
        ]
        .notna()
        .groupby(
            level="Date"
        )
        .transform("sum")
    )

    valid_rank = (
        df[
            "Forward_Excess_Return_252D"
        ]
        .notna()
        &
        (
            valid_count
            >= min_companies_for_rank
        )
    )

    df[
        "Cross_Sectional_Percentile_252D"
    ] = percentile.where(
        valid_rank
    )

    # ========================================================
    # Top-decile classification target
    # ========================================================

    top_decile = pd.Series(
        pd.NA,
        index=df.index,
        dtype="Int64",
    )

    top_decile.loc[
        valid_rank
    ] = (
        df.loc[
            valid_rank,
            "Cross_Sectional_Percentile_252D",
        ]
        >= top_percentile
    ).astype(int)

    df[
        "Top_10pct_252D"
    ] = top_decile

    # --------------------------------------------------------
    # Clean impossible numeric values
    # --------------------------------------------------------

    numeric_target_columns = [
        "Forward_Stock_Return_252D",
        "Forward_SPY_Return_252D",
        "Forward_Excess_Return_252D",
        "Cross_Sectional_Percentile_252D",
    ]

    df[
        numeric_target_columns
    ] = (
        df[
            numeric_target_columns
        ]
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
    )

    return df


def print_target_summary(
    panel: pd.DataFrame,
) -> None:
    """
    Print target availability and class distribution.
    """

    print(
        "\n252D target summary:"
    )

    for column in TARGET_COLUMNS:

        if column not in panel.columns:
            continue

        coverage = (
            panel[column]
            .notna()
            .mean()
        )

        print(
            f"{column:<40} "
            f"{coverage:>7.1%}"
        )

    if (
        "Top_10pct_252D"
        in panel.columns
    ):

        known = (
            panel[
                "Top_10pct_252D"
            ]
            .dropna()
        )

        if not known.empty:

            positive_rate = (
                known
                .astype(int)
                .mean()
            )

            print(
                "\nTop-10% positive rate: "
                f"{positive_rate:.1%}"
            )

            print(
                "Known classification rows: "
                f"{len(known):,}"
            )

    # --------------------------------------------------------
    # Last date where targets are actually known
    # --------------------------------------------------------

    if (
        "Forward_Excess_Return_252D"
        in panel.columns
    ):

        known_target = panel[
            "Forward_Excess_Return_252D"
        ].dropna()

        if not known_target.empty:

            last_known_date = (
                known_target
                .index
                .get_level_values(
                    "Date"
                )
                .max()
            )

            print(
                "\nLatest date with known "
                "252D outcome:"
            )

            print(
                last_known_date
            )

    print("=" * 60)