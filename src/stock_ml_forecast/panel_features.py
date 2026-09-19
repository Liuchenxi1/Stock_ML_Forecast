import numpy as np
import pandas as pd


MODEL_FEATURES_252D = [
    # Existing market features
    "Stock_Return_1D",
    "Benchmark_Return_1D",
    "Price_to_MA20",
    "Price_to_MA50",
    "Price_to_MA200",
    "RSI_14",
    "MACD_Pct",
    "Volatility_20D",
    "Yield_Curve_10Y_3M",

    # New regime features
    "SPY_Return_20D",
    "SPY_Return_63D",
    "SPY_Volatility_20D",

    "VIX_Change_20D",
    "VIX_Change_63D",

    "Treasury_3M_Change_20D",
    "Treasury_3M_Change_63D",

    "Treasury_10Y_Change_20D",
    "Treasury_10Y_Change_63D",

    "Yield_Curve_Change_20D",
    "Yield_Curve_Change_63D",

    # Fundamentals
    "Revenue_Growth_YoY",
    "Net_Income_Growth_YoY",
    "FCF_Growth_YoY",
    "Debt_to_Assets",
    "Debt_to_Equity",
    "Cash_to_Debt",
    "ROA",
    "ROE",
    "FCF_Margin",

    # Market confirmation
    "Volume_Ratio_20D",
    "Return_20D",
    "Return_63D",
    "Excess_Return_vs_SPY_63D",
]


def _safe_divide(
    numerator: pd.Series,
    denominator: pd.Series,
) -> pd.Series:
    """
    Divide while avoiding divide-by-zero and infinity.
    """

    denominator = denominator.replace(
        0,
        np.nan,
    )

    result = (
        numerator
        / denominator
    )

    return result.replace(
        [np.inf, -np.inf],
        np.nan,
    )


def _pct_growth(
    current: pd.Series,
    previous: pd.Series,
) -> pd.Series:
    """
    Growth rate:

        current / previous - 1

    Only calculate where the previous value is usable.
    """

    previous = previous.replace(
        0,
        np.nan,
    )

    result = (
        current
        / previous
        - 1
    )

    return result.replace(
        [np.inf, -np.inf],
        np.nan,
    )


def add_panel_features(
    panel: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add normalized cross-company features to the
    S&P 500 panel.

    Expected index:

        Date
        Ticker
    """

    df = panel.copy()

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

    df = df.sort_index()

    # ========================================================
    # Group by company
    # ========================================================

    by_ticker = df.groupby(
        level="Ticker",
        sort=False,
    )

    # ========================================================
    # Market / technical normalization
    # ========================================================

    df["MACD_Pct"] = _safe_divide(
        df["MACD"],
        df["Stock_Adj_Close"],
    )

    # --------------------------------------------------------
    # 20-day return
    # --------------------------------------------------------

    df["Return_20D"] = (
        by_ticker["Stock_Adj_Close"]
        .pct_change(
            periods=20,
            fill_method=None,
        )
    )

    # --------------------------------------------------------
    # 63-day return
    # --------------------------------------------------------

    df["Return_63D"] = (
        by_ticker["Stock_Adj_Close"]
        .pct_change(
            periods=63,
            fill_method=None,
        )
    )

    # --------------------------------------------------------
    # SPY 63-day return
    #
    # Benchmark_Close is repeated for each company,
    # but must still be calculated independently inside
    # each ticker group to preserve alignment.
    # --------------------------------------------------------

    df["SPY_Return_63D"] = (
        by_ticker["Benchmark_Close"]
        .pct_change(
            periods=63,
            fill_method=None,
        )
    )

    df[
        "Excess_Return_vs_SPY_63D"
    ] = (
        df["Return_63D"]
        - df["SPY_Return_63D"]
    )

    # --------------------------------------------------------
    # Relative volume
    # --------------------------------------------------------

    volume_mean_20 = (
        by_ticker["Stock_Volume"]
        .transform(
            lambda series:
                series
                .rolling(
                    window=20,
                    min_periods=20,
                )
                .mean()
        )
    )

    df["Volume_Ratio_20D"] = (
        _safe_divide(
            df["Stock_Volume"],
            volume_mean_20,
        )
    )

    # ========================================================
    # Free cash flow
    # ========================================================

    df["Free_Cash_Flow"] = (
        df["Operating_Cash_Flow"]
        - df["Capital_Expenditures"]
    )

    # ========================================================
    # Balance-sheet ratios
    # ========================================================

    df["Debt_to_Assets"] = (
        _safe_divide(
            df["Total_Debt"],
            df["Total_Assets"],
        )
    )

    df["Debt_to_Equity"] = (
        _safe_divide(
            df["Total_Debt"],
            df["Stockholders_Equity"],
        )
    )

    df["Cash_to_Debt"] = (
        _safe_divide(
            df["Cash"],
            df["Total_Debt"],
        )
    )

    # ========================================================
    # Profitability
    # ========================================================

    df["ROA"] = (
        _safe_divide(
            df["Net_Income"],
            df["Total_Assets"],
        )
    )

    df["ROE"] = (
        _safe_divide(
            df["Net_Income"],
            df["Stockholders_Equity"],
        )
    )

    df["FCF_Margin"] = (
        _safe_divide(
            df["Free_Cash_Flow"],
            df["Revenue"],
        )
    )

    # ========================================================
    # Approximate YoY fundamental growth
    #
    # 252 trading rows ~= one year.
    #
    # This is point-in-time safe because it only looks
    # backward within the same ticker.
    #
    # Later, we should improve this using SEC fiscal-period
    # observations directly.
    # ========================================================

    revenue_previous = (
        by_ticker["Revenue"]
        .shift(252)
    )

    income_previous = (
        by_ticker["Net_Income"]
        .shift(252)
    )

    fcf_previous = (
        df
        .groupby(
            level="Ticker",
            sort=False,
        )["Free_Cash_Flow"]
        .shift(252)
    )

    df["Revenue_Growth_YoY"] = (
        _pct_growth(
            df["Revenue"],
            revenue_previous,
        )
    )

    df["Net_Income_Growth_YoY"] = (
        _pct_growth(
            df["Net_Income"],
            income_previous,
        )
    )

    df["FCF_Growth_YoY"] = (
        _pct_growth(
            df["Free_Cash_Flow"],
            fcf_previous,
        )
    )

    return df.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    def print_regime_summary(
            panel: pd.DataFrame,
            start_date: str,
            end_date: str,
            label: str,
    ) -> None:

        dates = (
            panel.index
            .get_level_values("Date")
        )

        mask = (
                (dates >= pd.Timestamp(start_date))
                &
                (dates <= pd.Timestamp(end_date))
        )

        period = (
            panel.loc[mask]
            .reset_index()
            .groupby("Date")
            .first()
        )

        columns = [
            "VIX",
            "VIX_Change_20D",
            "Treasury_3M",
            "Treasury_3M_Change_63D",
            "Treasury_10Y",
            "Treasury_10Y_Change_63D",
            "Yield_Curve_10Y_3M",
            "Yield_Curve_Change_63D",
            "SPY_Return_63D",
            "SPY_Volatility_20D",
        ]

        print(
            f"\n{label} regime summary:"
        )

        print(
            period[
                columns
            ]
            .describe()
            .loc[
                [
                    "mean",
                    "std",
                    "min",
                    "max",
                ]
            ]
            .T
        )

        print("=" * 60)

def get_model_feature_frame(
    panel: pd.DataFrame,
) -> pd.DataFrame:
    """
    Return only the normalized model features.

    The Date/Ticker MultiIndex is preserved.
    """

    available_features = [
        feature
        for feature in MODEL_FEATURES_252D
        if feature in panel.columns
    ]

    missing_features = [
        feature
        for feature in MODEL_FEATURES_252D
        if feature not in panel.columns
    ]

    if missing_features:

        print(
            "\nMissing model features:"
        )

        for feature in missing_features:

            print(
                f"  - {feature}"
            )

    return panel[
        available_features
    ].copy()


def print_feature_coverage(
    panel: pd.DataFrame,
) -> None:
    """
    Print non-null coverage for every model feature.

    Example:

        RSI_14                     99.4%
        Revenue_Growth_YoY         74.2%

    This helps determine which features have too much
    missing data before model training.
    """

    available_features = [
        feature
        for feature in MODEL_FEATURES_252D
        if feature in panel.columns
    ]

    if not available_features:

        print(
            "No model features found."
        )

        return

    coverage = (
        panel[
            available_features
        ]
        .notna()
        .mean()
        .sort_values(
            ascending=False
        )
    )

    print(
        "\nFeature coverage:"
    )

    for feature, ratio in coverage.items():

        print(
            f"{feature:<30} "
            f"{ratio:>7.1%}"
        )

    print("=" * 60)