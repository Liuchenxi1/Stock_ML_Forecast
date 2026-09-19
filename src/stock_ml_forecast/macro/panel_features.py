import numpy as np
import pandas as pd


# ============================================================
# Feature groups
# ============================================================

MARKET_FEATURES = [
    "Stock_Return_1D",
    "Benchmark_Return_1D",
    "Price_to_MA20",
    "Price_to_MA50",
    "Price_to_MA200",
    "RSI_14",
    "MACD_Pct",
    "Volatility_20D",
    "Yield_Curve_10Y_3M",

    # Macro / regime
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

    # Market confirmation
    "Volume_Ratio_20D",
    "Return_20D",
    "Return_63D",
    "Excess_Return_vs_SPY_63D",
]


FUNDAMENTAL_FEATURES = [
    "Revenue_Growth_YoY",
    "Net_Income_Growth_YoY",
    "FCF_Growth_YoY",

    "Debt_to_Assets",
    "Debt_to_Equity",
    "Cash_to_Debt",

    "ROA",
    "ROE",
    "FCF_Margin",
]


MODEL_FEATURES_252D = (
    MARKET_FEATURES
    + FUNDAMENTAL_FEATURES
)


# ============================================================
# Utility functions
# ============================================================

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
    Calculate:

        current / previous - 1

    Previous values equal to zero are treated as missing.
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


# ============================================================
# Main panel feature engineering
# ============================================================

def add_panel_features(
    panel: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add normalized cross-company, fundamental,
    market-confirmation, and macro-regime features.

    Expected index:

        Date
        Ticker
    """

    df = panel.copy()

    # --------------------------------------------------------
    # Validate index
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

    df = df.sort_index()

    # --------------------------------------------------------
    # Group by company
    # --------------------------------------------------------

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
    # Stock returns
    # --------------------------------------------------------

    df["Return_20D"] = (
        by_ticker["Stock_Adj_Close"]
        .pct_change(
            periods=20,
            fill_method=None,
        )
    )

    df["Return_63D"] = (
        by_ticker["Stock_Adj_Close"]
        .pct_change(
            periods=63,
            fill_method=None,
        )
    )

    # --------------------------------------------------------
    # SPY returns
    # --------------------------------------------------------

    df["SPY_Return_20D"] = (
        by_ticker["Benchmark_Close"]
        .pct_change(
            periods=20,
            fill_method=None,
        )
    )

    df["SPY_Return_63D"] = (
        by_ticker["Benchmark_Close"]
        .pct_change(
            periods=63,
            fill_method=None,
        )
    )

    df["Excess_Return_vs_SPY_63D"] = (
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
    # Later we can replace this with true SEC
    # comparable-period calculations.
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

    # ========================================================
    # Macro / regime features
    # ========================================================

    # --------------------------------------------------------
    # SPY realized volatility
    # --------------------------------------------------------

    df["SPY_Volatility_20D"] = (
        by_ticker["Benchmark_Return_1D"]
        .transform(
            lambda series:
                series
                .rolling(
                    window=20,
                    min_periods=20,
                )
                .std()
        )
    )

    # --------------------------------------------------------
    # VIX regime movement
    # --------------------------------------------------------

    df["VIX_Change_20D"] = (
        by_ticker["VIX"]
        .pct_change(
            periods=20,
            fill_method=None,
        )
    )

    df["VIX_Change_63D"] = (
        by_ticker["VIX"]
        .pct_change(
            periods=63,
            fill_method=None,
        )
    )

    # --------------------------------------------------------
    # Short-term interest-rate velocity
    # --------------------------------------------------------

    df["Treasury_3M_Change_20D"] = (
        by_ticker["Treasury_3M"]
        .diff(20)
    )

    df["Treasury_3M_Change_63D"] = (
        by_ticker["Treasury_3M"]
        .diff(63)
    )

    # --------------------------------------------------------
    # Long-term interest-rate velocity
    # --------------------------------------------------------

    df["Treasury_10Y_Change_20D"] = (
        by_ticker["Treasury_10Y"]
        .diff(20)
    )

    df["Treasury_10Y_Change_63D"] = (
        by_ticker["Treasury_10Y"]
        .diff(63)
    )

    # --------------------------------------------------------
    # Yield-curve movement
    # --------------------------------------------------------

    df["Yield_Curve_Change_20D"] = (
        by_ticker["Yield_Curve_10Y_3M"]
        .diff(20)
    )

    df["Yield_Curve_Change_63D"] = (
        by_ticker["Yield_Curve_10Y_3M"]
        .diff(63)
    )

    # ========================================================
    # Final cleanup
    # ========================================================

    df = df.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    # --------------------------------------------------------
    # Safety check
    # --------------------------------------------------------

    missing_features = [
        feature
        for feature in MODEL_FEATURES_252D
        if feature not in df.columns
    ]

    if missing_features:
        raise RuntimeError(
            "add_panel_features failed to create "
            f"these model features: {missing_features}"
        )

    return df


# ============================================================
# Feature matrix
# ============================================================

def get_model_feature_frame(
    panel: pd.DataFrame,
) -> pd.DataFrame:
    """
    Return the complete model feature matrix.

    Raises an error instead of silently dropping
    missing features.
    """

    missing_features = [
        feature
        for feature in MODEL_FEATURES_252D
        if feature not in panel.columns
    ]

    if missing_features:
        raise ValueError(
            "Missing model features: "
            f"{missing_features}"
        )

    return panel[
        MODEL_FEATURES_252D
    ].copy()


# ============================================================
# Feature coverage
# ============================================================

def print_feature_coverage(
    panel: pd.DataFrame,
) -> None:
    """
    Print percentage of non-null observations
    for each model feature.
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

    print(
        "\nFeature coverage:"
    )

    if available_features:

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

        for feature, ratio in coverage.items():

            print(
                f"{feature:<32}"
                f"{ratio:>7.1%}"
            )

    if missing_features:

        print(
            "\nMissing model features:"
        )

        for feature in missing_features:

            print(
                f"  - {feature}"
            )

    print("=" * 60)


# ============================================================
# Macro regime diagnostics
# ============================================================

def print_regime_summary(
    panel: pd.DataFrame,
    start_date: str,
    end_date: str,
    label: str,
) -> None:
    """
    Print macro-regime statistics for a specified period.

    Because macro columns are repeated across all tickers,
    only one observation per Date is used.
    """

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
        "VIX_Change_63D",

        "Treasury_3M",
        "Treasury_3M_Change_20D",
        "Treasury_3M_Change_63D",

        "Treasury_10Y",
        "Treasury_10Y_Change_20D",
        "Treasury_10Y_Change_63D",

        "Yield_Curve_10Y_3M",
        "Yield_Curve_Change_20D",
        "Yield_Curve_Change_63D",

        "SPY_Return_20D",
        "SPY_Return_63D",
        "SPY_Volatility_20D",
    ]

    missing = [
        column
        for column in columns
        if column not in period.columns
    ]

    if missing:
        raise ValueError(
            "Cannot print regime summary. "
            f"Missing columns: {missing}"
        )

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