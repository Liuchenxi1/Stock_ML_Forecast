MODEL_FEATURES = [
    # Stock / market behavior

    "Stock_Return_1D",
    "Benchmark_Return_1D",

    # Trend
    "Price_to_MA20",
    "Price_to_MA50",
    "Price_to_MA200",
    "MA20_to_MA200",

    # Momentum
    "RSI_14",
    "MACD",
    "MACD_Signal",
    "MACD_Histogram",

    # Volatility
    "Volatility_20D",
    "Volatility_60D",

    # Money flow
    "Money_Flow_Multiplier",
    "Flow_5D",
    "Flow_20D",
    "Flow_60D",

    # Market risk / interest rates
    "VIX",
    "VIX_Change",
    "Treasury_3M",
    "Treasury_10Y",
    "Treasury_3M_Change",
    "Treasury_10Y_Change",
    "Yield_Curve_10Y_3M",

    # SEC fundamentals
    "Total_Assets",
    "Total_Liabilities",
    "Cash",
    "Total_Debt",
    "Stockholders_Equity",
    "Revenue",
    "Net_Income",
    "Operating_Cash_Flow",
    "Capital_Expenditures",
]


FORECAST_HORIZONS = (
    63,
    126,
    252,
)