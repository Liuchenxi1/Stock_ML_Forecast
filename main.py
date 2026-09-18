from stock_ml_forecast.universe import (
    get_sp500_universe,
)

from stock_ml_forecast.panel import (
    build_sp500_panel,
    load_sp500_panel,
)

from stock_ml_forecast.paths import (
    SP500_PANEL_PATH,
)

from stock_ml_forecast.panel_features import (
    add_panel_features,
    get_model_feature_frame,
    print_feature_coverage,
)


SEC_USER_AGENT = (
    "Stock ML Forecast your_email@example.com"
)

BENCHMARK_TICKER = "SPY"

START_DATE = "2016-01-01"

END_DATE = None

FORCE_REFRESH = False

MAX_COMPANIES = None


universe = get_sp500_universe()

print(
    f"S&P 500 universe: "
    f"{len(universe)} companies"
)

print(
    universe.head()
)

print("=" * 60)


if (
    SP500_PANEL_PATH.exists()
    and not FORCE_REFRESH
):

    print(
        "Loading existing processed panel"
    )

    panel = load_sp500_panel()

else:

    print(
        "Building S&P 500 panel"
    )

    panel = build_sp500_panel(
        universe=universe,
        user_agent=SEC_USER_AGENT,
        benchmark_ticker=BENCHMARK_TICKER,
        start_date=START_DATE,
        end_date=END_DATE,
        force_refresh=FORCE_REFRESH,
        max_companies=MAX_COMPANIES,
    )



print_feature_coverage(
    panel
)

X = get_model_feature_frame(
    panel
)

print(
    "Feature matrix shape:",
    X.shape,
)

print(
    "\nLatest feature rows:"
)

print(
    X.tail()
)

print("=" * 60)