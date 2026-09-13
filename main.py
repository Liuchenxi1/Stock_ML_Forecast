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


# ============================================================
# Configuration
# ============================================================

SEC_USER_AGENT = (
    "Stock ML Forecast shinnkiryu@gmail.com"
)

BENCHMARK_TICKER = "SPY"

START_DATE = "2016-01-01"

END_DATE = None


# False:
#     use local caches whenever they exist
#
# True:
#     download everything again
#
FORCE_REFRESH = False


# During development:
#
# Use 5 or 10 companies.
#
# When everything works:
#
# MAX_COMPANIES = None
#
MAX_COMPANIES = 5


# ============================================================
# S&P 500 universe
# ============================================================

universe = (
    get_sp500_universe()
)

print(
    f"S&P 500 universe: "
    f"{len(universe)} companies"
)

print(
    universe.head()
)

print("=" * 60)


# ============================================================
# Processed panel
# ============================================================

if (
    SP500_PANEL_PATH.exists()
    and not FORCE_REFRESH
):

    print(
        "Loading existing processed panel"
    )

    panel = (
        load_sp500_panel()
    )

else:

    print(
        "Building S&P 500 panel"
    )

    panel = (
        build_sp500_panel(
            universe=universe,
            user_agent=SEC_USER_AGENT,
            benchmark_ticker=BENCHMARK_TICKER,
            start_date=START_DATE,
            end_date=END_DATE,
            force_refresh=FORCE_REFRESH,
            max_companies=MAX_COMPANIES,
        )
    )


# ============================================================
# Inspect result
# ============================================================

print(
    "\nPanel shape:",
    panel.shape,
)

print(
    "Companies:",
    panel
    .index
    .get_level_values(
        "Ticker"
    )
    .nunique(),
)

print(
    "Date range:",
    panel
    .index
    .get_level_values(
        "Date"
    )
    .min(),
    "->",
    panel
    .index
    .get_level_values(
        "Date"
    )
    .max(),
)

print(
    "\nLatest observations:"
)

print(
    panel.tail()
)

print("=" * 60)