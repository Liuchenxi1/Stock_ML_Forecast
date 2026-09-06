# src/stock_ml_forecast/sec_data.py

import numpy as np
import pandas as pd
import requests


SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"

SEC_COMPANY_FACTS_URL = (
    "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
)


# ============================================================
# SEC XBRL concepts
# ============================================================

FUNDAMENTAL_CONCEPTS = {
    "Total_Assets": [
        "Assets",
    ],

    "Total_Liabilities": [
        "Liabilities",
    ],

    "Cash": [
        "CashAndCashEquivalentsAtCarryingValue",
        "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents",
    ],

    "Stockholders_Equity": [
        "StockholdersEquity",
        "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
    ],

    "Revenue": [
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "Revenues",
        "SalesRevenueNet",
    ],

    "Net_Income": [
        "NetIncomeLoss",
        "ProfitLoss",
    ],

    "Operating_Cash_Flow": [
        "NetCashProvidedByUsedInOperatingActivities",
    ],

    "Capital_Expenditures": [
        "PaymentsToAcquirePropertyPlantAndEquipment",
        "PaymentsForAdditionsToPropertyPlantAndEquipment",
    ],
}


DEBT_COMPONENT_CONCEPTS = {
    "Debt_Current": [
        "LongTermDebtAndFinanceLeaseObligationsCurrent",
        "LongTermDebtCurrent",
    ],

    "Debt_Noncurrent": [
        "LongTermDebtAndFinanceLeaseObligationsNoncurrent",
        "LongTermDebtNoncurrent",
    ],
}


# ============================================================
# SEC HTTP helpers
# ============================================================

def _make_headers(user_agent: str) -> dict:
    """
    Create HTTP headers required by the SEC.

    The SEC asks automated clients to identify themselves.
    Ideally use something such as:

        "Stock ML Forecast your_email@example.com"
    """

    if not user_agent:
        raise ValueError(
            "SEC user_agent cannot be empty. "
            "Use something like "
            "'Stock ML Forecast your_email@example.com'."
        )

    return {
        "User-Agent": user_agent,
        "Accept-Encoding": "gzip, deflate",
    }


# ============================================================
# Ticker -> CIK
# ============================================================

def lookup_sec_company(
    ticker: str,
    user_agent: str,
) -> dict | None:
    """
    Look up an SEC company using its ticker.

    Returns
    -------
    dict | None

    Example:
        {
            "ticker": "HCA",
            "title": "HCA Healthcare, Inc.",
            "cik": "0000860730"
        }
    """

    headers = _make_headers(user_agent)

    response = requests.get(
        SEC_TICKERS_URL,
        headers=headers,
        timeout=30,
    )

    response.raise_for_status()

    companies = response.json()

    ticker = ticker.strip().upper()

    for company in companies.values():

        company_ticker = str(
            company.get("ticker", "")
        ).upper()

        if company_ticker == ticker:

            return {
                "ticker": ticker,
                "title": company.get("title"),
                "cik": str(
                    company.get("cik_str")
                ).zfill(10),
            }

    return None


# ============================================================
# Download Company Facts
# ============================================================

def download_company_facts(
    cik: str,
    user_agent: str,
) -> dict:
    """
    Download SEC Company Facts for a CIK.
    """

    headers = _make_headers(user_agent)

    cik = str(cik).zfill(10)

    url = SEC_COMPANY_FACTS_URL.format(
        cik=cik
    )

    response = requests.get(
        url,
        headers=headers,
        timeout=30,
    )

    response.raise_for_status()

    return response.json()


# ============================================================
# Extract a single SEC concept
# ============================================================

def get_sec_fact(
    facts: dict,
    concept: str,
    unit: str = "USD",
    forms: tuple[str, ...] = ("10-Q", "10-K"),
) -> pd.DataFrame:
    """
    Extract filing observations for one SEC XBRL concept.
    """

    if concept not in facts:
        return pd.DataFrame()

    concept_data = facts[concept]

    units = concept_data.get("units", {})

    if not units:
        return pd.DataFrame()

    # Prefer requested unit.
    # If unavailable, fall back to the first unit.
    if unit not in units:
        available_units = list(units.keys())

        if not available_units:
            return pd.DataFrame()

        unit = available_units[0]

    rows = []

    for item in units[unit]:

        if item.get("form") not in forms:
            continue

        rows.append(
            {
                "start": item.get("start"),
                "end": item.get("end"),
                "filed": item.get("filed"),
                "form": item.get("form"),
                "fy": item.get("fy"),
                "fp": item.get("fp"),
                "value": item.get("val"),
            }
        )

    result = pd.DataFrame(rows)

    if result.empty:
        return result

    for column in (
        "start",
        "end",
        "filed",
    ):
        result[column] = pd.to_datetime(
            result[column],
            errors="coerce",
        )

    return result.sort_values(
        ["filed", "end"]
    )


# ============================================================
# Try alternative SEC concepts
# ============================================================

def first_available_fact(
    facts: dict,
    concepts: list[str],
) -> tuple[str | None, pd.DataFrame]:
    """
    Return the first SEC XBRL concept that has data.
    """

    for concept in concepts:

        table = get_sec_fact(
            facts=facts,
            concept=concept,
        )

        if not table.empty:
            return concept, table

    return None, pd.DataFrame()


# ============================================================
# Prepare filing-date table
# ============================================================

def prepare_fundamental_table(
    table: pd.DataFrame,
    column_name: str,
) -> pd.DataFrame:
    """
    Convert SEC observations into a point-in-time
    filing-date table.

    The financial value becomes available on the filing
    date rather than the fiscal period end.
    """

    if table.empty:
        return pd.DataFrame(
            columns=[
                "Available_Date",
                column_name,
            ]
        )

    result = table.copy()

    result = result.rename(
        columns={
            "filed": "Available_Date",
            "value": column_name,
        }
    )

    result = result[
        [
            "Available_Date",
            "end",
            column_name,
        ]
    ].dropna()

    # If multiple facts exist on the same filing date,
    # keep the observation associated with the latest
    # fiscal period end.
    result = (
        result
        .sort_values(
            ["Available_Date", "end"]
        )
        .drop_duplicates(
            "Available_Date",
            keep="last",
        )
        [
            [
                "Available_Date",
                column_name,
            ]
        ]
    )

    return result


# ============================================================
# Point-in-time merge
# ============================================================

def merge_point_in_time(
    market_df: pd.DataFrame,
    fundamental_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge SEC data into daily market data using the
    most recent filing available on or before each date.
    """

    if fundamental_df.empty:
        return market_df.copy()

    left = (
        market_df
        .reset_index()
        .sort_values("Date")
    )

    right = (
        fundamental_df
        .sort_values("Available_Date")
        .copy()
    )

    # Force both merge keys to the exact same dtype.
    left["Date"] = pd.to_datetime(
        left["Date"],
        errors="coerce",
    ).astype("datetime64[ns]")

    right["Available_Date"] = pd.to_datetime(
        right["Available_Date"],
        errors="coerce",
    ).astype("datetime64[ns]")

    # Remove invalid dates if any exist.
    left = left.dropna(subset=["Date"])
    right = right.dropna(subset=["Available_Date"])

    merged = pd.merge_asof(
        left.sort_values("Date"),
        right.sort_values("Available_Date"),
        left_on="Date",
        right_on="Available_Date",
        direction="backward",
    )

    merged = (
        merged
        .drop(
            columns=["Available_Date"],
            errors="ignore",
        )
        .set_index("Date")
    )

    return merged

# Main fundamental merge

def add_sec_fundamentals(
    df: pd.DataFrame,
    ticker: str,
    user_agent: str,
) -> pd.DataFrame:
    """
    Download SEC fundamentals and merge them into the
    daily market DataFrame.

    Fundamentals are merged using filing dates so the
    model cannot use information before it became public.
    """

    result = df.copy()

    # --------------------------------------------------------
    # Find company
    # --------------------------------------------------------

    company = lookup_sec_company(
        ticker=ticker,
        user_agent=user_agent,
    )

    if company is None:

        print(
            f"No SEC CIK found for {ticker}. "
            "Continuing without SEC fundamentals."
        )

        for column in FUNDAMENTAL_CONCEPTS:
            result[column] = np.nan

        result["Total_Debt"] = np.nan

        return result

    print(
        f"SEC company: {company['title']}"
    )

    print(
        f"Ticker: {company['ticker']}"
    )

    print(
        f"CIK: {company['cik']}"
    )

    # Download Company Facts

    sec_data = download_company_facts(
        cik=company["cik"],
        user_agent=user_agent,
    )

    facts = (
        sec_data
        .get("facts", {})
        .get("us-gaap", {})
    )

    print(
        f"US-GAAP concepts: {len(facts)}"
    )

    # Core fundamentals

    for (
        column_name,
        candidate_concepts,
    ) in FUNDAMENTAL_CONCEPTS.items():

        concept, table = first_available_fact(
            facts=facts,
            concepts=candidate_concepts,
        )

        if table.empty:

            print(
                f"{column_name}: NOT FOUND"
            )

            result[column_name] = np.nan

            continue

        prepared = prepare_fundamental_table(
            table=table,
            column_name=column_name,
        )

        result = merge_point_in_time(
            market_df=result,
            fundamental_df=prepared,
        )

        print(
            f"{column_name}: {concept} "
            f"({len(prepared)} filing observations)"
        )

    # Debt

    debt_columns = []

    for (
        column_name,
        candidate_concepts,
    ) in DEBT_COMPONENT_CONCEPTS.items():

        concept, table = first_available_fact(
            facts=facts,
            concepts=candidate_concepts,
        )

        if table.empty:
            continue

        prepared = prepare_fundamental_table(
            table=table,
            column_name=column_name,
        )

        result = merge_point_in_time(
            market_df=result,
            fundamental_df=prepared,
        )

        debt_columns.append(
            column_name
        )

        print(
            f"{column_name}: {concept}"
        )

    # Preferred:
    # current debt + noncurrent debt
    if debt_columns:

        result["Total_Debt"] = (
            result[debt_columns]
            .sum(
                axis=1,
                min_count=1,
            )
        )

    else:

        # Fallback combined debt concept
        concept, table = first_available_fact(
            facts=facts,
            concepts=[
                "LongTermDebt",
            ],
        )

        if not table.empty:

            prepared = prepare_fundamental_table(
                table=table,
                column_name="Total_Debt",
            )

            result = merge_point_in_time(
                market_df=result,
                fundamental_df=prepared,
            )

            print(
                f"Total_Debt fallback: {concept}"
            )

        else:

            print(
                "WARNING: No debt concept found."
            )

            result["Total_Debt"] = np.nan

    return result.sort_index()