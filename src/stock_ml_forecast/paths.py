from pathlib import Path


# Project root:
#
# Stock_ML_Forecast/
#
PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)


DATA_DIR = (
    PROJECT_ROOT
    / "data"
)

RAW_DATA_DIR = (
    DATA_DIR
    / "raw"
)

MARKET_DATA_DIR = (
    RAW_DATA_DIR
    / "market"
)

SEC_DATA_DIR = (
    RAW_DATA_DIR
    / "sec"
)

PROCESSED_DATA_DIR = (
    DATA_DIR
    / "processed"
)


SP500_PANEL_PATH = (
    PROCESSED_DATA_DIR
    / "sp500_panel.parquet"
)


def ensure_data_directories() -> None:
    """
    Create all required data directories.
    """

    for directory in (
        MARKET_DATA_DIR,
        SEC_DATA_DIR,
        PROCESSED_DATA_DIR,
    ):
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )