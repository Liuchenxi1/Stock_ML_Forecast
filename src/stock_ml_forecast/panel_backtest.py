from dataclasses import dataclass
import math

import numpy as np
import pandas as pd


@dataclass
class RankingBacktestResult:
    summary: pd.DataFrame
    monthly: pd.DataFrame


def _get_monthly_snapshot_dates(
    index: pd.MultiIndex,
) -> pd.DatetimeIndex:
    """
    Select one evaluation date per calendar month.

    We use the first available trading date in each month.
    """

    dates = pd.DatetimeIndex(
        index
        .get_level_values("Date")
        .unique()
    ).sort_values()

    date_table = pd.DataFrame(
        {
            "Date": dates,
        }
    )

    date_table["Month"] = (
        date_table["Date"]
        .dt
        .to_period("M")
    )

    monthly_dates = (
        date_table
        .groupby("Month")["Date"]
        .min()
    )

    return pd.DatetimeIndex(
        monthly_dates.to_numpy()
    )


def _evaluate_one_ranking_method(
    data: pd.DataFrame,
    score_column: str,
    method_name: str,
    top_fraction: float = 0.10,
    min_companies: int = 50,
) -> pd.DataFrame:
    """
    Evaluate one ranking method month by month.

    Higher score = better rank.
    """

    rows = []

    for date, group in data.groupby(
        level="Date"
    ):

        group = (
            group
            .dropna(
                subset=[
                    score_column,
                    "Actual_Top10",
                    "Actual_Excess_Return",
                ]
            )
            .copy()
        )

        if len(group) < min_companies:
            continue

        number_selected = max(
            1,
            math.ceil(
                len(group)
                * top_fraction
            ),
        )

        selected = (
            group
            .nlargest(
                number_selected,
                score_column,
            )
        )

        actual_top10_count = (
            group[
                "Actual_Top10"
            ]
            .sum()
        )

        true_positives = (
            selected[
                "Actual_Top10"
            ]
            .sum()
        )

        precision = (
            selected[
                "Actual_Top10"
            ]
            .mean()
        )

        if actual_top10_count > 0:

            recall = (
                true_positives
                / actual_top10_count
            )

        else:

            recall = np.nan

        selected_mean_return = (
            selected[
                "Actual_Excess_Return"
            ]
            .mean()
        )

        selected_median_return = (
            selected[
                "Actual_Excess_Return"
            ]
            .median()
        )

        universe_mean_return = (
            group[
                "Actual_Excess_Return"
            ]
            .mean()
        )

        return_lift = (
            selected_mean_return
            - universe_mean_return
        )

        rows.append(
            {
                "Date": date,
                "Method": method_name,

                "Universe_Size":
                    len(group),

                "Selected_Count":
                    len(selected),

                "Precision_at_10pct":
                    precision,

                "Recall_at_10pct":
                    recall,

                "Selected_Mean_Excess_Return":
                    selected_mean_return,

                "Selected_Median_Excess_Return":
                    selected_median_return,

                "Universe_Mean_Excess_Return":
                    universe_mean_return,

                "Return_Lift":
                    return_lift,
            }
        )

    return pd.DataFrame(
        rows
    )


def run_monthly_ranking_backtest(
    classifier,
    X: pd.DataFrame,
    y_top10: pd.Series,
    y_return: pd.Series,
    label: str,
    top_fraction: float = 0.10,
    min_companies: int = 50,
) -> RankingBacktestResult:
    """
    Compare ML ranking against simple momentum baselines.

    Methods
    -------
    ML_Probability
        Predicted probability of future Top-10% membership.

    Momentum_63D
        Rank directly by trailing 63-day stock return.

    Relative_Momentum_63D
        Rank by trailing 63-day excess return vs SPY.
    """

    # ========================================================
    # Monthly snapshots only
    # ========================================================

    monthly_dates = (
        _get_monthly_snapshot_dates(
            X.index
        )
    )

    date_values = (
        X.index
        .get_level_values("Date")
    )

    monthly_mask = (
        date_values.isin(
            monthly_dates
        )
    )

    X_monthly = (
        X.loc[
            monthly_mask
        ]
        .copy()
    )

    y_top10_monthly = (
        y_top10.loc[
            X_monthly.index
        ]
        .copy()
    )

    y_return_monthly = (
        y_return.loc[
            X_monthly.index
        ]
        .copy()
    )

    # ========================================================
    # Build common evaluation table
    # ========================================================

    evaluation = pd.DataFrame(
        index=X_monthly.index
    )

    evaluation[
        "ML_Probability"
    ] = (
        classifier
        .predict_proba(
            X_monthly
        )[:, 1]
    )

    evaluation[
        "Momentum_63D"
    ] = (
        X_monthly[
            "Return_63D"
        ]
    )

    evaluation[
        "Relative_Momentum_63D"
    ] = (
        X_monthly[
            "Excess_Return_vs_SPY_63D"
        ]
    )

    evaluation[
        "Actual_Top10"
    ] = (
        y_top10_monthly
        .astype(int)
    )

    evaluation[
        "Actual_Excess_Return"
    ] = (
        y_return_monthly
        .astype(float)
    )

    # --------------------------------------------------------
    # Fair comparison:
    #
    # Require both momentum baseline features to exist.
    #
    # Then ML and momentum are evaluated on the same stocks.
    # --------------------------------------------------------

    evaluation = (
        evaluation
        .dropna(
            subset=[
                "Momentum_63D",
                "Relative_Momentum_63D",
                "Actual_Top10",
                "Actual_Excess_Return",
            ]
        )
        .copy()
    )

    # ========================================================
    # Evaluate all ranking methods
    # ========================================================

    methods = {
        "ML Top-10 Probability":
            "ML_Probability",

        "63D Momentum":
            "Momentum_63D",

        "63D Relative Momentum":
            "Relative_Momentum_63D",
    }

    results = []

    for (
        method_name,
        score_column,
    ) in methods.items():

        result = (
            _evaluate_one_ranking_method(
                data=evaluation,
                score_column=score_column,
                method_name=method_name,
                top_fraction=top_fraction,
                min_companies=min_companies,
            )
        )

        results.append(
            result
        )

    monthly = pd.concat(
        results,
        ignore_index=True,
    )

    # ========================================================
    # Summary
    # ========================================================

    summary = (
        monthly
        .groupby("Method")
        .agg(
            Evaluation_Months=(
                "Date",
                "nunique",
            ),

            Mean_Universe_Size=(
                "Universe_Size",
                "mean",
            ),

            Precision_at_10pct=(
                "Precision_at_10pct",
                "mean",
            ),

            Recall_at_10pct=(
                "Recall_at_10pct",
                "mean",
            ),

            Mean_Selected_Excess_Return=(
                "Selected_Mean_Excess_Return",
                "mean",
            ),

            Median_Selected_Excess_Return=(
                "Selected_Median_Excess_Return",
                "median",
            ),

            Universe_Excess_Return=(
                "Universe_Mean_Excess_Return",
                "mean",
            ),

            Return_Lift=(
                "Return_Lift",
                "mean",
            ),
        )
        .reset_index()
    )

    summary[
        "Precision_Lift_vs_Random"
    ] = (
        summary[
            "Precision_at_10pct"
        ]
        / top_fraction
    )

    summary[
        "Dataset"
    ] = label

    return RankingBacktestResult(
        summary=summary,
        monthly=monthly,
    )


def print_monthly_backtest(
    result: RankingBacktestResult,
    label: str,
) -> None:
    """
    Print compact monthly ranking comparison.
    """

    print(
        f"\n{label} monthly ranking backtest:"
    )

    for _, row in (
        result.summary.iterrows()
    ):

        print(
            f"\n{row['Method']}"
        )

        print(
            f"{'Evaluation months':<32}"
            f"{int(row['Evaluation_Months'])}"
        )

        print(
            f"{'Average universe size':<32}"
            f"{row['Mean_Universe_Size']:.0f}"
        )

        print(
            f"{'Precision @ 10%':<32}"
            f"{row['Precision_at_10pct']:.2%}"
        )

        print(
            f"{'Recall @ 10%':<32}"
            f"{row['Recall_at_10pct']:.2%}"
        )

        print(
            f"{'Precision lift vs random':<32}"
            f"{row['Precision_Lift_vs_Random']:.2f}x"
        )

        print(
            f"{'Mean selected excess return':<32}"
            f"{row['Mean_Selected_Excess_Return']:.2%}"
        )

        print(
            f"{'Median selected excess return':<32}"
            f"{row['Median_Selected_Excess_Return']:.2%}"
        )

        print(
            f"{'Universe excess return':<32}"
            f"{row['Universe_Excess_Return']:.2%}"
        )

        print(
            f"{'Return lift':<32}"
            f"{row['Return_Lift']:.2%}"
        )

    print("=" * 60)