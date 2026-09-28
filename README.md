# Stock ML Forecast

A machine-learning research project for ranking S&P 500 companies by their probability of becoming future outperformers over approximately the next 252 trading days.

The goal is not simply to predict whether a stock goes up or down. The main classifier ranks companies by their probability of finishing in the **top 10% of future S&P 500 performers** based on 252-day excess return versus SPY.

> This project is experimental quantitative research. Backtest and walk-forward results are not guaranteed future returns and should not be interpreted as investment advice.

---

## Current Dataset

Two processed panels are currently used:

```text
Normal panel:
~1.36 million Date/Ticker rows
503 current S&P 500 securities
2016 -> latest

Long-history experiment panel:
2,624,627 Date/Ticker rows
503 current S&P 500 securities
2006-01-03 -> 2026-09-28
```

The long-history panel was created to test whether exposure to earlier market regimes such as the 2008 financial crisis improves later out-of-sample ranking performance.

---

## Prediction Target

The main classification target is:

```text
Top_10pct_252D
```

A positive label means the stock finished in approximately the top 10% of the cross-sectional S&P 500 universe based on future 252-trading-day excess return versus SPY.

The project also calculates:

```text
Forward_Stock_Return_252D
Forward_SPY_Return_252D
Forward_Excess_Return_252D
```

The positive classification rate is approximately:

```text
10.1%
```

Because the target is highly imbalanced, ordinary classification accuracy is not the primary metric. The main evaluation metric is **Precision @ 10%**.

---

## Models

The current baseline models are:

```text
HistGradientBoostingClassifier
HistGradientBoostingRegressor
```

The classifier is the primary model because the project goal is cross-sectional ranking.

The regressor predicts future 252-day excess return and is used as a secondary research signal.

---

## Feature Sets

### MARKET_MACRO — 24 Features

This is the current benchmark feature set.

#### Stock / Technical

```text
Stock_Return_1D
Benchmark_Return_1D
Price_to_MA20
Price_to_MA50
Price_to_MA200
RSI_14
MACD_Pct
Volatility_20D
Volume_Ratio_20D
Return_20D
Return_63D
Excess_Return_vs_SPY_63D
```

#### Market / SPY

```text
SPY_Return_20D
SPY_Return_63D
SPY_Volatility_20D
```

#### Macro / Regime

```text
Yield_Curve_10Y_3M
VIX_Change_20D
VIX_Change_63D
Treasury_3M_Change_20D
Treasury_3M_Change_63D
Treasury_10Y_Change_20D
Treasury_10Y_Change_63D
Yield_Curve_Change_20D
Yield_Curve_Change_63D
```

---

### Fundamentals — 9 Features

SEC Company Facts are used to construct:

```text
Revenue_Growth_YoY
Net_Income_Growth_YoY
FCF_Growth_YoY
Debt_to_Assets
Debt_to_Equity
Cash_to_Debt
ROA
ROE
FCF_Margin
```

Free cash flow is currently calculated as:

```text
Free_Cash_Flow = Operating_Cash_Flow - Capital_Expenditures
```

The combined full model contains:

```text
24 MARKET_MACRO
+ 9 fundamentals
= 33 total features
```

---

## Time-Series Validation

Financial observations are never randomly shuffled.

The project uses expanding-window walk-forward validation with approximately a 252-trading-day purge between training and validation. This reduces leakage because each target itself looks one trading year into the future.

Example:

```text
Train:       historical observations
Purge:       ~252 trading days
Validation:  next ~6 months
```

The same eight validation periods are reused across feature-set and history-length experiments so results can be compared directly.

---

## MARKET_MACRO Walk-Forward Results

### 2016-History MARKET_MACRO

```text
Fold 1: 30.90%
Fold 2: 26.82%
Fold 3: 21.10%
Fold 4: 20.57%
Fold 5: 18.57%
Fold 6: 29.14%
Fold 7: 30.00%
Fold 8: 31.43%

Average Precision @ 10%: 26.07%
```

### 2006-History MARKET_MACRO

```text
Fold 1: 34.99%
Fold 2: 28.57%
Fold 3: 23.03%
Fold 4: 21.80%
Fold 5: 16.86%
Fold 6: 29.71%
Fold 7: 28.86%
Fold 8: 27.71%

Average Precision @ 10%: 26.44%
```

The longer history improved the average only slightly:

```text
26.07% -> 26.44%
+0.37 percentage points
```

It did improve the previously weaker Fold 3 and Fold 4 periods:

```text
Fold 3: 21.10% -> 23.03%
Fold 4: 20.57% -> 21.80%
```

This suggests older regimes add some useful information, but simply adding more history does not improve every period.

---

## Long-History Full 33-Feature Experiment

The 9 fundamental features were added back to the 2006-history MARKET_MACRO model while keeping the same eight validation periods.

Results:

| Fold | MARKET_MACRO 24 | ALL 33 | Change |
|---:|---:|---:|---:|
| 1 | 34.99% | 31.20% | -3.79 pp |
| 2 | 28.57% | 25.66% | -2.91 pp |
| 3 | 23.03% | 15.16% | -7.87 pp |
| 4 | 21.80% | 14.90% | -6.90 pp |
| 5 | 16.86% | 20.29% | +3.43 pp |
| 6 | 29.71% | 32.29% | +2.58 pp |
| 7 | 28.86% | 29.14% | +0.28 pp |
| 8 | 27.71% | 22.86% | -4.85 pp |

Average:

```text
MARKET_MACRO 24: 26.44%
ALL 33:          23.94%
Difference:      -2.51 percentage points
```

The full 33-feature model improved 3 of 8 folds but worsened 5 of 8 folds.

The largest degradation occurred in Fold 3 and Fold 4:

```text
Fold 3: 23.03% -> 15.16%
Fold 4: 21.80% -> 14.90%
```

The current evidence therefore favors keeping MARKET_MACRO as the benchmark and testing fundamental features selectively rather than automatically combining all nine.

---

## Earlier ALL-Feature Result

Before the MARKET_MACRO ablation, the earlier 33-feature walk-forward model produced:

```text
Fold 1: 17.49%
Fold 2: 20.12%
Fold 3:  8.75%
Fold 4:  4.05%
Fold 5: 20.00%
Fold 6: 29.71%
Fold 7: 27.14%
Fold 8: 24.00%

Average Precision @ 10%: 18.91%
```

This was the main reason for testing MARKET_MACRO separately.

---

## Current Benchmark

The current research benchmark is:

```text
Feature set:      MARKET_MACRO
Features:         24
History:          2006 -> latest
Model:            HistGradientBoostingClassifier
Average P@10:     26.44%
```

Random top-10% selection would produce approximately 10% precision, so the current walk-forward result represents roughly 2.6x the random concentration of future top-decile observations.

This is a ranking diagnostic, not a portfolio CAGR.

---

## Fundamental Research Direction

Fundamental features are not being discarded.

The current result suggests that some individual fundamental variables may contain useful signal, but the full 9-feature block reduces robustness when combined with MARKET_MACRO.

The next planned tests are:

```text
MARKET_MACRO + Growth group
MARKET_MACRO + Balance Sheet group
MARKET_MACRO + Profitability group
```

and eventually individual additions such as:

```text
MARKET_MACRO + Revenue_Growth_YoY
MARKET_MACRO + Cash_to_Debt
MARKET_MACRO + FCF_Growth_YoY
```

This will help identify which fundamental variables add signal and which introduce noise.

---

## Sector Rotation Research

Sector-leadership analysis showed that model performance varies when market leadership rotates across sectors.

The next planned feature family includes sector-relative variables such as:

```text
Sector_Return_20D
Sector_Return_63D
Sector_Return_126D
Sector_Rank_63D
Sector_Momentum_Acceleration
Stock_vs_Sector_Return_20D
Stock_vs_Sector_Return_63D
Sector_Breadth
Sector_Volatility_20D
```

These features are intended to let the model detect shifts such as Technology leadership weakening while Energy or Industrials strengthen.

---

## Current Ranking Pipeline

The project can train the MARKET_MACRO model on all observations with known 252-day outcomes and rank the latest S&P 500 snapshot by classifier score.

The classifier output should primarily be interpreted as a **ranking score** rather than a perfectly calibrated probability.

A score such as:

```text
0.37
```

should not be interpreted as an expected +37% return.

The ranking pipeline selects approximately the top 10% of the current universe as research candidates.

---

## Important Limitations

The current project still has several important limitations:

- **Survivorship bias:** historical data is currently built using today's S&P 500 membership. This becomes especially important for the 2006-history experiment.
- **SEC fiscal-period approximation:** some growth features still use approximate row-based comparisons instead of true comparable fiscal periods.
- **Uneven fundamental coverage:** SEC history is thinner for some companies and earlier years.
- **Overlapping 252-day labels:** daily and monthly ranking results should not be interpreted directly as independent portfolio returns.
- **Regime sensitivity:** ranking performance still varies materially across market environments.
- **Probability calibration:** classifier scores are currently more appropriate for ranking than as literal probabilities.

---

## Project Structure

```text
src/stock_ml_forecast/
  market/
    market_data.py
    features.py
  macro/
    panel_features.py
  fundamentals/
    sec_data.py
  data/
    cache.py
    universe.py
    panel.py
  modeling/
    panel_targets.py
    panel_dataset.py
    panel_models.py
  evaluation/
    panel_backtest.py
    panel_walk_forward.py
    panel_ablation.py
    panel_importance.py

experiments/
  fold_regime_analysis.py
  sector_leadership_analysis.py
  current_ranking.py

long_history_model/
  build_long_history_panel.py
  train_long_history_model.py
  walk_forward_long_history.py
  walk_forward_all_features_long_history.py
  results/
```

---

## Current Status

```text
Normal S&P 500 panel                     complete
Long-history 2006 panel                 complete
24-feature MARKET_MACRO model           complete
33-feature full model                   complete
252D targets                            complete
Purged chronological validation         complete
Monthly ranking evaluation              complete
Walk-forward validation                 complete
Feature importance                      complete
Feature-group ablation                  complete
Long-history comparison                 complete
Long-history 33-feature comparison      complete
Current benchmark                       MARKET_MACRO 2006+
Average benchmark Precision @ 10%       26.44%
Sector-relative feature experiment      planned
Fundamental subgroup experiment         planned
Historical S&P membership               planned
```

---

## Research Goal

The long-term goal is to build a robust cross-sectional ranking system that can identify non-obvious companies with the potential to become future high-growth or high-outperformance leaders.

The project is intentionally structured as an iterative research process: every new feature family, data source, or modeling change must outperform a fixed benchmark under the same walk-forward validation framework before it is promoted into the main model.
