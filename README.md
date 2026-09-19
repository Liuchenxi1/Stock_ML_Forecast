# Stock_ML_Forecast

A machine-learning research project for ranking S&P 500 companies by their probability of becoming future outperformers over the next approximately 252 trading days.

The goal is not to predict whether a stock simply goes up or down. The model tries to identify which companies are most likely to rank in the **top 10% of future performers** relative to the rest of the S&P 500.

> This project is experimental quantitative research. Backtest results are not guaranteed future returns and should not be interpreted as investment advice.

---

## Data Used

The current dataset contains roughly:

```text
503 companies
~1.35 million Date/Ticker observations
33 model features
```

The model combines three main feature groups.

### Market / Technical

Examples:

```text
Stock_Return_1D
Price_to_MA20 / MA50 / MA200
RSI_14
MACD_Pct
Volatility_20D
Volume_Ratio_20D
Return_20D
Return_63D
Excess_Return_vs_SPY_63D
```

### Macro / Market Regime

Examples:

```text
SPY_Return_20D
SPY_Return_63D
SPY_Volatility_20D
VIX_Change_20D / 63D
Treasury_3M_Change_20D / 63D
Treasury_10Y_Change_20D / 63D
Yield_Curve_10Y_3M
Yield_Curve_Change_20D / 63D
```

These features help describe the market environment, including volatility, interest-rate movement, and changes in the yield curve.

### Fundamentals

SEC Company Facts are used to create features such as:

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

---

## Prediction Target

For each company and date, the model looks approximately 252 trading days into the future.

The main classification target is:

```text
Top_10pct_252D
```

A positive label means the stock finished in approximately the top 10% of S&P 500 companies based on its future 252-day return.

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

Because only about 10% of observations are positive, normal classification accuracy is not very useful. The project focuses instead on ranking metrics such as **Precision @ 10%**.

---

## Model Training

The current baseline models are:

```text
HistGradientBoostingClassifier
HistGradientBoostingRegressor
```

The classifier is the more important model because the main goal is to rank companies by their probability of becoming future top-decile performers.

Financial data is **not randomly shuffled**. Training always happens on older data and evaluation happens on later unseen data.

A 252-trading-day purge gap is placed between training and validation periods because the target itself looks 252 trading days into the future. This prevents training labels from overlapping with the validation period.

Example:

```text
Fold 3
Train:       2016-01-04 -> 2020-01-03
Purge:       ~252 trading days
Validation:  2021-01-05 -> 2021-07-06
```

This means the model learns only from information available before the validation period and then predicts a later period it has never seen.

---

## Walk-Forward Validation

The project uses expanding-window walk-forward validation.

For each fold:

1. Train on historical data.
2. Leave a 252-trading-day purge gap.
3. Validate on the next approximately six months.
4. Expand the training history.
5. Repeat.

Current ML Precision @ 10% by fold:

```text
Fold 1: 17.49%
Fold 2: 20.12%
Fold 3:  8.75%
Fold 4:  4.05%
Fold 5: 20.00%
Fold 6: 29.71%
Fold 7: 27.14%
Fold 8: 24.00%
```

Random top-10% selection would produce approximately 10% precision.

Folds 3 and 4 were unusually weak. These periods are consistent with a major market-regime transition. Because the target looks one year forward, predictions made during 2021 were judged partly on performance during the very different market environment of 2022.

Adding additional macro-regime features did not materially fix these two folds, suggesting the problem is not simply missing VIX or interest-rate information. Some future market shocks cannot be inferred reliably from trailing data available at the prediction date.

---

## Monthly Ranking Backtest

Daily 252-day targets overlap heavily, so the project also evaluates one prediction snapshot per month.

The ML model is compared against simple 63-day momentum.

Recent test-period results for the full model showed:

```text
ML Precision @ 10%       ~25-26%
Momentum Precision       ~20%
```

This means the ML-selected top 10% contained roughly 2.5 times the concentration of actual future top-decile stocks compared with random selection.

These are **ranking diagnostics**, not portfolio CAGR. The 252-day outcome windows still overlap across monthly snapshots.

---

## Feature Importance

Recent permutation importance showed the strongest features included:

```text
Volatility_20D
Cash_to_Debt
Revenue_Growth_YoY
FCF_Growth_YoY
SPY_Volatility_20D
Excess_Return_vs_SPY_63D
Net_Income_Growth_YoY
ROA
Debt_to_Assets
Debt_to_Equity
```

This shows that both market behavior and company fundamentals contain predictive information.

Permutation importance measures how much model performance falls when a feature is shuffled. It does not show whether a high or low value is favorable.

---

## Feature-Group Ablation Results

Three versions of the model were compared:

```text
ALL             33 features
MARKET_MACRO    24 features
FUNDAMENTALS     9 features
```

Results:

| Model | Validation Precision @ 10% | Test Precision @ 10% | Test Return Lift | Test Median Selected Return |
|---|---:|---:|---:|---:|
| ALL | 26.00% | 26.44% | 51.43% | -0.57% |
| MARKET + MACRO | **31.00%** | **30.78%** | **61.44%** | **9.83%** |
| FUNDAMENTALS | 22.67% | 20.11% | 33.49% | -4.79% |

The strongest current model is **MARKET + MACRO**.

Fundamentals still contain useful signal by themselves, but the current fundamental features reduce performance when combined with the market/macro model.

One likely reason is data quality. Some fundamental growth features currently use an approximate 252-trading-row lag rather than true comparable SEC fiscal periods.

For now, MARKET + MACRO is the benchmark model, while the fundamental dataset will be improved separately before being added back.

---

## Current Limitations

The main limitations are:

- **Survivorship bias:** the historical panel currently uses today's S&P 500 constituents.
- **Approximate SEC growth calculations:** true fiscal-quarter and fiscal-year comparisons still need to be implemented.
- **Overlapping 252-day targets:** ranking results should not be interpreted directly as independent portfolio returns.
- **Regime sensitivity:** model performance varies substantially across different market environments.

---

## Next Steps

The next research steps are:

1. [ ] Run walk-forward validation using the MARKET + MACRO feature set only.
2. [ ] Improve SEC fundamental features using true comparable fiscal periods.
3. [ ] Add better fundamental acceleration features such as revenue acceleration, margin expansion, debt growth, and cash-flow growth.
4. [ ] Re-run the ablation test after improving the SEC data.
5. [ ] Later reduce survivorship bias using historical S&P 500 membership.
6. [ ] Build the final current-universe ranking pipeline.

---

## Current Status

```text
S&P 500 panel                complete
33-feature dataset           complete
252D targets                 complete
Purged chronological split   complete
Baseline ML models           complete
Monthly ranking evaluation   complete
Walk-forward validation      complete
Feature importance           complete
Feature-group ablation       complete
Current champion             MARKET + MACRO
Final live ranking           not yet implemented
```
