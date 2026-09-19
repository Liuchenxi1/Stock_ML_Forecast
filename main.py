from stock_ml_forecast.data.universe import (
    get_sp500_universe,
)

from stock_ml_forecast.data.panel import (
    build_sp500_panel,
    load_sp500_panel,
)

from stock_ml_forecast.paths import (
    SP500_PANEL_PATH,
)

from stock_ml_forecast.macro.panel_features import (
    get_model_feature_frame,
    print_feature_coverage,
)

from stock_ml_forecast.modeling.panel_targets import (
    add_252d_targets,
    print_target_summary
)

from stock_ml_forecast.modeling.panel_dataset import (
    prepare_panel_training_data,
    purged_chronological_split,
    print_split_summary,
)


from stock_ml_forecast.modeling.panel_models import (
    train_panel_models,
    evaluate_classifier,
    evaluate_regressor,
    evaluate_mean_baseline,
    evaluate_top_decile_ranking,
)

from stock_ml_forecast.evaluation.panel_backtest import (
    run_monthly_ranking_backtest,
    print_monthly_backtest,
)

from stock_ml_forecast.evaluation.panel_walk_forward import (
    run_walk_forward_validation,
    print_walk_forward_summary,
)

from stock_ml_forecast.evaluation.panel_importance import (
    calculate_feature_importance,
    print_feature_importance,
)

from stock_ml_forecast.evaluation.panel_ablation import (
    run_feature_ablation,
    print_ablation_summary,
)

SEC_USER_AGENT = (
    "Stock ML Forecast shinnkiryu@gmail.com"
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

panel = add_252d_targets(
    panel
)

print_target_summary(
    panel
)

training_data = (
    prepare_panel_training_data(
        panel
    )
)

split = (
    purged_chronological_split(
        training_data,
        train_fraction=0.70,
        validation_fraction=0.15,
        purge_horizon=252,
    )
)

print_split_summary(
    split
)

models = train_panel_models(
    X_train=split.X_train,
    y_top10_train=split.y_top10_train,
    y_return_train=split.y_return_train,
)


evaluate_classifier(
    model=models.classifier,
    X=split.X_validation,
    y=split.y_top10_validation,
    label="Validation",
)

evaluate_classifier(
    model=models.classifier,
    X=split.X_test,
    y=split.y_top10_test,
    label="Test",
)


evaluate_mean_baseline(
    y_train=split.y_return_train,
    y=split.y_return_validation,
    label="Validation",
)

evaluate_regressor(
    model=models.regressor,
    X=split.X_validation,
    y=split.y_return_validation,
    label="Validation",
)


evaluate_mean_baseline(
    y_train=split.y_return_train,
    y=split.y_return_test,
    label="Test",
)

evaluate_regressor(
    model=models.regressor,
    X=split.X_test,
    y=split.y_return_test,
    label="Test",
)

print("=" * 60)

evaluate_top_decile_ranking(
    classifier=models.classifier,
    regressor=models.regressor,
    X=split.X_validation,
    y_top10=split.y_top10_validation,
    y_return=split.y_return_validation,
    label="Validation",
)

evaluate_top_decile_ranking(
    classifier=models.classifier,
    regressor=models.regressor,
    X=split.X_test,
    y_top10=split.y_top10_test,
    y_return=split.y_return_test,
    label="Test",
)

# ============================================================
# Monthly ranking backtest
# ============================================================

validation_backtest = (
    run_monthly_ranking_backtest(
        classifier=models.classifier,
        X=split.X_validation,
        y_top10=split.y_top10_validation,
        y_return=split.y_return_validation,
        label="Validation",
    )
)

print_monthly_backtest(
    validation_backtest,
    label="Validation",
)


test_backtest = (
    run_monthly_ranking_backtest(
        classifier=models.classifier,
        X=split.X_test,
        y_top10=split.y_top10_test,
        y_return=split.y_return_test,
        label="Test",
    )
)

print_monthly_backtest(
    test_backtest,
    label="Test",
)

walk_forward = (
    run_walk_forward_validation(
        data=training_data,
        final_test_start="2024-03-26",
        purge_horizon=252,
        validation_days=126,
        step_days=126,
        min_train_days=756,
    )
)

print_walk_forward_summary(
    walk_forward
)

importance = (
    calculate_feature_importance(
        model=models.classifier,
        X=split.X_validation,
        y=split.y_top10_validation,
    )
)

print_feature_importance(
    importance
)

ablation = run_feature_ablation(
    split=split,
)

print_ablation_summary(
    ablation
)