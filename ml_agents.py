"""
ml_agents.py — Five CrewAI agents that replace the sklearn Pipeline.

Flow (sequential):
  1. DataAnalyst          → inspects data, produces schema report
  2. PreprocessingEngineer→ writes + runs pandas preprocessing code
  3. MLEngineer           → trains multiple models without sklearn Pipeline
  4. EvaluatorAgent       → computes metrics, generates 6-panel chart + importance
  5. CodeGenerator        → writes clean reproducible script + final review

Results are written to STATE (ml_state.py) via tools (ml_tools.py).
Gradio reads STATE to render the same outputs as the original app.
"""
import os
import numpy as np
import pandas as pd
from crewai import Agent, Crew, Process, Task, LLM

from ml_state import STATE
from ml_tools import (
    DataInspectionTool, CodeRunnerTool, ChartTool,
    FileSaverTool, ComparisonBuilderTool, make_tools,
)


# ══════════════════════════════════════════════════════════════════════════════
# Helper — build the LLM object
# ══════════════════════════════════════════════════════════════════════════════
def _make_llm(api_key: str, model: str) -> LLM:
    return LLM(model=model, api_key=api_key.strip())


# ══════════════════════════════════════════════════════════════════════════════
# 1. DATA ANALYST AGENT
# ══════════════════════════════════════════════════════════════════════════════
def data_analyst_agent(llm: LLM) -> Agent:
    return Agent(
        role="Senior Data Analyst",
        goal=(
            "Produce a complete schema report for the uploaded dataset: "
            "identify column types (numeric, categorical, datetime, text, ID-like), "
            "missing value patterns, target distribution and skewness, "
            "and recommend preprocessing steps for each column type. "
            "Output a structured report that the Preprocessing agent can follow exactly."
        ),
        backstory=(
            "You are an expert data analyst who has worked on hundreds of tabular ML projects. "
            "You are meticulous about data quality, data types, and leakage prevention. "
            "You always call DataInspectionTool first to see what you are working with, "
            "then produce a precise, actionable schema report. "
            "You never guess — you let the data speak. "
            "You flag ID-like columns (high cardinality names ending in _id, row counters), "
            "datetime columns, free-text columns (long strings, many words), "
            "and columns that are mostly missing. "
            "You explicitly state whether log-transforming the target is advisable."
        ),
        tools=[DataInspectionTool()],
        llm=llm,
        allow_delegation=False,
        verbose=True,
    )


def data_analyst_task(agent: Agent, target_col: str, eval_source: str) -> Task:
    return Task(
        description=(
            f"Analyse the uploaded dataset for a regression task.  "
            f"Target column: '{target_col}'.  "
            f"Evaluation source: {eval_source}.\n\n"
            "Steps:\n"
            f"1. Call DataInspectionTool with target_col='{target_col}'.\n"
            "2. Based on the report, categorise every feature column as one of:\n"
            "   NUMERIC / LOW_CARD_CAT (<=30 unique) / HIGH_CARD_CAT (>30 unique) / "
            "   DATETIME / TEXT / ID_LIKE (drop) / CONSTANT (drop) / ALL_MISSING (drop).\n"
            "3. State whether to log-transform the target (skew > 1.5 and all values >= 0).\n"
            "4. Output a JSON-like schema block inside triple backticks with keys:\n"
            "   numeric_cols, low_card_cols, high_card_cols, datetime_cols, "
            "   text_cols, drop_cols, log_target (bool).\n"
            "5. Write a brief natural-language preprocessing recommendation for each type."
        ),
        expected_output=(
            "A structured schema report with column classifications and a JSON block "
            "containing the exact column lists. Preprocessing recommendations per type."
        ),
        agent=agent,
    )


# ══════════════════════════════════════════════════════════════════════════════
# 2. PREPROCESSING ENGINEER AGENT
# ══════════════════════════════════════════════════════════════════════════════
def preprocessing_agent(llm: LLM) -> Agent:
    return Agent(
        role="ML Preprocessing Engineer",
        goal=(
            "Write and execute complete Python preprocessing code using pandas and numpy. "
            "Do NOT use sklearn Pipeline, ColumnTransformer, or any sklearn preprocessor. "
            "Produce result_X_train, result_X_eval, result_feature_names, "
            "result_preprocess_fn, and result_preprocessing_summary."
        ),
        backstory=(
            "You are a hands-on ML engineer who prefers explicit, readable preprocessing code "
            "over opaque sklearn pipelines. You write clean pandas transformations that "
            "any engineer can read and modify. "
            "Your golden rule: ALL statistics (medians, frequency maps, encoder mappings) "
            "are computed on training data only and then APPLIED to eval data. "
            "You never fit on eval data — this is leakage. "
            "You handle: median imputation for numerics, frequency encoding for high-cardinality "
            "categoricals, one-hot encoding for low-cardinality categoricals, calendar features "
            "for datetimes, character/word statistics for free text, and log1p for skewed targets. "
            "You always write a result_preprocess_fn callable so new rows can be predicted later."
        ),
        tools=[CodeRunnerTool()],
        llm=llm,
        allow_delegation=False,
        verbose=True,
    )


def preprocessing_task(agent: Agent, target_col: str, test_size: float,
                        shuffle: bool, random_state: int, has_eval: bool) -> Task:
    split_instruction = (
        "Evaluation data is a separate file already in eval_df. "
        "Use train_df as training, eval_df as evaluation."
        if has_eval else
        f"Split train_df with test_size={test_size}, shuffle={shuffle}, "
        f"random_state={random_state} to get training and evaluation sets."
    )
    return Task(
        description=(
            f"Write and execute preprocessing code for target='{target_col}'.  "
            f"Data split: {split_instruction}\n\n"
            "Your code MUST:\n"
            "1. Extract y_train and y_eval (target Series). Apply log1p if the Data Analyst "
            "   recommended log_target=True, but store the flag so predictions use expm1.\n"
            "2. Identify and DROP: ID-like columns, all-missing columns, constant columns.\n"
            "3. For NUMERIC columns: fill NaN with column median (from training), "
            "   add a __missing binary flag, then divide by (IQR + 1e-8) for robust scaling. "
            "   ALL statistics from training data only.\n"
            "4. For LOW_CARD_CAT (<=30 unique): fill NaN with 'Unknown', then one-hot encode. "
            "   Store the known categories so unseen values map to zeros at predict time.\n"
            "5. For HIGH_CARD_CAT (>30 unique): fill NaN with 'Unknown', then replace with "
            "   frequency in training data (proportion 0-1). Unknown maps to 0.\n"
            "6. For DATETIME: parse with pd.to_datetime, extract year/month/day/dayofweek/"
            "   quarter/is_month_end/elapsed_days/missing flag.\n"
            "7. For TEXT: extract char_len, word_count, digit_count, upper_count, punct_count.\n"
            "8. Horizontally concatenate all processed parts into result_X_train and result_X_eval.\n"
            "9. Save: result_X_train, result_X_eval, result_feature_names.\n"
            "10. Write result_preprocess_fn — a closure/function that takes a single raw DataFrame "
            "    row (with same columns as train_df minus target) and returns a 2D numpy array "
            "    ready to pass to model.predict(). It must apply identical transformations.\n"
            "11. Set result_preprocessing_summary to a multi-line string describing what was done.\n"
            "Use CodeRunnerTool to execute. If there is an error, fix and re-run."
        ),
        expected_output=(
            "Confirmation that result_X_train, result_X_eval, result_feature_names, "
            "result_preprocess_fn, and result_preprocessing_summary are saved to state. "
            "Print shapes of X_train and X_eval."
        ),
        agent=agent,
    )


# ══════════════════════════════════════════════════════════════════════════════
# 3. ML ENGINEER AGENT
# ══════════════════════════════════════════════════════════════════════════════
def ml_engineer_agent(llm: LLM) -> Agent:
    return Agent(
        role="Senior ML Engineer",
        goal=(
            "Train multiple regression models on the preprocessed features. "
            "Do NOT wrap models in sklearn Pipeline. "
            "Train each model directly on X_train / y_train. "
            "For each model save result_model_NAME, result_pred_NAME, result_metrics_NAME."
        ),
        backstory=(
            "You are an expert ML engineer who selects and trains models efficiently. "
            "You always train at least 3 diverse models: one tree-based ensemble, "
            "one gradient boosting model, and one linear baseline. "
            "You use XGBRegressor if available, LGBMRegressor if available, "
            "and RandomForestRegressor as a reliable fallback. "
            "You compute MAE, RMSE, R², and MAPE for every model. "
            "You name your models clearly: 'xgboost', 'lightgbm', 'random_forest', "
            "'ridge', etc. — lowercase with underscores. "
            "You never train on evaluation data. You check X_train is not None before proceeding."
        ),
        tools=[CodeRunnerTool(), ComparisonBuilderTool()],
        llm=llm,
        allow_delegation=False,
        verbose=True,
    )


def ml_engineer_task(agent: Agent, preferred_models: str) -> Task:
    return Task(
        description=(
            f"Train regression models on the preprocessed X_train / y_train.  "
            f"User preference: {preferred_models}.\n\n"
            "Steps:\n"
            "1. Verify X_train and y_train are available (not None).\n"
            "2. Choose 3-4 models based on preference and data size.  "
            "   If data has >10k rows prefer XGBoost + LightGBM + RandomForest.  "
            "   Always include at least one linear model (Ridge) as baseline.\n"
            "3. For each model, write code that:\n"
            "   a. Instantiates and fits the model on X_train, y_train.\n"
            "   b. Predicts on X_eval.\n"
            "   c. Computes MAE=mean_absolute_error(y_eval,pred), "
            "      MSE=mean_squared_error(y_eval,pred), "
            "      RMSE=np.sqrt(MSE), R2=r2_score(y_eval,pred), "
            "      MAPE=mean_absolute_percentage_error(y_eval,pred)*100.\n"
            "   d. Sets result_model_NAME, result_pred_NAME, result_metrics_NAME.\n"
            "4. Execute via CodeRunnerTool.  Fix any errors and retry.\n"
            "5. After all models trained, call ComparisonBuilderTool.\n"
            "6. Print a summary table of all models and their RMSE."
        ),
        expected_output=(
            "All models trained, saved to state via result_model_* variables. "
            "Comparison table built. Best model by RMSE identified."
        ),
        agent=agent,
    )


# ══════════════════════════════════════════════════════════════════════════════
# 4. EVALUATOR AGENT
# ══════════════════════════════════════════════════════════════════════════════
def evaluator_agent(llm: LLM) -> Agent:
    return Agent(
        role="ML Evaluator and Visualisation Specialist",
        goal=(
            "Generate the performance dashboard (6 charts) and feature importance charts "
            "for the best model. Save PNG bytes to state under standard keys."
        ),
        backstory=(
            "You are a specialist in ML evaluation and data visualisation. "
            "You create clear, publication-quality charts using matplotlib on a dark theme. "
            "You always generate: actual-vs-predicted scatter, residuals-vs-predicted, "
            "residual histogram, error-% histogram, cumulative error CDF, and metric bar chart. "
            "You also plot feature importances (model.feature_importances_ if available, "
            "else model.coef_). "
            "You use dark background #0d0d1a, purple #a78bfa for primary series, "
            "cyan #06b6d4 for reference lines. "
            "You save charts under keys: 'performance', 'importance', 'comparison'."
        ),
        tools=[ChartTool(), ComparisonBuilderTool()],
        llm=llm,
        allow_delegation=False,
        verbose=True,
    )


def evaluator_task(agent: Agent) -> Task:
    return Task(
        description=(
            "Generate evaluation charts for all trained models.\n\n"
            "Step 1 — Performance dashboard (chart_name='performance'):\n"
            "Write matplotlib code that creates a 2×3 figure with:\n"
            "  [0,0] Actual vs Predicted scatter (best model by RMSE)\n"
            "  [0,1] Residuals vs Predicted\n"
            "  [0,2] Residual histogram\n"
            "  [1,0] Abs-error-% histogram\n"
            "  [1,1] Cumulative error CDF with 10%/25%/50% threshold markers\n"
            "  [1,2] Horizontal bar chart of MAE/RMSE/R² for best model\n"
            "End code with: fig = plt.gcf()\n"
            "Call ChartTool with chart_name='performance' and that code.\n\n"
            "Step 2 — Feature importances (chart_name='importance'):\n"
            "Write code that gets model.feature_importances_ (tree models) or "
            "abs(model.coef_) (linear models) and plots top-15 as horizontal bar chart.\n"
            "Call ChartTool with chart_name='importance'.\n\n"
            "Step 3 — Cross-model comparison (chart_name='comparison'), only if >1 model:\n"
            "Bar chart of RMSE for each model side by side.\n"
            "Call ChartTool with chart_name='comparison'.\n\n"
            "Use dark theme: facecolor='#0d0d1a', bar color='#a78bfa'.\n"
            "All charts must end with: fig = plt.gcf()"
        ),
        expected_output=(
            "Charts 'performance', 'importance', and 'comparison' saved to state."
        ),
        agent=agent,
    )


# ══════════════════════════════════════════════════════════════════════════════
# 5. CODE GENERATOR AGENT
# ══════════════════════════════════════════════════════════════════════════════
def code_generator_agent(llm: LLM) -> Agent:
    return Agent(
        role="ML Code Synthesis and Review Agent",
        goal=(
            "Write a clean, self-contained, runnable Python script that exactly reproduces "
            "the preprocessing and training performed. Save it as result_generated_code. "
            "Then produce a structured review of the entire pipeline."
        ),
        backstory=(
            "You are a senior ML engineer who values reproducibility above all. "
            "You write Python scripts that anyone can download and run immediately. "
            "You document every design decision as a comment. "
            "You review the pipeline objectively: note strengths, risks, and one concrete "
            "next improvement. Your review uses four sections: "
            "### Data Analysis Agent, ### Preprocessing Agent, ### ML Engineer Agent, ### Review."
        ),
        tools=[CodeRunnerTool(), FileSaverTool()],
        llm=llm,
        allow_delegation=False,
        verbose=True,
    )


def code_generator_task(agent: Agent, target_col: str,
                        train_path: str, eval_path: str,
                        id_col: str) -> Task:
    return Task(
        description=(
            f"Write a reproducible Python script for target='{target_col}', "
            f"training file='{train_path}', eval='{eval_path}'.\n\n"
            "The script must:\n"
            "1. Import only: pandas, numpy, matplotlib, and whatever ML library was used "
            "   (xgboost, lightgbm, sklearn models — but NOT sklearn Pipeline).\n"
            "2. Load the data files.\n"
            "3. Implement ALL preprocessing steps exactly as executed:\n"
            "   - Same column drops, same imputation values, same encoding maps.\n"
            "   - All fit statistics hard-coded (medians, frequency maps, category lists).\n"
            "4. Train the best model with the same hyperparameters.\n"
            "5. Evaluate and print MAE, RMSE, R².\n"
            "6. Save the model with joblib.\n"
            f"7. If a test file exists, predict and write submission.csv "
            f"   {'with column ' + id_col if id_col != '(none)' else 'with row index'}.\n\n"
            "Save this script as result_generated_code in a CodeRunnerTool call "
            "(just assign the string, don't actually run the generated script).\n"
            "Then call FileSaverTool to write model + code + zip to disk.\n\n"
            "Finally, write a structured pipeline review covering:\n"
            "### Data Analysis Agent — what was discovered\n"
            "### Preprocessing Agent — what was done and why (leakage safeguards)\n"
            "### ML Engineer Agent — best model, metrics, comparison\n"
            "### Review — one concrete next improvement\n"
            "Store the review text in result_review_text (use CodeRunnerTool to set it in STATE)."
        ),
        expected_output=(
            "Reproducible Python script saved as generated_code in state. "
            "Files saved (model, code, zip). Structured review text written."
        ),
        agent=agent,
    )


# ══════════════════════════════════════════════════════════════════════════════
# CREW RUNNER — called by Gradio's on_run()
# ══════════════════════════════════════════════════════════════════════════════
def run_agent_crew(
    api_key: str,
    model_name: str,
    target_col: str,
    preferred_models: str,
    train_path: str,
    eval_path: str,
    id_col: str,
    test_size: float,
    shuffle: bool,
    random_state: int,
    has_eval: bool,
    has_test: bool,
) -> str:
    """
    Orchestrate all five agents sequentially.
    Returns a log string.  All actual results are in STATE.
    """
    STATE.reset_run()
    STATE.set("target_col", target_col)
    eval_source = "Validation file" if has_eval else \
                  f"Train/eval split ({int((1-test_size)*100)}/{int(test_size*100)})"
    STATE.set("eval_source", eval_source)
    STATE.log("Crew starting — five agents will run sequentially")

    try:
        llm = _make_llm(api_key, model_name)
    except Exception as exc:
        STATE.log(f"LLM initialisation failed: {exc}")
        return STATE.get_log()

    # ── Build agents ────────────────────────────────────────────────────────
    a1 = data_analyst_agent(llm)
    a2 = preprocessing_agent(llm)
    a3 = ml_engineer_agent(llm)
    a4 = evaluator_agent(llm)
    a5 = code_generator_agent(llm)

    # ── Build tasks ─────────────────────────────────────────────────────────
    t1 = data_analyst_task(a1, target_col, eval_source)
    t2 = preprocessing_task(a2, target_col, test_size, shuffle,
                             random_state, has_eval)
    t3 = ml_engineer_task(a3, preferred_models)
    t4 = evaluator_task(a4)
    t5 = code_generator_task(a5, target_col, train_path,
                              eval_path or "", id_col)

    # Give downstream tasks context from upstream
    t2.context = [t1]
    t3.context = [t1, t2]
    t4.context = [t3]
    t5.context = [t1, t2, t3, t4]

    # ── Assemble and run crew ───────────────────────────────────────────────
    crew = Crew(
        agents=[a1, a2, a3, a4, a5],
        tasks=[t1, t2, t3, t4, t5],
        process=Process.sequential,
        verbose=True,
    )

    STATE.log("Crew kickoff — this may take several minutes")
    try:
        result = crew.kickoff()
        # If the code generator didn't set review_text via result_*, grab crew output
        if not STATE.get("review_text", "").strip():
            raw = getattr(result, "raw", None) or str(result)
            STATE.set("review_text", raw)
        STATE.log("Crew completed successfully")
    except Exception as exc:
        STATE.log(f"Crew error: {type(exc).__name__}: {exc}")
        STATE.set("error", str(exc))

    # Ensure active_model is set
    if not STATE.get("active_model") and STATE.get("models"):
        cdf = STATE.get("comparison_df")
        if cdf is not None and len(cdf):
            STATE.set("active_model", cdf.iloc[0]["Model"])
        else:
            STATE.set("active_model", next(iter(STATE.get("models", {}))))

    STATE.log("run_agent_crew: done")
    return STATE.get_log()
