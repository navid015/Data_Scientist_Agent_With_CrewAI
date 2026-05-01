<<<<<<< HEAD
"""
app.py — Gradio UI for the fully agent-based AutoML application.

The UI is preserved from the original project. All ML logic (preprocessing,
training, evaluation, code generation) is now performed by five CrewAI agents
(ml_agents.py) that write results to the shared STATE (ml_state.py).
sklearn is used ONLY for individual model classes — no Pipeline, no ColumnTransformer.
"""
import os
import traceback

import gradio as gr
import pandas as pd

from ml_state import STATE
from helpers import (
    FILE_TYPES, OPENAI_MODELS, MODEL_GROUPS,
    _info_banner, _empty_html,
    load_train, load_valid, load_test,
    render_preprocessing_summary, render_metrics_html,
    render_comparison_html, render_charts_html,
    render_feat_html, render_cross_model_html,
    render_sample_html, render_test_download,
    do_custom_predict, update_sample_row,
    select_active_model, build_custom_inputs_df,
)

# ── CSS ────────────────────────────────────────────────────────────────────────
_css_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "styles.css")
CSS = open(_css_path).read() if os.path.exists(_css_path) else ""

# ══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE FUNCTION  (called by Run Pipeline button)
# ══════════════════════════════════════════════════════════════════════════════

def on_run(
    train_df_state, valid_df_state, test_df_state,
    train_path_state, valid_path_state, test_path_state,
    target_col, model_group,
    test_size, shuffle_split, random_state,
    api_key, openai_model, id_col_choice,
    progress=gr.Progress(),
):
    empty = _empty_html()
    _blank = (empty,)*9 + ("","",None,None,None,None,pd.DataFrame(),"",
               gr.update(choices=[],value=None), gr.update(maximum=0,value=0))

    # ── Validate ───────────────────────────────────────────────────────────
    if train_df_state is None:
        return (_info_banner("Upload a training dataset first.", "error"),) + _blank
    if not target_col:
        return (_info_banner("Select a target column.", "error"),) + _blank
    if not api_key or not api_key.strip():
        return (_info_banner(
            "An OpenAI API key is required — the agents use it to reason and write code.",
            "error"),) + _blank

    # ── Prepare STATE with uploaded data ──────────────────────────────────
    STATE.set("train_df",   train_df_state)
    STATE.set("eval_df",    valid_df_state)
    STATE.set("test_df",    test_df_state)
    STATE.set("target_col", target_col)

    has_eval = valid_df_state is not None
    has_test = test_df_state  is not None

    eval_source = ("Validation file" if has_eval
                   else f"Train/eval split ({int((1-test_size)*100)}/{int(test_size*100)})")
    STATE.set("eval_source", eval_source)

    # ── Import and run crew ────────────────────────────────────────────────
    from ml_agents import run_agent_crew
    progress(0.05, desc="Starting CrewAI agents...")
    STATE.log(f"on_run: target={target_col}, model_group={model_group}")

    try:
        log_text = run_agent_crew(
            api_key        = api_key,
            model_name     = openai_model,
            target_col     = target_col,
            preferred_models = model_group,
            train_path     = train_path_state or "train.csv",
            eval_path      = valid_path_state or "",
            id_col         = id_col_choice or "(none)",
            test_size      = float(test_size),
            shuffle        = bool(shuffle_split),
            random_state   = int(random_state),
            has_eval       = has_eval,
            has_test       = has_test,
        )
    except Exception as exc:
        tb = traceback.format_exc()
        return (
            _info_banner(
                f"Agent crew failed: {type(exc).__name__}: {exc}"
                f"<details><summary>Traceback</summary><pre>{tb}</pre></details>",
                "error"),
        ) + _blank

    progress(0.90, desc="Rendering results...")

    # ── Collect outputs from STATE ─────────────────────────────────────────
    active   = STATE.get("active_model")
    models   = STATE.get("models", {})
    all_names = list(models.keys())

    # Metrics & charts
    preprocess_html  = render_preprocessing_summary()
    metrics_html     = render_metrics_html()
    comparison_html  = render_comparison_html()
    charts_html      = render_charts_html("performance")
    feat_html        = render_feat_html()
    cross_html       = render_cross_model_html()
    sample_html      = render_sample_html(0)

    # Test predictions
    test_html, submission_path = render_test_download(id_col_choice or "(none)")

    # Files
    code_file_path  = STATE.get("code_file_path")
    model_file_path = STATE.get("model_file_path")
    zip_path        = STATE.get("zip_path")

    # Code + review
    generated_code  = STATE.get("generated_code", "")
    review_text     = STATE.get("review_text", "")

    # Custom-predict prefill
    custom_default  = build_custom_inputs_df()

    # Eval set size for slider
    X_eval = STATE.get("X_eval")
    n_eval  = (X_eval.shape[0] if hasattr(X_eval, "shape") else len(X_eval)) if X_eval is not None else 0

    err = STATE.get("error")
    if err:
        status = _info_banner(
            f"Pipeline completed with errors: {err}<br>{log_text.split(chr(10))[-1]}",
            "warn")
    else:
        best = STATE.get("comparison_df")
        best_name = (best.iloc[0]["Model"] if best is not None and len(best) else active) or "unknown"
        status = _info_banner(
            f"Agent workflow complete — best by RMSE: <b>{best_name}</b>", "ok")

    progress(1.0, desc="Done!")

    return (
        status,                                            # 0  status banner
        preprocess_html,                                   # 1  preprocessing
        metrics_html,                                      # 2  metrics
        comparison_html,                                   # 3  comparison table
        charts_html,                                       # 4  performance charts
        feat_html,                                         # 5  feature importances
        sample_html,                                       # 6  sample prediction
        test_html,                                         # 7  test download HTML
        cross_html,                                        # 8  cross-model chart
        generated_code,                                    # 9  code text
        review_text,                                       # 10 review text
        code_file_path,                                    # 11 code file
        submission_path,                                   # 12 submission csv
        model_file_path,                                   # 13 model file
        zip_path,                                          # 14 zip bundle
        custom_default,                                    # 15 custom inputs df
        log_text,                                          # 16 agent log
        gr.update(choices=all_names, value=active),        # 17 model dropdown
        gr.update(maximum=max(n_eval-1,0), value=0),       # 18 row slider
    )


# ══════════════════════════════════════════════════════════════════════════════
# GRADIO UI
# ══════════════════════════════════════════════════════════════════════════════

with gr.Blocks(css=CSS, title="CrewAI AutoML — Agent Based", theme=gr.themes.Base()) as demo:

    # ── Invisible state variables ──────────────────────────────────────────
    train_df_state   = gr.State(None)
    valid_df_state   = gr.State(None)
    test_df_state    = gr.State(None)
    train_path_state = gr.State(None)
    valid_path_state = gr.State(None)
    test_path_state  = gr.State(None)

    # ── Header ────────────────────────────────────────────────────────────
    gr.HTML("""
    <div class="header-band">
      <p class="header-title">⚡ Agent-Based AutoML  —  Powered by CrewAI</p>
      <p class="header-sub">
        Five specialised AI agents collaborate to inspect your data, write preprocessing
        code, train multiple models, evaluate results, and generate a reproducible script
        — no scikit-learn Pipeline, no hardcoded rules, fully LLM-driven.
=======
import os
import gradio as gr
import pandas as pd
from helpers import *  # noqa: F403, F401

# Load CSS from external file
_css_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "styles.css")
with open(_css_path, "r") as _f:
    CSS = _f.read()

# ── UI ──────────────────────────────────────────────────────────────────────────
with gr.Blocks(css=CSS, title="CrewAI AutoML", theme=gr.themes.Base()) as demo:
    pipeline_state = gr.State(None)
    train_df_state = gr.State(None)
    valid_df_state = gr.State(None)
    test_df_state = gr.State(None)
    train_path_state = gr.State(None)
    valid_path_state = gr.State(None)
    test_path_state = gr.State(None)

    gr.HTML("""
    <div class="header-band">
      <p class="header-title">⚡ ML Engineer Agent with CrewAI for End-to-End Regression Modeling</p>
      <p class="header-sub">
        Autonomous AI agents collaborate to preprocess any dataset, train & compare
        multiple models, generate reproducible code, and deliver expert-level review
        — all powered by CrewAI's multi-agent orchestration.
>>>>>>> 6292a482cf1e2f547f6a880fb59fdb116f3f1bc2
      </p>
    </div>
    """)

<<<<<<< HEAD
    status_out = gr.HTML(_info_banner(
        "👋 Upload a training dataset on the <b>Data</b> tab to begin.", "info"))

    # ── Tabs ──────────────────────────────────────────────────────────────
    with gr.Tabs(elem_id="main-tabs"):

        # ══ TAB 1: DATA ═══════════════════════════════════════════════════
        with gr.Tab("📁 Data"):
=======
    # ════════════════════════ Top status banner ════════════════════════
    status_out = gr.HTML(_info_banner(
        "👋 Upload a training dataset on the <b>Data</b> tab to begin.", "info"
    ))

    gr.HTML("""
    <style>
    #main-tabs > div:first-child {
        display:flex !important; width:100% !important; gap:0 !important;
        padding:0 !important; margin:0 !important;
        background:#1a1a2e !important; border-bottom:2px solid #2a2a3e !important;
        border-radius:12px 12px 0 0 !important;
    }
    #main-tabs > div:first-child > button {
        flex:1 1 0% !important; text-align:center !important;
        padding:22px 10px !important; margin:0 !important;
        font-size:19px !important; font-weight:700 !important;
        letter-spacing:0.5px !important;
        color:#64748b !important; background:transparent !important;
        border:none !important; border-bottom:3px solid transparent !important;
        border-radius:0 !important; cursor:pointer !important;
        transition:all .25s ease !important;
        min-width:0 !important; white-space:nowrap !important;
    }
    #main-tabs > div:first-child > button:hover {
        color:#a78bfa !important; background:rgba(167,139,250,.10) !important;
    }
    #main-tabs > div:first-child > button.selected {
        color:#a78bfa !important; background:#13131f !important;
        border-bottom:3px solid #a78bfa !important;
    }
    </style>
    """)

    with gr.Tabs(elem_id="main-tabs") as tabs:

        # ════════════════════════ Tab 1: DATA ════════════════════════
        with gr.Tab("📁 Data"):
            gr.HTML('''
            <div style="display:flex;align-items:center;gap:8px;margin-bottom:14px;">
              <div style="width:6px;height:24px;border-radius:3px;
                          background:linear-gradient(180deg,#a78bfa,#06b6d4);"></div>
              <span style="color:#e2e8f0;font-size:14px;font-weight:600;">
                Upload Your Datasets</span>
            </div>
            ''')
>>>>>>> 6292a482cf1e2f547f6a880fb59fdb116f3f1bc2
            with gr.Row(equal_height=True):
                with gr.Column(scale=1):
                    gr.HTML('<div class="step-badge"><span class="num">1</span> Training Dataset '
                            '<span style="color:#ef4444;margin-left:4px;">required</span></div>')
<<<<<<< HEAD
                    train_file   = gr.File(label="Train CSV / Excel / Parquet / JSON",
                                           file_types=FILE_TYPES, elem_classes=["upload-zone"])
=======
                    train_file = gr.File(
                        label="Train CSV / Excel / Parquet / JSON",
                        file_types=FILE_TYPES,
                        elem_classes=["upload-zone"],
                    )
>>>>>>> 6292a482cf1e2f547f6a880fb59fdb116f3f1bc2
                    train_status = gr.Markdown("*Upload a training file to begin.*")

                with gr.Column(scale=1):
                    gr.HTML('<div class="opt-badge">⚙ Optional — Validation File</div>')
<<<<<<< HEAD
                    valid_file   = gr.File(label="Validation file",
                                           file_types=FILE_TYPES, elem_classes=["upload-zone"])
                    valid_status = gr.Markdown("*No validation file — agents will split training data.*")

                with gr.Column(scale=1):
                    gr.HTML('<div class="opt-badge">⚙ Optional — Test File</div>')
                    test_file   = gr.File(label="Test file (no target needed)",
                                          file_types=FILE_TYPES, elem_classes=["upload-zone"])
                    test_status = gr.Markdown("*No test file uploaded.*")

            reset_btn = gr.Button("🔄 Reset All Uploads", elem_classes=["reset-btn"])
            gr.HTML('<div class="divider"></div>')
            gr.HTML('<div class="section-label">Training Data Preview</div>')
            preview_html = gr.HTML(_empty_html("No dataset loaded yet."))

        # ══ TAB 2: CONFIGURE ══════════════════════════════════════════════
        with gr.Tab("⚙️ Configure"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.HTML('<div class="step-badge"><span class="num">2</span> Target & Model Group</div>')
                    target_dropdown = gr.Dropdown(
                        label="Column to predict", choices=[], interactive=True)
                    model_group_dd = gr.Dropdown(
                        label="Model group for the ML Engineer agent",
                        choices=MODEL_GROUPS,
                        value=MODEL_GROUPS[0],
                        interactive=True,
                    )
                    gr.Markdown(
                        "*The agent will choose and train models from this group. "
                        "It always includes at least one linear baseline.*"
                    )
                    id_col_dropdown = gr.Dropdown(
                        label="ID column to preserve in submission.csv",
                        choices=["(none)"], value="(none)", interactive=True)

                with gr.Column(scale=1):
                    gr.HTML('<div class="step-badge"><span class="num">3</span> '
                            'Split Controls</div>')
                    test_size     = gr.Slider(label="Eval split size",
                                              minimum=0.10, maximum=0.40,
                                              value=0.20, step=0.05)
                    shuffle_split = gr.Checkbox(label="Shuffle train/eval split", value=True)
                    random_state  = gr.Number(label="Random state", value=42, precision=0)

                    gr.HTML('<div class="step-badge" style="margin-top:14px;">'
                            '<span class="num">4</span> CrewAI / LLM Settings</div>')
                    api_key_input = gr.Textbox(
                        label="OpenAI API Key (required)",
                        placeholder="sk-…",
                        type="password",
                    )
                    openai_model_dd = gr.Dropdown(
                        label="LLM for agents",
                        choices=OPENAI_MODELS, value="gpt-4o-mini",
                        interactive=True,
                    )
                    gr.Markdown(
                        "**gpt-4o-mini** is fast and cheap for most datasets. "
                        "Use **gpt-4o** for complex schemas or large feature sets."
                    )

            run_btn = gr.Button("🚀 Run Agent Pipeline", elem_classes=["run-btn"])
            log_out = gr.Textbox(
                label="Live Agent Log", interactive=False, lines=12,
                elem_classes=["log-output"],
                placeholder="Agent activity streams here once you press Run…",
            )

        # ══ TAB 3: RESULTS ════════════════════════════════════════════════
        with gr.Tab("📊 Results"):
            gr.HTML("""
            <div style="background:linear-gradient(135deg,#1a0533 0%,#0d1729 100%);
                        border:1px solid #3b1d6e;border-radius:16px;
                        padding:20px 24px 14px;margin-bottom:16px;">
              <span style="color:#f1f5f9;font-size:15px;font-weight:700;">Select Active Model</span>
              <p style="color:#94a3b8;font-size:12px;margin:4px 0 0;">
                Switch between trained models to compare metrics and charts — no retraining needed.
              </p>
            </div>
            """)
            results_model_dd = gr.Dropdown(
                label="", choices=[], value=None, interactive=True,
                elem_classes=["model-selector-dd"])
=======
                    valid_file = gr.File(
                        label="Validation file",
                        file_types=FILE_TYPES,
                        elem_classes=["upload-zone"],
                    )
                    valid_status = gr.Markdown(
                        "*No validation file. The app will split training data.*"
                    )

                with gr.Column(scale=1):
                    gr.HTML('<div class="opt-badge">⚙ Optional — Test File</div>')
                    test_file = gr.File(
                        label="Test file (no target needed)",
                        file_types=FILE_TYPES,
                        elem_classes=["upload-zone"],
                    )
                    test_status = gr.Markdown("*No test file uploaded.*")

            reset_btn = gr.Button(
                "🔄 Reset All Uploads",
                elem_classes=["reset-btn"],
            )

            gr.HTML('<div class="divider"></div>')
            gr.HTML('<div class="section-label">Training Data Preview</div>')
            preview_html = gr.HTML(
                '<p style="color:#475569;font-size:13px;padding:20px;text-align:center;">'
                'No dataset loaded yet.</p>'
            )

        # ════════════════════════ Tab 2: CONFIGURE ════════════════════════
        with gr.Tab("⚙️ Configure"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.HTML('<div class="step-badge"><span class="num">2</span> Target & Models</div>')
                    target_dropdown = gr.Dropdown(
                        label="Column to predict", choices=[], interactive=True,
                    )
                    selected_models = gr.CheckboxGroup(
                        label="Models to train (pick one or more)",
                        choices=MODEL_NAMES,
                        value=[],
                    )
                    gr.Markdown(
                        "*The first model you select becomes the default shown "
                        "in the Results tab. You can switch between trained "
                        "models there without retraining.*"
                    )
                    id_col_dropdown = gr.Dropdown(
                        label="Test-set ID column to preserve in submission.csv",
                        choices=["(none)"], value="(none)", interactive=True,
                    )

                with gr.Column(scale=1):
                    gr.HTML('<div class="step-badge"><span class="num">3</span> '
                            'Split & Preprocessing Controls</div>')
                    test_size = gr.Slider(
                        label="Eval split size when no validation file is uploaded",
                        minimum=0.10, maximum=0.40, value=0.20, step=0.05,
                    )
                    shuffle_split = gr.Checkbox(
                        label="Shuffle train/eval split", value=True,
                    )
                    random_state = gr.Number(
                        label="Random state", value=42, precision=0,
                    )
                    id_threshold = gr.Slider(
                        label="Drop text columns as ID-like if unique-ratio is above",
                        minimum=0.80, maximum=1.00, value=0.95, step=0.01,
                    )

                    gr.HTML('<div class="step-badge" style="margin-top:14px">'
                            '<span class="num">4</span> Advanced Options</div>')
                    enable_cv = gr.Checkbox(
                        label="Compute K-fold CV RMSE", value=False,
                    )
                    cv_folds = gr.Slider(
                        label="CV / tuning folds", minimum=2, maximum=10,
                        value=3, step=1,
                    )
                    enable_tuning = gr.Checkbox(
                        label="Light-tune the best model (RandomizedSearchCV)",
                        value=False,
                    )

                    gr.HTML('<div class="step-badge" style="margin-top:14px">'
                            '<span class="num">5</span> CrewAI Review (optional)</div>')
                    run_review = gr.Checkbox(
                        label="Run CrewAI review after training",
                        value=False,
                    )
                    api_key_input = gr.Textbox(
                        label="OpenAI API Key",
                        placeholder="sk-… or any provider-compatible key",
                        type="password",
                    )
                    openai_model_dropdown = gr.Dropdown(
                        label="OpenAI model for the reviewer",
                        choices=OPENAI_MODELS, value="gpt-4o-mini",
                        interactive=True,
                    )

            run_btn = gr.Button("🚀 Run Pipeline", elem_classes=["run-btn"])
            log_out = gr.Textbox(
                label="Live Run Log", interactive=False,
                lines=10, elem_classes=["log-output"],
                placeholder="Pipeline activity will stream here once you press Run…",
            )

        # ════════════════════════ Tab 3: RESULTS ════════════════════════
        with gr.Tab("📊 Results"):
            gr.HTML('''
            <div style="background:linear-gradient(135deg,#1a0533 0%,#0d1729 100%);
                        border:1px solid #3b1d6e;border-radius:16px;
                        padding:24px 28px 18px;margin-bottom:16px;
                        box-shadow:0 4px 24px rgba(124,58,237,.15);">
              <div style="display:flex;align-items:center;gap:12px;margin-bottom:6px;">
                <div style="width:8px;height:32px;border-radius:4px;
                            background:linear-gradient(180deg,#a78bfa,#06b6d4);"></div>
                <span style="color:#f1f5f9;font-size:16px;font-weight:700;
                             letter-spacing:.3px;">Select Active Model</span>
              </div>
              <p style="color:#94a3b8;font-size:12px;margin:0 0 4px 20px;line-height:1.5;">
                Switch between trained models to compare metrics, charts and
                feature importances — instantly, no retraining.
              </p>
            </div>
            ''')
            results_model_dd = gr.Dropdown(
                label="",
                choices=[], value=None, interactive=True,
                elem_classes=["model-selector-dd"],
            )
>>>>>>> 6292a482cf1e2f547f6a880fb59fdb116f3f1bc2

            gr.HTML('<div class="divider"></div>')
            gr.HTML('<div class="section-label">Model Metrics</div>')
            metrics_out = gr.HTML(_empty_html())

            gr.HTML('<div class="divider"></div>')
            gr.HTML('<div class="section-label">Preprocessing Summary</div>')
            preprocessing_out = gr.HTML(_empty_html())

            gr.HTML('<div class="divider"></div>')
            gr.HTML('<div class="section-label">Model Comparison Table</div>')
            comparison_out = gr.HTML(_empty_html())

            gr.HTML('<div class="divider"></div>')
<<<<<<< HEAD
            gr.HTML('<div class="section-label">Cross-Model Comparison</div>')
            cross_model_out = gr.HTML(_empty_html("Train more than one model to see this chart."))
=======
            gr.HTML('<div class="section-label">Cross-Model Comparison Charts</div>')
            cross_model_out = gr.HTML(_empty_html(
                "Select two or more models in the Configure tab to enable "
                "side-by-side comparison charts."
            ))
>>>>>>> 6292a482cf1e2f547f6a880fb59fdb116f3f1bc2

            gr.HTML('<div class="divider"></div>')
            gr.HTML('<div class="section-label">Performance Charts (active model)</div>')
            charts_out = gr.HTML(_empty_html())

            gr.HTML('<div class="divider"></div>')
<<<<<<< HEAD
            gr.HTML('<div class="section-label">Feature Importances (active model)</div>')
            feat_out = gr.HTML(_empty_html())

        # ══ TAB 4: PREDICT ════════════════════════════════════════════════
        with gr.Tab("🔮 Predict"):
            gr.HTML('<div class="section-label">Sample Prediction (from evaluation set)</div>')
            row_idx_slider = gr.Slider(
                label="Pick a row from the evaluation set",
                minimum=0, maximum=0, value=0, step=1, interactive=True)
=======
            gr.HTML('<div class="section-label">Feature Importances — Impurity vs Permutation (active model)</div>')
            feat_out = gr.HTML(_empty_html())

        # ════════════════════════ Tab 4: PREDICT ════════════════════════
        with gr.Tab("🔮 Predict"):
            gr.HTML('<div class="section-label">Sample Prediction (from evaluation set)</div>')
            with gr.Row():
                row_idx_slider = gr.Slider(
                    label="Pick a row from the evaluation set",
                    minimum=0, maximum=0, value=0, step=1, interactive=True,
                )
>>>>>>> 6292a482cf1e2f547f6a880fb59fdb116f3f1bc2
            sample_out = gr.HTML(_empty_html())

            gr.HTML('<div class="divider"></div>')
            gr.HTML("""
            <div style="background:#0f1729;border:1px solid #1e3a4a;border-radius:14px;
<<<<<<< HEAD
                        padding:16px 20px;margin-bottom:12px;">
              <span style="color:#06b6d4;font-size:11px;font-weight:700;
                           letter-spacing:2px;text-transform:uppercase;">
                Custom Raw-Row Prediction
              </span>
              <p style="color:#64748b;font-size:12px;margin:6px 0 0;">
                Edit raw feature values. The agent-written preprocessing function is applied automatically.
=======
                        padding:18px 20px;margin-bottom:12px;">
              <span style="color:#06b6d4;font-size:11px;font-weight:700;
                           letter-spacing:2px;text-transform:uppercase;">
                🔬 Custom Raw Row Prediction
              </span>
              <p style="color:#64748b;font-size:12px;margin:6px 0 0;">
                Edit raw feature values. The same fitted preprocessing pipeline
                is applied automatically.
>>>>>>> 6292a482cf1e2f547f6a880fb59fdb116f3f1bc2
              </p>
            </div>
            """)
            with gr.Row():
                with gr.Column(scale=2):
                    custom_inputs = gr.Dataframe(
                        label="Raw feature values", interactive=True,
<<<<<<< HEAD
                        wrap=True, row_count=(1,"fixed"))
                    # Model selector for custom prediction
                    custom_model_dd = gr.Dropdown(
                        label="Model to use for prediction",
                        choices=[], value=None, interactive=True)
                with gr.Column(scale=1, min_width=200):
                    predict_custom_btn = gr.Button(
                        "🔬 Predict My Row", elem_classes=["predict-btn"])
                    custom_pred_out = gr.HTML(
                        _empty_html("Run the pipeline first, then enter values above."))

            gr.HTML('<div class="divider"></div>')
            gr.HTML('<div class="section-label">Test-Set Predictions</div>')
            test_dl_out   = gr.HTML(_empty_html("Upload a test file to generate predictions."))
            submission_file = gr.File(label="⬇ submission.csv", interactive=False)

        # ══ TAB 5: CODE & REVIEW ══════════════════════════════════════════
        with gr.Tab("💻 Code & Review"):
            gr.HTML('<div class="section-label">Agent Pipeline Review</div>')
            review_out = gr.Textbox(
                label="", interactive=False, lines=12,
                elem_classes=["review-output"],
                placeholder="The Code Generator agent's pipeline review appears here…",
            )
            gr.HTML('<div class="divider"></div>')
            gr.HTML('<div class="section-label">Generated Reproducible Code</div>')
            code_out = gr.Textbox(
                label="", interactive=False, lines=24,
                elem_classes=["code-output"],
                placeholder="Agent-written reproducible Python script appears here…",
            )
            gr.HTML('<div class="section-label" style="margin-top:14px;">Downloads</div>')
            with gr.Row():
                code_file   = gr.File(label="⬇ pipeline.py",           interactive=False)
                model_file  = gr.File(label="⬇ fitted_model.joblib",   interactive=False)
                bundle_file = gr.File(label="⬇ all artifacts (.zip)",  interactive=False)

    # ══════════════════════════════════════════════════════════════════════
    # WIRING
    # ══════════════════════════════════════════════════════════════════════

    # File uploads
    train_file.change(
        fn=load_train, inputs=[train_file],
        outputs=[train_df_state, train_path_state, target_dropdown,
                 id_col_dropdown, preview_html, train_status])
    valid_file.change(
        fn=load_valid, inputs=[valid_file],
        outputs=[valid_df_state, valid_path_state, valid_status])
    test_file.change(
        fn=load_test, inputs=[test_file],
        outputs=[test_df_state, test_path_state, test_status])

    # Run button
=======
                        wrap=True, row_count=(1, "fixed"),
                    )
                with gr.Column(scale=1, min_width=200):
                    predict_custom_btn = gr.Button(
                        "🔬 Predict My Row", elem_classes=["predict-btn"],
                    )
                    custom_pred_out = gr.HTML(_empty_html(
                        "Run the pipeline first, then enter values above."
                    ))

            gr.HTML('<div class="divider"></div>')
            gr.HTML('<div class="section-label">Test-Set Predictions</div>')
            test_dl_out = gr.HTML(_empty_html(
                "Upload a test file to generate submission predictions."
            ))
            submission_file = gr.File(
                label="⬇ submission.csv", interactive=False,
            )

        # ════════════════════════ Tab 5: CODE & REVIEW ════════════════════════
        with gr.Tab("💻 Code & Review"):
            gr.HTML('<div class="section-label">CrewAI Review</div>')
            review_out = gr.Textbox(
                label="", interactive=False, lines=10,
                elem_classes=["review-output"],
                placeholder="If enabled in Configure tab, the CrewAI agent's "
                            "summary appears here.",
            )

            gr.HTML('<div class="divider"></div>')
            gr.HTML('<div class="section-label">Generated Reproducible Code</div>')
            code_out = gr.Textbox(
                label="", interactive=False, lines=22,
                elem_classes=["code-output"],
                placeholder="Reproducible code appears here once training completes…",
            )

            gr.HTML('<div class="section-label" style="margin-top:14px;">Downloads</div>')
            with gr.Row():
                code_file = gr.File(label="⬇ pipeline.py", interactive=False)
                model_file = gr.File(label="⬇ fitted_pipeline.joblib", interactive=False)
                bundle_file = gr.File(label="⬇ all artifacts (.zip)", interactive=False)

    # ════════════════════════ Wiring ════════════════════════
    train_file.change(
        fn=load_train, inputs=[train_file],
        outputs=[train_df_state, train_path_state,
                 target_dropdown, id_col_dropdown,
                 preview_html, train_status],
    )
    valid_file.change(
        fn=load_valid, inputs=[valid_file],
        outputs=[valid_df_state, valid_path_state, valid_status],
    )
    test_file.change(
        fn=load_test, inputs=[test_file],
        outputs=[test_df_state, test_path_state, test_status],
    )

>>>>>>> 6292a482cf1e2f547f6a880fb59fdb116f3f1bc2
    run_btn.click(
        fn=on_run,
        inputs=[
            train_df_state, valid_df_state, test_df_state,
            train_path_state, valid_path_state, test_path_state,
<<<<<<< HEAD
            target_dropdown, model_group_dd,
            test_size, shuffle_split, random_state,
            api_key_input, openai_model_dd, id_col_dropdown,
        ],
        outputs=[
            status_out,          # 0
            preprocessing_out,   # 1
            metrics_out,         # 2
            comparison_out,      # 3
            charts_out,          # 4
            feat_out,            # 5
            sample_out,          # 6
            test_dl_out,         # 7
            cross_model_out,     # 8
            code_out,            # 9
            review_out,          # 10
            code_file,           # 11
            submission_file,     # 12
            model_file,          # 13
            bundle_file,         # 14
            custom_inputs,       # 15
            log_out,             # 16
            results_model_dd,    # 17
            row_idx_slider,      # 18
        ],
    )

    # After run, also populate the custom-predict model dropdown
    def _sync_custom_dd(choices_update):
        """Mirror results_model_dd choices into custom_model_dd."""
        choices = choices_update.get("choices", []) if isinstance(choices_update, dict) else []
        val     = choices[0] if choices else None
        return gr.update(choices=choices, value=val)

    results_model_dd.change(
        fn=select_active_model,
        inputs=[results_model_dd],
        outputs=[metrics_out, charts_out, feat_out, sample_out],
    )

    row_idx_slider.change(
        fn=update_sample_row,
        inputs=[row_idx_slider],
        outputs=[sample_out],
    )

    predict_custom_btn.click(
        fn=do_custom_predict,
        inputs=[custom_model_dd, custom_inputs],
        outputs=[custom_pred_out],
    )

    # Reset
    def _reset_all():
        STATE.reset_run()
        STATE.set("train_df", None)
        STATE.set("eval_df",  None)
        STATE.set("test_df",  None)
        return (
            None, None, None,
            gr.update(choices=[], value=None),
            gr.update(choices=["(none)"], value="(none)"),
            _empty_html("No dataset loaded yet."),
            "*Upload a training file to begin.*",
            None, None, None, "*No validation file — agents will split training data.*",
            None, None, None, "*No test file uploaded.*",
            _info_banner("👋 Upload a training dataset on the <b>Data</b> tab to begin.", "info"),
=======
            target_dropdown, selected_models,
            test_size, shuffle_split, random_state, id_threshold,
            enable_cv, cv_folds, enable_tuning,
            api_key_input, openai_model_dropdown, id_col_dropdown,
            run_review,
        ],
        outputs=[
            status_out,                                 # 0
            preprocessing_out,                          # 1
            metrics_out,                                # 2
            comparison_out,                             # 3
            charts_out,                                 # 4
            feat_out,                                   # 5
            sample_out,                                 # 6
            test_dl_out,                                # 7
            cross_model_out,                            # 8
            code_out,                                   # 9
            review_out,                                 # 10
            code_file,                                  # 11
            submission_file,                            # 12
            model_file,                                 # 13
            bundle_file,                                # 14
            custom_inputs,                              # 15
            pipeline_state,                             # 16
            log_out,                                    # 17
            results_model_dd,                           # 18
            row_idx_slider,                             # 19
        ],
    )

    predict_custom_btn.click(
        fn=do_custom_predict,
        inputs=[pipeline_state, custom_inputs],
        outputs=[custom_pred_out],
    )

    row_idx_slider.change(
        fn=update_sample_row,
        inputs=[pipeline_state, row_idx_slider],
        outputs=[sample_out],
    )

    # Switching the active model in the Results tab re-renders metrics,
    # dashboard charts, importances and the sample prediction — without
    # retraining. The state is updated so the Predict tab also follows.
    results_model_dd.change(
        fn=select_results_model,
        inputs=[pipeline_state, results_model_dd],
        outputs=[pipeline_state, metrics_out, charts_out, feat_out, sample_out],
    )

    # Reset all uploads to initial state
    def _reset_all():
        return (
            None, None,                                     # train_file, train_df_state
            None,                                           # train_path_state
            gr.update(choices=[], value=None),               # target_dropdown
            gr.update(choices=["(none)"], value="(none)"),   # id_col_dropdown
            '<p style="color:#475569;font-size:13px;padding:20px;text-align:center;">'
            'No dataset loaded yet.</p>',                    # preview_html
            "*Upload a training file to begin.*",            # train_status
            None, None, None,                               # valid_file, valid_df_state, valid_path_state
            "*No validation file. The app will split training data.*",  # valid_status
            None, None, None,                               # test_file, test_df_state, test_path_state
            "*No test file uploaded.*",                      # test_status
            _info_banner("👋 Upload a training dataset on the <b>Data</b> tab to begin.", "info"),  # status_out
>>>>>>> 6292a482cf1e2f547f6a880fb59fdb116f3f1bc2
        )

    reset_btn.click(
        fn=_reset_all,
        inputs=[],
        outputs=[
            train_file, train_df_state, train_path_state,
            target_dropdown, id_col_dropdown,
            preview_html, train_status,
            valid_file, valid_df_state, valid_path_state, valid_status,
            test_file, test_df_state, test_path_state, test_status,
            status_out,
        ],
    )


if __name__ == "__main__":
    demo.launch()
