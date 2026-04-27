"""
CrewAI AutoML — Entry point.
Loads CSS from styles.css, imports all helpers, builds the Gradio UI.
"""
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
    # Store the original filenames so the generated reproducible script can
    # reference them by name (instead of placeholders like 'train.csv').
    train_path_state = gr.State(None)
    valid_path_state = gr.State(None)
    test_path_state = gr.State(None)

    gr.HTML("""
    <div class="header-band">
      <p class="header-title">⚡ Data Scientist Agent with CrewAI for End-to-End Regression Modeling</p>
      <p class="header-sub">
        Autonomous AI agents collaborate to preprocess any dataset, train & compare
        multiple models, generate reproducible code, and deliver expert-level review
        — all powered by CrewAI's multi-agent orchestration.
      </p>
    </div>
    """)

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
            with gr.Row(equal_height=True):
                with gr.Column(scale=1):
                    gr.HTML('<div class="step-badge"><span class="num">1</span> Training Dataset '
                            '<span style="color:#ef4444;margin-left:4px;">required</span></div>')
                    train_file = gr.File(
                        label="Train CSV / Excel / Parquet / JSON",
                        file_types=FILE_TYPES,
                        elem_classes=["upload-zone"],
                    )
                    train_status = gr.Markdown("*Upload a training file to begin.*")

                with gr.Column(scale=1):
                    gr.HTML('<div class="opt-badge">⚙ Optional — Validation File</div>')
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
            gr.HTML('<div class="section-label">Cross-Model Comparison Charts</div>')
            cross_model_out = gr.HTML(_empty_html(
                "Select two or more models in the Configure tab to enable "
                "side-by-side comparison charts."
            ))

            gr.HTML('<div class="divider"></div>')
            gr.HTML('<div class="section-label">Performance Charts (active model)</div>')
            charts_out = gr.HTML(_empty_html())

            gr.HTML('<div class="divider"></div>')
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
            sample_out = gr.HTML(_empty_html())

            gr.HTML('<div class="divider"></div>')
            gr.HTML("""
            <div style="background:#0f1729;border:1px solid #1e3a4a;border-radius:14px;
                        padding:18px 20px;margin-bottom:12px;">
              <span style="color:#06b6d4;font-size:11px;font-weight:700;
                           letter-spacing:2px;text-transform:uppercase;">
                🔬 Custom Raw Row Prediction
              </span>
              <p style="color:#64748b;font-size:12px;margin:6px 0 0;">
                Edit raw feature values. The same fitted preprocessing pipeline
                is applied automatically.
              </p>
            </div>
            """)
            with gr.Row():
                with gr.Column(scale=2):
                    custom_inputs = gr.Dataframe(
                        label="Raw feature values", interactive=True,
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

    run_btn.click(
        fn=on_run,
        inputs=[
            train_df_state, valid_df_state, test_df_state,
            train_path_state, valid_path_state, test_path_state,
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
