<<<<<<< HEAD
"""
helpers.py — Rendering and utility functions for the agent-based AutoML app.
Reads from STATE (populated by agents) to produce HTML/file outputs for Gradio.
No sklearn Pipeline, ColumnTransformer, or preprocessing logic lives here.
"""
import base64, io, os, tempfile, traceback
from typing import Optional, Tuple
import gradio as gr
import numpy as np
import pandas as pd
from ml_state import STATE

# Colour constants
GREEN = "#10b981"; AMBER = "#f59e0b"; RED = "#ef4444"
PURPLE = "#a78bfa"; CYAN = "#06b6d4"

FILE_TYPES    = [".csv", ".tsv", ".xlsx", ".xls", ".parquet", ".json", ".feather"]
OPENAI_MODELS = ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini", "gpt-4.1", "gpt-4-turbo"]
MODEL_GROUPS  = [
    "XGBoost + LightGBM + RandomForest (recommended)",
    "XGBoost + Ridge + RandomForest",
    "LightGBM + ExtraTrees + Ridge",
    "RandomForest + GradientBoosting + Ridge",
    "All available boosting models",
    "Linear models only (Ridge + Lasso + ElasticNet)",
    "Let the agent decide",
]

# ── HTML helpers ───────────────────────────────────────────────────────────────

def _empty_html(msg="Run the pipeline to see results."):
    return f'<p style="color:#475569;font-size:13px;padding:8px 0;">{msg}</p>'

def _info_banner(msg, kind="info"):
    palette = {"info":(CYAN,"#0f1f2e"),"ok":(GREEN,"#0f2018"),
               "warn":(AMBER,"#1f1709"),"error":(RED,"#1f0f12")}
    fg, bg = palette.get(kind, palette["info"])
    return (f'<div style="background:{bg};border:1px solid {fg};border-left:4px solid {fg};'
            f'border-radius:8px;padding:10px 14px;color:{fg};font-size:13px;'
            f'font-family:DM Mono,monospace;">{msg}</div>')

def _png_to_html(png_bytes):
    b64 = base64.b64encode(png_bytes).decode()
    return f'<img src="data:image/png;base64,{b64}" style="width:100%;border-radius:10px;margin-top:6px;">'

def _html_table(df, max_rows=20):
    if df is None or df.empty:
        return _empty_html("No rows to display.")
    show = df.head(max_rows)
    hdr  = "".join(f"<th>{c}</th>" for c in show.columns)
    rows = "".join("<tr>"+"".join(f"<td>{v}</td>" for v in r)+"</tr>"
                   for _,r in show.iterrows())
    more = (f"<p style='color:#64748b;font-size:11px;'>Showing {len(show)} of {len(df)}.</p>"
            if len(df) > len(show) else "")
    return (f"<div style='overflow-x:auto;border:1px solid #2a2a3e;border-radius:10px;'>"
            f"<table style='width:100%;border-collapse:collapse;font-size:12px;font-family:DM Mono,monospace;'>"
            f"<thead><tr style='background:#1a1a2e;color:#a78bfa;'>{hdr}</tr></thead>"
            f"<tbody style='color:#e2e8f0;'>{rows}</tbody></table></div>{more}")

def _build_preview_html(df, n=10):
    preview = df.head(n)
    hdr = "".join(f"<th>{c}</th>" for c in df.columns)
    rows_html = "".join(
        "<tr>"+"".join(f'<td>{str(v)[:28]+"..." if len(str(v))>28 else v}</td>'
                       for v in row)+"</tr>"
        for _,row in preview.iterrows()
    )
    return (f'<div style="overflow-x:auto;border-radius:10px;border:1px solid #2a2a3e;">'
            f'<table style="width:100%;border-collapse:collapse;font-size:12px;font-family:DM Mono,monospace;">'
            f'<thead><tr style="background:#1a1a2e;color:#a78bfa;">{hdr}</tr></thead>'
            f'<tbody style="color:#e2e8f0;">{rows_html}</tbody></table></div>'
            f'<p style="font-size:11px;color:#64748b;margin:4px 0 0;">'
            f'Showing first {min(n,len(df))} of {len(df):,} rows</p>')

# ── File I/O ───────────────────────────────────────────────────────────────────

def _read_file(path):
    ext = os.path.splitext(path)[-1].lower()
    if ext in (".csv",".tsv",".txt"): return pd.read_csv(path)
    if ext in (".xlsx",".xls"):       return pd.read_excel(path)
    if ext == ".parquet":             return pd.read_parquet(path)
    if ext == ".json":                return pd.read_json(path)
    if ext in (".feather",".ft"):     return pd.read_feather(path)
    raise ValueError(f"Unsupported format: {ext}")

def _clean_df(df):
    if df is None or df.empty: return df
    drop = [c for c in df.columns if isinstance(c,str) and c.startswith("Unnamed:")]
    if drop: df = df.drop(columns=drop)
    for c in df.columns:
        if pd.api.types.is_bool_dtype(df[c]): df[c] = df[c].astype(int)
    return df

def load_train(file):
    if file is None:
        return (None, None, gr.update(choices=[],value=None),
                gr.update(choices=["(none)"],value="(none)"),
                "", "*Upload a training file to begin.*")
    try:
        df = _clean_df(_read_file(file.name))
        STATE.set("train_df", df)
        num   = df.select_dtypes(include=[np.number]).columns.tolist()
        fname = os.path.basename(file.name)
        cols  = df.columns.tolist()
        tv    = num[0] if num else cols[0]
        return (df, fname,
                gr.update(choices=cols, value=tv),
                gr.update(choices=["(none)"]+cols, value="(none)"),
                _build_preview_html(df),
                f"✅ **{fname}** — {len(df):,} rows × {len(df.columns)} cols")
    except Exception as e:
        return (None, None, gr.update(choices=[],value=None),
                gr.update(choices=["(none)"],value="(none)"),
                "", f"❌ {e}")

def load_valid(file):
    if file is None:
        STATE.set("eval_df", None)
        return None, None, "*No validation file — agents will split training data.*"
    try:
        df = _clean_df(_read_file(file.name))
        STATE.set("eval_df", df)
        fname = os.path.basename(file.name)
        return df, fname, f"✅ **{fname}** — {len(df):,} rows × {len(df.columns)} cols"
    except Exception as e:
        return None, None, f"❌ {e}"

def load_test(file):
    if file is None:
        STATE.set("test_df", None)
        return None, None, "*No test file uploaded.*"
    try:
        df = _clean_df(_read_file(file.name))
        STATE.set("test_df", df)
        fname = os.path.basename(file.name)
        return df, fname, f"✅ **{fname}** — {len(df):,} rows × {len(df.columns)} cols"
    except Exception as e:
        return None, None, f"❌ {e}"

# ── Renderers ─────────────────────────────────────────────────────────────────

def render_preprocessing_summary():
    summary = STATE.get("preprocessing_summary","")
    if not summary.strip():
        return _empty_html("Preprocessing summary will appear after the pipeline runs.")
    rows_html = ""
    for line in summary.strip().split("\n"):
        line = line.strip()
        if not line: continue
        if ":" in line:
            k,_,v = line.partition(":")
            rows_html += (f"<tr><td style='padding:8px 12px;color:#94a3b8;width:30%;font-size:12px;font-weight:500;'>{k.strip()}</td>"
                          f"<td style='padding:8px 12px;color:#e2e8f0;font-size:12px;font-family:DM Mono,monospace;'>{v.strip()}</td></tr>")
        else:
            rows_html += f"<tr><td colspan='2' style='padding:6px 12px;color:#64748b;font-size:11px;'>{line}</td></tr>"
    return (f"<div style='background:#13131f;border:1px solid #2a2a3e;border-radius:12px;overflow:hidden;'>"
            f"<div style='background:#1a1a2e;padding:10px 14px;border-bottom:1px solid #2a2a3e;'>"
            f"<span style='color:#10b981;font-size:11px;font-weight:600;letter-spacing:2px;text-transform:uppercase;'>Preprocessing Summary (Agent-Written)</span></div>"
            f"<table style='width:100%;border-collapse:collapse;'>{rows_html}</table></div>")

def render_metrics_html(model_name=None):
    if model_name is None: model_name = STATE.get("active_model")
    metrics = STATE.get("metrics", {})
    if not model_name or model_name not in metrics:
        return _empty_html("Metrics will appear after training.")
    m      = metrics[model_name]
    target = STATE.get("target_col","target")
    src    = STATE.get("eval_source","eval set")
    r2     = m.get("R2",0)
    r2c    = GREEN if r2>=0.8 else AMBER if r2>=0.5 else RED
    rows   = [("MAE",f"{m.get('MAE',float('nan')):.4f}",PURPLE),
              ("MSE",f"{m.get('MSE',float('nan')):.4f}",PURPLE),
              ("RMSE",f"{m.get('RMSE',float('nan')):.4f}",PURPLE),
              ("R2",f"{r2:.4f}",r2c),
              ("MAPE %",f"{m.get('MAPE',float('nan')):.2f}",AMBER)]
    rows_html = "".join(
        f"<tr><td style='padding:9px 14px;color:#cbd5e1;font-size:12px;text-transform:uppercase;letter-spacing:1px;font-weight:500;'>{lb}</td>"
        f"<td style='padding:9px 14px;color:{c};font-size:16px;font-weight:700;text-align:right;font-family:DM Mono,monospace;'>{v}</td></tr>"
        for lb,v,c in rows)
    return (f"<div style='background:#13131f;border:1px solid #2a2a3e;border-radius:12px;overflow:hidden;'>"
            f"<div style='background:#1a1a2e;padding:10px 14px;border-bottom:1px solid #2a2a3e;'>"
            f"<span style='color:#a78bfa;font-size:11px;font-weight:600;letter-spacing:2px;text-transform:uppercase;'>{model_name} · {target} · {src}</span></div>"
            f"<table style='width:100%;border-collapse:collapse;'>{rows_html}</table></div>")

def render_comparison_html():
    df = STATE.get("comparison_df")
    if df is None or df.empty:
        return _empty_html("Comparison table will appear after training.")
    display = df[["Model","MAE","MSE","RMSE","R2"]].copy()
    for col in ["MAE","MSE","RMSE","R2"]:
        if col in display.columns:
            display[col] = display[col].map(lambda x: f"{x:.4f}")
    return _html_table(display, max_rows=20)

def render_charts_html(chart_name="performance"):
    charts = STATE.get("charts_png",{})
    if chart_name not in charts or not charts[chart_name]:
        return _empty_html(f"Chart '{chart_name}' will appear after training.")
    return _png_to_html(charts[chart_name])

def render_feat_html():
    charts = STATE.get("charts_png",{})
    if "importance" not in charts or not charts["importance"]:
        return _empty_html("Feature importance chart will appear after training.")
    note = ('<p style="color:#94a3b8;font-size:11px;margin:8px 0 0;line-height:1.5;">'
            'Feature importances were computed directly from the trained model '
            '(feature_importances_ for tree models, abs(coef_) for linear models).</p>')
    return _png_to_html(charts["importance"]) + note

def render_cross_model_html():
    charts = STATE.get("charts_png",{})
    if "comparison" not in charts or not charts["comparison"]:
        return _empty_html("Cross-model comparison chart appears when more than one model is trained.")
    return _png_to_html(charts["comparison"])

def render_sample_html(row_idx=0):
    X_eval  = STATE.get("X_eval")
    y_eval  = STATE.get("y_eval")
    target  = STATE.get("target_col","target")
    active  = STATE.get("active_model")
    models  = STATE.get("models",{})
    if X_eval is None or y_eval is None or not active or active not in models:
        return _empty_html("Sample prediction will appear after training.")
    model   = models[active]
    n       = X_eval.shape[0] if hasattr(X_eval,"shape") else len(X_eval)
    row_idx = max(0, min(int(row_idx), n-1))
    if isinstance(X_eval, pd.DataFrame):
        row_arr = X_eval.iloc[[row_idx]].values
        feat_d  = dict(list(X_eval.iloc[row_idx].items())[:10])
    else:
        row_arr = X_eval[[row_idx]]
        feat_d  = {}
    actual    = float(y_eval.iloc[row_idx] if isinstance(y_eval,pd.Series) else y_eval[row_idx])
    predicted = float(model.predict(row_arr)[0])
    err_pct   = abs(actual-predicted)/(abs(actual)+1e-9)*100
    err_col   = GREEN if err_pct<10 else AMBER if err_pct<25 else RED
    feat_rows = "".join(
        f"<tr><td style='padding:5px 10px;color:#64748b;font-size:11px;font-family:DM Mono,monospace;'>{k}</td>"
        f"<td style='padding:5px 10px;color:#cbd5e1;font-size:11px;font-family:DM Mono,monospace;text-align:right;'>{v}</td></tr>"
        for k,v in feat_d.items())
    return (f"<div style='background:#13131f;border:1px solid #2a2a3e;border-radius:12px;overflow:hidden;'>"
            f"<div style='background:#1a1a2e;padding:10px 14px;border-bottom:1px solid #2a2a3e;'>"
            f"<span style='color:#06b6d4;font-size:11px;font-weight:600;letter-spacing:2px;text-transform:uppercase;'>"
            f"Sample · Row {row_idx} · {target}</span></div>"
            f"<div style='display:flex;border-bottom:1px solid #2a2a3e;'>"
            f"<div style='flex:1;padding:16px 18px;border-right:1px solid #2a2a3e;'>"
            f"<div style='color:#64748b;font-size:10px;text-transform:uppercase;margin-bottom:6px;'>Actual</div>"
            f"<div style='color:{GREEN};font-size:24px;font-weight:700;font-family:DM Mono,monospace;'>{actual:.4f}</div></div>"
            f"<div style='flex:1;padding:16px 18px;border-right:1px solid #2a2a3e;'>"
            f"<div style='color:#64748b;font-size:10px;text-transform:uppercase;margin-bottom:6px;'>Predicted</div>"
            f"<div style='color:#a78bfa;font-size:24px;font-weight:700;font-family:DM Mono,monospace;'>{predicted:.4f}</div></div>"
            f"<div style='flex:1;padding:16px 18px;'>"
            f"<div style='color:#64748b;font-size:10px;text-transform:uppercase;margin-bottom:6px;'>Error %</div>"
            f"<div style='color:{err_col};font-size:24px;font-weight:700;font-family:DM Mono,monospace;'>{err_pct:.1f}%</div>"
            f"</div></div>"
            f"<div style='padding:10px 4px 6px;'>"
            f"<div style='color:#64748b;font-size:10px;text-transform:uppercase;margin:0 10px 6px;'>Preprocessed Features (first 10)</div>"
            f"<table style='width:100%;border-collapse:collapse;'>{feat_rows}</table></div></div>")

def render_test_download(id_col="(none)"):
    test_df = STATE.get("test_df")
    active  = STATE.get("active_model")
    models  = STATE.get("models",{})
    fn      = STATE.get("preprocess_fn")
    if test_df is None:
        return _empty_html("No test file was uploaded."), None
    if not active or active not in models or fn is None:
        return _empty_html("Run the pipeline first to generate test predictions."), None
    try:
        X_test = fn(test_df)
        preds  = np.asarray(models[active].predict(X_test)).flatten()
        if id_col and id_col != "(none)" and id_col in test_df.columns:
            out_df = pd.DataFrame({id_col: test_df[id_col].values, "prediction": preds})
        else:
            out_df = pd.DataFrame({"id": np.arange(len(preds)), "prediction": preds})
        path = os.path.join(__import__("tempfile").gettempdir(), "submission.csv")
        out_df.to_csv(path, index=False)
        STATE.set("submission_path", path)
        return _html_table(out_df, max_rows=8), path
    except Exception as e:
        return _info_banner(f"Prediction failed: {e}", "warn"), None

def do_custom_predict(model_name, raw_df):
    models = STATE.get("models",{})
    fn     = STATE.get("preprocess_fn")
    if not model_name or model_name not in models:
        return _empty_html("Run the pipeline first.")
    if raw_df is None or len(raw_df)==0:
        return _empty_html("Enter at least one row of feature values.")
    if fn is None:
        return _empty_html("Preprocessing function not available — re-run the pipeline.")
    try:
        row  = raw_df.iloc[[0]].copy()
        pred = float(models[model_name].predict(fn(row))[0])
        feat_rows = "".join(
            f"<tr><td style='padding:5px 12px;color:#64748b;font-size:11px;font-family:DM Mono,monospace;'>{c}</td>"
            f"<td style='padding:5px 12px;color:#cbd5e1;font-size:11px;font-family:DM Mono,monospace;text-align:right;'>{row.iloc[0][c]}</td></tr>"
            for c in row.columns[:20])
        return (f"<div style='background:#13131f;border:1px solid #2a2a3e;border-radius:12px;overflow:hidden;'>"
                f"<div style='background:#0f2030;padding:10px 14px;border-bottom:1px solid #2a2a3e;'>"
                f"<span style='color:#06b6d4;font-size:11px;font-weight:600;letter-spacing:2px;text-transform:uppercase;'>Custom Prediction · {model_name}</span></div>"
                f"<div style='padding:18px;'><div style='color:#64748b;font-size:10px;text-transform:uppercase;margin-bottom:8px;'>Predicted Value</div>"
                f"<div style='color:#a78bfa;font-size:36px;font-weight:700;font-family:DM Mono,monospace;'>{pred:.4f}</div></div>"
                f"<div style='border-top:1px solid #2a2a3e;padding:4px;'>"
                f"<table style='width:100%;border-collapse:collapse;'>{feat_rows}</table></div></div>")
    except Exception as e:
        return _info_banner(f"Prediction error: {e}", "error")

def update_sample_row(row_idx):
    return render_sample_html(int(row_idx))

def select_active_model(model_name):
    if not model_name or model_name not in STATE.get("models",{}):
        return (render_metrics_html(), render_charts_html("performance"),
                render_feat_html(), render_sample_html(0))
    STATE.set("active_model", model_name)
    return (render_metrics_html(model_name), render_charts_html("performance"),
            render_feat_html(), render_sample_html(0))

def build_custom_inputs_df():
    train_df = STATE.get("train_df")
    target   = STATE.get("target_col","")
    if train_df is None: return pd.DataFrame()
    feat_cols = [c for c in train_df.columns if c != target]
    if not feat_cols: return pd.DataFrame()
    full = train_df[feat_cols].dropna()
    if len(full)==0: full = train_df[feat_cols]
    return full.head(1).reset_index(drop=True).copy()
=======
import base64
import importlib
import io
import os
import tempfile
import traceback
import zipfile
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple

import gradio as gr
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    mean_absolute_percentage_error,
)
from sklearn.model_selection import (
    KFold,
    RandomizedSearchCV,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler

# Local custom CrewAI tool — now actually used.
try:
    from notebookExecutor import NotebookCodeExecutor
    _HAS_NB_EXECUTOR = True
except Exception:
    _HAS_NB_EXECUTOR = False


# ── Model registry ──────────────────────────────────────────────────────────────

__all__ = [
    "AMBER",
    "CYAN",
    "DARK_AX",
    "DARK_BG",
    "DARK_FG",
    "DARK_GRD",
    "DateFeatureExtractor",
    "FILE_TYPES",
    "FrequencyEncoder",
    "GREEN",
    "MISSING_TOKENS",
    "MODEL_NAMES",
    "OPENAI_MODELS",
    "PURPLE",
    "RED",
    "TextStatsTransformer",
    "_active_pipeline",
    "_build_charts",
    "_build_comparison_charts",
    "_build_preprocessor",
    "_build_preview_html",
    "_build_reproducible_code",
    "_bundle_artifacts_zip",
    "_clean_loaded_df",
    "_coerce_numeric_series",
    "_create_code_file",
    "_detect_schema",
    "_empty_html",
    "_fig_to_b64",
    "_fig_to_png_bytes",
    "_format_kwargs",
    "_get_feature_names",
    "_grid_for_pipeline",
    "_html_table",
    "_info_banner",
    "_is_datetime_like",
    "_is_id_like_name",
    "_is_monotonic_counter",
    "_is_text_like",
    "_load_model",
    "_local_agent_report",
    "_metrics_dict",
    "_numeric_like_ratio",
    "_read_file",
    "_render_importances",
    "_render_metrics",
    "_render_preprocessing_summary",
    "_render_sample",
    "_render_test_download",
    "_run_crewai_review",
    "_safe_onehot",
    "_safe_onehot_encoder",
    "_save_model_to_disk",
    "_should_log_transform",
    "_strip_tuning_prefix",
    "_wrap_target",
    "do_custom_predict",
    "load_optional",
    "load_test",
    "load_train",
    "load_valid",
    "on_run",
    "select_results_model",
    "update_sample_row",
]

MODEL_REGISTRY: Dict[str, Tuple[str, str, Dict[str, Any]]] = {
    "Random Forest":              ("sklearn.ensemble",     "RandomForestRegressor",         {"n_estimators": 200, "random_state": 42, "n_jobs": -1}),
    "Extra Trees":                ("sklearn.ensemble",     "ExtraTreesRegressor",           {"n_estimators": 200, "random_state": 42, "n_jobs": -1}),
    "Gradient Boosting":          ("sklearn.ensemble",     "GradientBoostingRegressor",     {"n_estimators": 150, "random_state": 42}),
    "Hist Gradient Boosting":     ("sklearn.ensemble",     "HistGradientBoostingRegressor", {"random_state": 42}),
    "AdaBoost":                   ("sklearn.ensemble",     "AdaBoostRegressor",             {"n_estimators": 150, "random_state": 42}),
    "Linear Regression":          ("sklearn.linear_model", "LinearRegression",              {}),
    "Ridge Regression":           ("sklearn.linear_model", "Ridge",                         {"alpha": 1.0}),
    "Lasso Regression":           ("sklearn.linear_model", "Lasso",                         {"alpha": 0.05, "max_iter": 10000}),
    "ElasticNet":                 ("sklearn.linear_model", "ElasticNet",                    {"alpha": 0.05, "l1_ratio": 0.5, "max_iter": 10000}),
    "Bayesian Ridge":             ("sklearn.linear_model", "BayesianRidge",                 {}),
    "Huber Regressor":            ("sklearn.linear_model", "HuberRegressor",                {"max_iter": 1000}),
    "SGD Regressor":              ("sklearn.linear_model", "SGDRegressor",                  {"random_state": 42, "max_iter": 2000}),
    "Decision Tree":              ("sklearn.tree",         "DecisionTreeRegressor",         {"random_state": 42}),
    "K-Nearest Neighbors":        ("sklearn.neighbors",    "KNeighborsRegressor",           {"n_neighbors": 5}),
    "Support Vector Regression":  ("sklearn.svm",          "SVR",                           {"kernel": "rbf", "C": 1.0}),
}
MODEL_NAMES = list(MODEL_REGISTRY.keys())

OPENAI_MODELS = [
    "gpt-4o-mini",
    "gpt-4o",
    "gpt-4.1-mini",
    "gpt-4.1",
    "gpt-4-turbo",
]

FILE_TYPES = [".csv", ".tsv", ".txt", ".xlsx", ".xls", ".parquet", ".json",
              ".jsonl", ".ndjson", ".feather", ".ft"]
MISSING_TOKENS = {"", "na", "n/a", "none", "null", "nan", "?", "-", "--"}

TUNING_PARAM_GRIDS: Dict[str, Dict[str, List[Any]]] = {
    "Random Forest":          {"model__n_estimators": [150, 250, 400], "model__max_depth": [None, 8, 16, 24], "model__min_samples_leaf": [1, 2, 4]},
    "Extra Trees":            {"model__n_estimators": [150, 250, 400], "model__max_depth": [None, 8, 16, 24], "model__min_samples_leaf": [1, 2, 4]},
    "Gradient Boosting":      {"model__n_estimators": [100, 200, 350], "model__learning_rate": [0.03, 0.05, 0.1], "model__max_depth": [2, 3, 4]},
    "Hist Gradient Boosting": {"model__learning_rate": [0.03, 0.05, 0.1], "model__max_iter": [100, 200, 350], "model__max_leaf_nodes": [15, 31, 63]},
    "Ridge Regression":       {"model__alpha": [0.01, 0.1, 1.0, 10.0, 100.0]},
    "Lasso Regression":       {"model__alpha": [0.001, 0.01, 0.05, 0.1, 1.0]},
    "ElasticNet":             {"model__alpha": [0.001, 0.01, 0.05, 0.1], "model__l1_ratio": [0.2, 0.5, 0.8]},
}

# Theme constants
DARK_BG  = "#0d0d1a"
DARK_FG  = "#e2e8f0"
DARK_AX  = "#1a1a2e"
DARK_GRD = "#2a2a3e"
PURPLE   = "#a78bfa"
CYAN     = "#06b6d4"
GREEN    = "#10b981"
AMBER    = "#f59e0b"
RED      = "#ef4444"


# ── General helpers ─────────────────────────────────────────────────────────────
def _read_file(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[-1].lower()
    if ext == ".csv":
        return pd.read_csv(path)
    if ext in (".xlsx", ".xls"):
        return pd.read_excel(path)
    if ext == ".parquet":
        return pd.read_parquet(path)
    if ext == ".json":
        return pd.read_json(path)
    raise ValueError(f"Unsupported file format: {ext}")


def _load_model(model_name: str):
    mod_path, cls_name, kwargs = MODEL_REGISTRY[model_name]
    module = importlib.import_module(mod_path)
    cls = getattr(module, cls_name)
    return cls(**kwargs)


def _safe_onehot_encoder():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def _empty_html(msg: str = "Run the pipeline to see results.") -> str:
    return f'<p style="color:#475569;font-size:13px;padding:8px 0;">{msg}</p>'


def _info_banner(msg: str, kind: str = "info") -> str:
    palette = {
        "info":    (CYAN,   "#0f1f2e"),
        "ok":      (GREEN,  "#0f2018"),
        "warn":    (AMBER,  "#1f1709"),
        "error":   (RED,    "#1f0f12"),
    }
    fg, bg = palette.get(kind, palette["info"])
    return (
        f'<div style="background:{bg};border:1px solid {fg};border-left:4px solid {fg};'
        f'border-radius:8px;padding:10px 14px;color:{fg};font-size:13px;'
        f'font-family:\'DM Mono\',monospace;">{msg}</div>'
    )


def _fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight",
                facecolor=DARK_BG, edgecolor="none")
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode()
    plt.close(fig)
    return f'<img src="data:image/png;base64,{b64}" style="width:100%;border-radius:10px;margin-top:6px;">'


def _fig_to_png_bytes(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight",
                facecolor=DARK_BG, edgecolor="none")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def _build_preview_html(df: pd.DataFrame, n: int = 10) -> str:
    preview = df.head(n)
    headers = "".join(f"<th>{c}</th>" for c in df.columns)
    rows_html = "".join(
        "<tr>" + "".join(
            f'<td title="{str(v)}">{str(v)[:28] + "…" if len(str(v)) > 28 else v}</td>'
            for v in row
        ) + "</tr>"
        for _, row in preview.iterrows()
    )
    return (
        '<div style="overflow-x:auto;border-radius:10px;border:1px solid #2a2a3e;margin-top:6px;">'
        '<table style="width:100%;border-collapse:collapse;font-family:\'DM Mono\',monospace;font-size:12px;">'
        f'<thead><tr style="background:#1a1a2e;color:#a78bfa;">{headers}</tr></thead>'
        f'<tbody style="color:#e2e8f0;">{rows_html}</tbody></table></div>'
        f'<p style="font-size:11px;color:#64748b;margin:4px 0 0;">'
        f'Showing first {min(n, len(df))} of {len(df):,} rows</p>'
    )


def _html_table(df: pd.DataFrame, max_rows: int = 20) -> str:
    if df is None or df.empty:
        return _empty_html("No rows to display.")
    show = df.head(max_rows).copy()
    headers = "".join(f"<th>{c}</th>" for c in show.columns)
    rows = "".join(
        "<tr>" + "".join(f"<td>{v}</td>" for v in row) + "</tr>"
        for _, row in show.iterrows()
    )
    more = (f"<p style='color:#64748b;font-size:11px;margin:4px 0;'>"
            f"Showing {len(show)} of {len(df)} rows.</p>") if len(df) > len(show) else ""
    return (
        "<div style='overflow-x:auto;border:1px solid #2a2a3e;border-radius:10px;'>"
        "<table style='width:100%;border-collapse:collapse;font-family:\"DM Mono\",monospace;font-size:12px;'>"
        f"<thead><tr style='background:#1a1a2e;color:#a78bfa;'>{headers}</tr></thead>"
        f"<tbody style='color:#e2e8f0;'>{rows}</tbody></table></div>{more}"
    )


# ── Dataset loaders ─────────────────────────────────────────────────────────────
def _clean_loaded_df(df: pd.DataFrame, numeric_threshold: float = 0.90) -> pd.DataFrame:
    """Standardise missing tokens, drop Unnamed index cols, coerce numeric-like
    strings (currency, percent), convert booleans to int."""
    if df is None or df.empty:
        return df
    out = df.copy()
    # Normalise missing tokens in object/string columns
    for c in out.select_dtypes(include=["object", "string"]).columns:
        vals = out[c].astype("string")
        out.loc[vals.str.strip().str.lower().isin(MISSING_TOKENS), c] = np.nan
    # Drop Unnamed index columns
    drop_cols = []
    for c in out.columns:
        if isinstance(c, str) and c.startswith("Unnamed:"):
            try:
                vals_num = pd.to_numeric(out[c], errors="coerce")
                if vals_num.notna().all() and (vals_num.diff().dropna() == 1).all():
                    drop_cols.append(c); continue
            except Exception: pass
            if out[c].isna().all():
                drop_cols.append(c)
    if drop_cols:
        out = out.drop(columns=drop_cols)
    # Booleans → int; numeric-like strings (currency, %) → float
    for c in out.columns:
        if pd.api.types.is_bool_dtype(out[c]):
            out[c] = out[c].astype(int)
        elif pd.api.types.is_object_dtype(out[c]) or pd.api.types.is_string_dtype(out[c]):
            if _numeric_like_ratio(out[c]) >= numeric_threshold:
                out[c] = _coerce_numeric_series(out[c])
    return out


def load_train(file):
    if file is None:
        return (None, None,
                gr.update(choices=[], value=None),
                gr.update(choices=[], value=None),
                "", "No training file uploaded.")
    try:
        df = _read_file(file.name)
        df = _clean_loaded_df(df)
        num = df.select_dtypes(include=[np.number]).columns.tolist()
        preview = _build_preview_html(df)
        fname = os.path.basename(file.name)
        status = (f"✅ **{fname}** — "
                  f"{len(df):,} rows × {len(df.columns)} cols | {len(num)} numeric")
        target_choices = df.columns.tolist()
        target_value = num[0] if num else target_choices[0]
        # All columns are candidates for the optional id column
        id_choices = ["(none)"] + target_choices
        return (df, fname,
                gr.update(choices=target_choices, value=target_value),
                gr.update(choices=id_choices, value="(none)"),
                preview, status)
    except Exception as e:
        return (None, None,
                gr.update(choices=[], value=None),
                gr.update(choices=[], value=None),
                "", f"❌ {e}")


def load_optional(file, label="file"):
    if file is None:
        return None, None, f"*No {label} uploaded (optional).*"
    try:
        df = _read_file(file.name)
        df = _clean_loaded_df(df)
        fname = os.path.basename(file.name)
        return df, fname, (f"✅ **{fname}** — "
                           f"{len(df):,} rows × {len(df.columns)} cols")
    except Exception as e:
        return None, None, f"❌ {e}"


def load_valid(file):
    return load_optional(file, "validation file")


def load_test(file):
    return load_optional(file, "test file")


# ── Custom transformers & preprocessing (Kaggle-ready) ───────────────────────

class FrequencyEncoder(BaseEstimator, TransformerMixin):
    """Leakage-safe frequency encoding for high-cardinality categoricals."""
    def fit(self, X, y=None):
        Xdf = pd.DataFrame(X).copy()
        self.feature_names_in_ = list(getattr(X, "columns", [f"x{i}" for i in range(Xdf.shape[1])]))
        Xdf.columns = self.feature_names_in_; self.maps_ = {}; n = max(len(Xdf), 1)
        for c in self.feature_names_in_:
            self.maps_[c] = (Xdf[c].astype("string").fillna("Unknown").value_counts(dropna=False) / n).to_dict()
        return self
    def transform(self, X):
        Xdf = pd.DataFrame(X).copy(); Xdf.columns = self.feature_names_in_
        out = pd.DataFrame(index=Xdf.index)
        for c in self.feature_names_in_:
            out[f"{c}__freq"] = Xdf[c].astype("string").fillna("Unknown").map(self.maps_.get(c, {})).fillna(0.0).astype(float)
        return out.values
    def get_feature_names_out(self, input_features=None):
        cols = list(input_features) if input_features is not None else self.feature_names_in_
        return np.array([f"{c}__freq" for c in cols], dtype=object)


class DateFeatureExtractor(BaseEstimator, TransformerMixin):
    """Extracts calendar and elapsed-time features from date columns."""
    _SUFFIXES = ["year","month","day","dayofweek","quarter","is_month_start","is_month_end","elapsed_days","missing"]
    def fit(self, X, y=None):
        Xdf = pd.DataFrame(X)
        self.feature_names_in_ = list(getattr(X, "columns", [f"d{i}" for i in range(Xdf.shape[1])]))
        return self
    def transform(self, X):
        Xdf = pd.DataFrame(X).copy(); Xdf.columns = self.feature_names_in_; out = pd.DataFrame(index=Xdf.index)
        for c in self.feature_names_in_:
            dt = pd.to_datetime(Xdf[c], errors="coerce")
            out[f"{c}__year"]=dt.dt.year; out[f"{c}__month"]=dt.dt.month; out[f"{c}__day"]=dt.dt.day
            out[f"{c}__dayofweek"]=dt.dt.dayofweek; out[f"{c}__quarter"]=dt.dt.quarter
            out[f"{c}__is_month_start"]=dt.dt.is_month_start.astype(float)
            out[f"{c}__is_month_end"]=dt.dt.is_month_end.astype(float)
            out[f"{c}__elapsed_days"]=dt.astype("int64", errors="ignore")/86_400_000_000_000
            out.loc[dt.isna(), f"{c}__elapsed_days"]=np.nan
            out[f"{c}__missing"]=dt.isna().astype(float)
        return out.values
    def get_feature_names_out(self, input_features=None):
        cols = list(input_features) if input_features is not None else self.feature_names_in_
        return np.array([f"{c}__{s}" for c in cols for s in self._SUFFIXES], dtype=object)


class TextStatsTransformer(BaseEstimator, TransformerMixin):
    """Extracts lightweight numeric stats from free-text columns."""
    _SUFFIXES = ["char_len","word_count","digit_count","upper_count","punct_count"]
    def fit(self, X, y=None):
        Xdf = pd.DataFrame(X)
        self.feature_names_in_ = list(getattr(X, "columns", [f"t{i}" for i in range(Xdf.shape[1])]))
        return self
    def transform(self, X):
        Xdf = pd.DataFrame(X).copy(); Xdf.columns = self.feature_names_in_; out = pd.DataFrame(index=Xdf.index)
        for c in self.feature_names_in_:
            s = Xdf[c].astype("string").fillna("")
            out[f"{c}__char_len"]=s.str.len().astype(float)
            out[f"{c}__word_count"]=s.str.split().map(lambda v: len(v) if isinstance(v,list) else 0).astype(float)
            out[f"{c}__digit_count"]=s.str.count(r"\d").astype(float)
            out[f"{c}__upper_count"]=s.str.count(r"[A-Z]").astype(float)
            out[f"{c}__punct_count"]=s.str.count(r"[^\w\s]").astype(float)
        return out.values
    def get_feature_names_out(self, input_features=None):
        cols = list(input_features) if input_features is not None else self.feature_names_in_
        return np.array([f"{c}__{s}" for c in cols for s in self._SUFFIXES], dtype=object)


# ── Data cleaning helpers ────────────────────────────────────────────────────
def _coerce_numeric_series(s: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(s): return pd.to_numeric(s, errors="coerce")
    if pd.api.types.is_bool_dtype(s): return s.astype(float)
    txt = s.astype("string").str.strip()
    txt = txt.replace({tok: pd.NA for tok in MISSING_TOKENS})
    txt = txt.str.replace(r"^\((.*)\)$", r"-\1", regex=True)
    txt = txt.str.replace(r"[$£€₹,]", "", regex=True)
    txt = txt.str.replace("%", "", regex=False)
    return pd.to_numeric(txt, errors="coerce")


def _numeric_like_ratio(s: pd.Series) -> float:
    nn = s.dropna()
    return 0.0 if len(nn) == 0 else float(_coerce_numeric_series(nn).notna().mean())


def _is_id_like_name(col) -> bool:
    name = str(col).strip().lower()
    return (name in {"id","uuid","guid","key","record","serial","index","row","transaction"}
            or name.endswith("_id") or name.endswith(" id")
            or name.startswith("id_") or name.startswith("id ")
            or "uuid" in name or "guid" in name)


def _is_monotonic_counter(s: pd.Series) -> bool:
    vals = pd.to_numeric(s, errors="coerce").dropna()
    if len(vals) < 3: return False
    diffs = vals.diff().dropna()
    return bool((diffs == 1).mean() > 0.98 or (diffs == -1).mean() > 0.98)


def _is_datetime_like(s: pd.Series) -> bool:
    sample = s.dropna().astype(str).head(100)
    if len(sample) < 3 or _numeric_like_ratio(sample) > 0.80: return False
    return bool(pd.to_datetime(sample, errors="coerce").notna().mean() >= 0.80)


def _is_text_like(s: pd.Series, unique_ratio: float) -> bool:
    sample = s.dropna().astype(str).head(250)
    if len(sample) == 0: return False
    return bool((sample.str.len().mean() >= 35 or sample.str.split().map(len).mean() >= 5) and unique_ratio >= 0.20)


# ── Schema detection (Kaggle-ready) ─────────────────────────────────────────
def _detect_schema(train_df, target_col, id_threshold=0.95):
    if target_col not in train_df.columns:
        raise ValueError(f"Target column '{target_col}' is not in the training file.")
    y = _coerce_numeric_series(train_df[target_col])
    if y.notna().mean() < 0.50:
        raise ValueError("Target must be numeric for regression.")
    raw = [c for c in train_df.columns if c != target_col]
    n = max(len(train_df), 1)
    all_missing = [c for c in raw if train_df[c].isna().all()]
    constant = [c for c in raw if c not in all_missing and train_df[c].nunique(dropna=True) <= 1]
    duplicates = []
    seen = {}
    for c in raw:
        if c in all_missing or c in constant: continue
        try:
            key = int(pd.util.hash_pandas_object(train_df[c], index=False).sum())
            if key in seen and train_df[c].equals(train_df[seen[key]]): duplicates.append(c)
            else: seen[key] = c
        except Exception: pass
    excluded = set(all_missing + constant + duplicates)
    avail = [c for c in raw if c not in excluded]
    dt_cols = [c for c in avail if pd.api.types.is_datetime64_any_dtype(train_df[c]) or _is_datetime_like(train_df[c])]
    num_cols, cat_cands, txt_cols, id_cols = [], [], [], []
    card = {}
    for c in avail:
        if c in dt_cols: continue
        nu = int(train_df[c].nunique(dropna=True)); ur = nu/n; card[c] = nu
        is_id = _is_id_like_name(c)
        if is_id and ur >= 0.50: id_cols.append(c); continue
        if _is_monotonic_counter(train_df[c]): id_cols.append(c); continue
        if pd.api.types.is_numeric_dtype(train_df[c]): num_cols.append(c)
        elif _is_text_like(train_df[c], ur): txt_cols.append(c)
        elif ur >= id_threshold and is_id: id_cols.append(c)
        else: cat_cands.append(c)
    hi = [c for c in cat_cands if card.get(c,0) > 30]
    lo = [c for c in cat_cands if c not in hi]
    cat = lo + hi
    feat = num_cols + cat + dt_cols + txt_cols
    dropped = all_missing + constant + duplicates + id_cols
    if not feat:
        raise ValueError("No usable feature columns remain.")
    return {"numeric_cols": num_cols, "low_card_cols": lo, "high_card_cols": hi,
            "categorical_cols": cat, "datetime_cols": dt_cols, "text_cols": txt_cols,
            "feature_cols": feat, "id_cols": id_cols, "all_missing_cols": all_missing,
            "constant_cols": constant, "duplicate_cols": duplicates,
            "dropped_cols": dropped, "cardinality": card}


def _safe_onehot():
    try: return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError: return OneHotEncoder(handle_unknown="ignore", sparse=False)


def _build_preprocessor(schema):
    t = []
    if schema["numeric_cols"]:
        t.append(("num", Pipeline([("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                                   ("scaler", RobustScaler())]), schema["numeric_cols"]))
    if schema["low_card_cols"]:
        t.append(("cat_low", Pipeline([("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
                                       ("onehot", _safe_onehot())]), schema["low_card_cols"]))
    if schema["high_card_cols"]:
        t.append(("cat_high", Pipeline([("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
                                        ("freq", FrequencyEncoder())]), schema["high_card_cols"]))
    if schema["datetime_cols"]:
        t.append(("date", Pipeline([("features", DateFeatureExtractor()),
                                    ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                                    ("scaler", RobustScaler())]), schema["datetime_cols"]))
    if schema["text_cols"]:
        t.append(("text", Pipeline([("stats", TextStatsTransformer()),
                                    ("imputer", SimpleImputer(strategy="median")),
                                    ("scaler", RobustScaler())]), schema["text_cols"]))
    return ColumnTransformer(transformers=t, remainder="drop", verbose_feature_names_out=False)


def _load_model(name):
    mod, cls, kw = MODEL_REGISTRY[name]
    return getattr(importlib.import_module(mod), cls)(**kw)


def _should_log_transform(y):
    vals = pd.to_numeric(y, errors="coerce").dropna()
    if len(vals) < 30 or (vals <= -1).any(): return False
    try: return bool(vals.min() >= 0 and vals.skew() > 1.5)
    except Exception: return False


def _wrap_target(model, use_log):
    if not use_log: return model
    return TransformedTargetRegressor(regressor=model, func=np.log1p, inverse_func=np.expm1)


def _strip_tuning_prefix(p):
    for pfx in ("model__regressor__", "model__"):
        if p.startswith(pfx): return p[len(pfx):]
    return p


def _grid_for_pipeline(grid, use_log):
    if not use_log: return grid
    return {("model__regressor__" + k[len("model__"):] if k.startswith("model__") and not k.startswith("model__regressor__") else k): v for k, v in grid.items()}


def _get_feature_names(preprocessor):
    try: return list(preprocessor.get_feature_names_out())
    except Exception: return []


def _metrics_dict(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    out = {"MAE": float(mean_absolute_error(y_true, y_pred)), "MSE": float(mse),
           "RMSE": float(np.sqrt(mse)), "R2": float(r2_score(y_true, y_pred))}
    try: out["MAPE %"] = float(mean_absolute_percentage_error(y_true, y_pred) * 100)
    except Exception: out["MAPE %"] = float("nan")
    return out


def _render_metrics(metrics, model_name, target_col, src):
    r2 = metrics["R2"]
    r2c = GREEN if r2 >= 0.8 else AMBER if r2 >= 0.5 else RED
    rows = [
        ("MAE", f"{metrics['MAE']:.4f}", PURPLE),
        ("MSE", f"{metrics['MSE']:.4f}", PURPLE),
        ("RMSE", f"{metrics['RMSE']:.4f}", PURPLE),
        ("R²", f"{metrics['R2']:.4f}", r2c),
    ]
    rows_html = "".join(
        f"<tr><td style='padding:9px 14px;color:#cbd5e1;font-size:12px;text-transform:uppercase;letter-spacing:1px;font-weight:500;'>{m}</td>"
        f"<td style='padding:9px 14px;color:{c};font-family:\"DM Mono\",monospace;font-size:16px;font-weight:700;text-align:right;'>{v}</td></tr>"
        for m, v, c in rows
    )
    return (
        "<div style='background:#13131f;border:1px solid #2a2a3e;border-radius:12px;overflow:hidden;'>"
        "<div style='background:#1a1a2e;padding:10px 14px;border-bottom:1px solid #2a2a3e;'>"
        f"<span style='color:#a78bfa;font-size:11px;font-weight:600;letter-spacing:2px;text-transform:uppercase;'>{model_name} · {target_col} · {src}</span></div>"
        f"<table style='width:100%;border-collapse:collapse;'>{rows_html}</table></div>"
    )


def _render_preprocessing_summary(summary):
    schema = summary.get("schema", {})
    card = schema.get("cardinality", {})
    lo = schema.get("low_card_cols", [])
    hi = schema.get("high_card_cols", [])
    txt = schema.get("text_cols", [])
    dt = schema.get("datetime_cols", [])
    all_m = schema.get("all_missing_cols", [])
    const = schema.get("constant_cols", [])
    dupes = schema.get("duplicate_cols", [])
    target_tx = summary.get("target_transform", "None")
    def _show(cols, max_n=12):
        if not cols: return "None"
        return ", ".join(str(c) for c in cols[:max_n]) + (" …" if len(cols) > max_n else "")
    rows = [
        ("Training shape", f"{summary['train_rows']:,} rows × {summary['train_cols']:,} cols"),
        ("Evaluation source", summary["eval_source"]),
        ("Numeric features", f"{len(summary['numeric_cols'])}: {_show(summary['numeric_cols'])}"),
        ("Low-cardinality categoricals", f"{len(lo)}: {_show(lo)}"),
        ("High-cardinality categoricals (freq-encoded)", f"{len(hi)}: {_show(hi)}"),
        ("Date/time columns (engineered)", f"{len(dt)}: {_show(dt)}"),
        ("Free-text columns (stats extracted)", f"{len(txt)}: {_show(txt)}"),
        ("Dropped ID-like columns", f"{len(schema.get('id_cols',[]))}: {_show(schema.get('id_cols',[]))}"),
        ("Dropped all-missing/constant/duplicate", f"{len(all_m)+len(const)+len(dupes)}: {_show(all_m+const+dupes)}"),
        ("Encoded feature count", f"{summary['encoded_feature_count']:,}"),
        ("Target transformation", target_tx),
        ("Preprocessing strategy",
         "Numeric: median impute + missing indicator + RobustScaler; "
         "Low-card cat: 'Unknown' impute + OneHotEncoder; "
         "High-card cat: FrequencyEncoder; "
         "Dates: calendar features; Text: stats features"),
        ("Leakage safeguard", "All transformers fitted inside the sklearn Pipeline on training data only."),
    ]
    html_rows = "".join(
        f"<tr><td style='padding:8px 12px;color:#94a3b8;width:28%;font-size:12px;font-weight:500;'>{k}</td>"
        f"<td style='padding:8px 12px;color:#e2e8f0;font-family:\"DM Mono\",monospace;font-size:12px;'>{v}</td></tr>"
        for k, v in rows
    )
    return (
        "<div style='background:#13131f;border:1px solid #2a2a3e;border-radius:12px;overflow:hidden;'>"
        "<div style='background:#1a1a2e;padding:10px 14px;border-bottom:1px solid #2a2a3e;'>"
        "<span style='color:#10b981;font-size:11px;font-weight:600;letter-spacing:2px;text-transform:uppercase;'>Preprocessing Summary</span></div>"
        f"<table style='width:100%;border-collapse:collapse;'>{html_rows}</table></div>"
    )


# ── Charts ──────────────────────────────────────────────────────────────────────
def _build_charts(y_true, y_pred, model_name, label="Evaluation"):
    """Returns a tuple (html_for_display, raw_png_bytes_for_download)."""
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    residuals = y_true - y_pred

    plt.rcParams.update({
        "figure.facecolor": DARK_BG, "axes.facecolor": DARK_AX,
        "axes.edgecolor": DARK_GRD, "axes.labelcolor": DARK_FG,
        "xtick.color": DARK_FG, "ytick.color": DARK_FG,
        "text.color": DARK_FG, "grid.color": DARK_GRD,
        "grid.linewidth": 0.5, "font.family": "monospace",
    })

    fig = plt.figure(figsize=(14, 10), facecolor=DARK_BG)
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.38)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.scatter(y_true, y_pred, alpha=0.55, s=18, color=PURPLE, edgecolors="none")
    mn, mx = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    ax1.plot([mn, mx], [mn, mx], color=CYAN, lw=1.5, ls="--", label="Perfect fit")
    ax1.set_xlabel("Actual"); ax1.set_ylabel("Predicted")
    ax1.set_title("Actual vs Predicted", fontsize=11, pad=8)
    ax1.legend(fontsize=8); ax1.grid(True, alpha=0.4)

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.scatter(y_pred, residuals, alpha=0.55, s=18, color=AMBER, edgecolors="none")
    ax2.axhline(0, color=CYAN, lw=1.5, ls="--")
    ax2.set_xlabel("Predicted"); ax2.set_ylabel("Residual")
    ax2.set_title("Residuals vs Predicted", fontsize=11, pad=8)
    ax2.grid(True, alpha=0.4)

    ax3 = fig.add_subplot(gs[0, 2])
    ax3.hist(residuals, bins=30, color=PURPLE, alpha=0.75, edgecolor=DARK_BG, linewidth=0.4)
    ax3.axvline(0, color=CYAN, lw=1.5, ls="--")
    ax3.set_xlabel("Residual"); ax3.set_ylabel("Count")
    ax3.set_title("Residual Distribution", fontsize=11, pad=8)
    ax3.grid(True, alpha=0.4)

    pct_err = np.abs(residuals) / (np.abs(y_true) + 1e-9) * 100
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.hist(np.clip(pct_err, 0, 200), bins=30, color=GREEN, alpha=0.75, edgecolor=DARK_BG, linewidth=0.4)
    ax4.axvline(np.median(pct_err), color=AMBER, lw=1.5, ls="--",
                label=f"Median {np.median(pct_err):.1f}%")
    ax4.set_xlabel("Abs Error %"); ax4.set_ylabel("Count")
    ax4.set_title("Error % Distribution", fontsize=11, pad=8)
    ax4.legend(fontsize=8); ax4.grid(True, alpha=0.4)

    ax5 = fig.add_subplot(gs[1, 1])
    sorted_err = np.sort(pct_err)
    cumulative = np.arange(1, len(sorted_err) + 1) / len(sorted_err) * 100
    ax5.plot(np.clip(sorted_err, 0, 200), cumulative, color=CYAN, lw=2)
    for th in [10, 25, 50]:
        pct_within = (pct_err <= th).mean() * 100
        ax5.axvline(th, color=AMBER, lw=0.8, ls=":", alpha=0.7)
        ax5.text(th + 1, 5, f"{pct_within:.0f}%\n≤{th}%", color=AMBER, fontsize=7, va="bottom")
    ax5.set_xlabel("Abs Error %"); ax5.set_ylabel("Cumulative %")
    ax5.set_title("Cumulative Error Curve", fontsize=11, pad=8)
    ax5.set_xlim(0, 200); ax5.set_ylim(0, 105); ax5.grid(True, alpha=0.4)

    ax6 = fig.add_subplot(gs[1, 2])
    metrics = _metrics_dict(y_true, y_pred)
    labels = ["MAE", "RMSE", "R²"]
    values = [metrics["MAE"], metrics["RMSE"], metrics["R2"]]
    colors = [PURPLE, AMBER, GREEN if metrics["R2"] >= 0.7 else RED]
    bars = ax6.barh(labels, values, color=colors, alpha=0.85, height=0.55)
    max_val = max([v for v in values if not np.isnan(v)] + [1])
    for bar, val in zip(bars, values):
        ax6.text(bar.get_width() * 1.02, bar.get_y() + bar.get_height() / 2,
                 f"{val:.4f}" if not np.isnan(val) else "N/A",
                 va="center", fontsize=9, color=DARK_FG)
    ax6.set_title(f"{model_name} · {label}", fontsize=10, pad=8)
    ax6.set_xlim(0, max_val * 1.25)
    ax6.grid(True, axis="x", alpha=0.4); ax6.invert_yaxis()

    fig.suptitle(f"Model Performance Dashboard — {model_name}",
                 color=DARK_FG, fontsize=13, y=1.01, fontweight="bold")

    # Capture PNG bytes BEFORE the helper closes the figure
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight",
                facecolor=DARK_BG, edgecolor="none")
    buf.seek(0)
    png_bytes = buf.read()
    b64 = base64.b64encode(png_bytes).decode()
    plt.close(fig)
    html = f'<img src="data:image/png;base64,{b64}" style="width:100%;border-radius:10px;margin-top:6px;">'
    return html, png_bytes


def _render_importances(pipeline, feature_names, model_name, top_n=15,
                        X_eval=None, y_eval=None):
    """Returns (html, png_bytes). Shows impurity-based importance (or abs
    coefficients) AND permutation importance side by side when X_eval/y_eval
    are provided. Permutation importance is more reliable when features are
    correlated because it measures the actual drop in model performance."""
    from sklearn.inspection import permutation_importance as _perm_imp

    model = pipeline.named_steps["model"]
    # Unwrap TransformedTargetRegressor if present
    if hasattr(model, "regressor_"):
        model = model.regressor_

    importances = None
    kind = None
    if hasattr(model, "feature_importances_"):
        importances = np.asarray(model.feature_importances_)
        kind = "Impurity-Based Importance"
    elif hasattr(model, "coef_"):
        importances = np.abs(np.asarray(model.coef_).flatten())
        kind = "Abs Coefficient"

    # Compute permutation importance if eval data available
    perm_result = None
    if X_eval is not None and y_eval is not None:
        try:
            perm_result = _perm_imp(
                pipeline, X_eval, y_eval,
                n_repeats=10, random_state=42, n_jobs=-1,
                scoring="neg_root_mean_squared_error",
            )
        except Exception:
            perm_result = None

    if importances is None and perm_result is None:
        return _empty_html("This model does not expose feature importances or coefficients."), None

    plt.rcParams.update({
        "figure.facecolor": DARK_BG, "axes.facecolor": DARK_AX,
        "axes.edgecolor": DARK_GRD, "axes.labelcolor": DARK_FG,
        "xtick.color": DARK_FG, "ytick.color": DARK_FG,
        "text.color": DARK_FG, "grid.color": DARK_GRD,
        "grid.linewidth": 0.5,
    })

    has_impurity = importances is not None
    has_perm = perm_result is not None
    ncols = (1 if has_impurity else 0) + (1 if has_perm else 0)
    fig, axes = plt.subplots(1, ncols, figsize=(10 * ncols, max(4, top_n * 0.38)),
                             facecolor=DARK_BG)
    if ncols == 1:
        axes = [axes]

    col_idx = 0

    # ── Left panel: impurity-based / coefficient ──
    if has_impurity:
        ax = axes[col_idx]; col_idx += 1
        n_feat = min(len(importances), len(feature_names))
        imp_vals = importances[:n_feat]
        imp_names = feature_names[:n_feat]
        idx = np.argsort(imp_vals)[::-1][:top_n]
        names_sorted = [imp_names[i] for i in idx]
        vals_sorted = imp_vals[idx]
        bars = ax.barh(names_sorted[::-1], vals_sorted[::-1],
                       color=PURPLE, alpha=0.85, height=0.65)
        for bar, val in zip(bars, vals_sorted[::-1]):
            ax.text(bar.get_width() + (vals_sorted.max() or 1) * 0.01,
                    bar.get_y() + bar.get_height() / 2,
                    f"{val:.4f}", va="center", fontsize=8, color=DARK_FG)
        ax.set_xlabel(kind)
        ax.set_title(f"{kind} — {model_name}", fontsize=11, pad=8)
        ax.grid(True, axis="x", alpha=0.4)
        ax.set_xlim(0, (vals_sorted.max() or 1) * 1.18)

    # ── Right panel: permutation importance ──
    if has_perm:
        ax = axes[col_idx]
        # Permutation importance uses original feature names from X_eval
        perm_names = list(X_eval.columns) if hasattr(X_eval, "columns") else [
            f"feat_{i}" for i in range(len(perm_result.importances_mean))
        ]
        perm_means = perm_result.importances_mean
        perm_stds = perm_result.importances_std
        n_perm = min(len(perm_means), len(perm_names))
        perm_means = perm_means[:n_perm]
        perm_stds = perm_stds[:n_perm]
        perm_names = perm_names[:n_perm]
        pidx = np.argsort(perm_means)[::-1][:top_n]
        p_names = [perm_names[i] for i in pidx]
        p_vals = perm_means[pidx]
        p_errs = perm_stds[pidx]
        bars = ax.barh(p_names[::-1], p_vals[::-1],
                       xerr=p_errs[::-1],
                       color=CYAN, alpha=0.85, height=0.65,
                       error_kw={"ecolor": "#475569", "capsize": 3, "lw": 1})
        for bar, val in zip(bars, p_vals[::-1]):
            ax.text(bar.get_width() + (p_vals.max() or 1) * 0.01,
                    bar.get_y() + bar.get_height() / 2,
                    f"{val:.4f}", va="center", fontsize=8, color=DARK_FG)
        ax.set_xlabel("Δ RMSE when shuffled")
        ax.set_title(f"Permutation Importance — {model_name}", fontsize=11, pad=8)
        ax.grid(True, axis="x", alpha=0.4)
        ax.set_xlim(0, (p_vals.max() or 1) * 1.25)

    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight",
                facecolor=DARK_BG, edgecolor="none")
    buf.seek(0)
    png_bytes = buf.read()
    b64 = base64.b64encode(png_bytes).decode()
    plt.close(fig)

    note = ""
    if has_impurity and has_perm:
        note = (
            '<p style="color:#94a3b8;font-size:11px;margin:8px 0 0;line-height:1.5;">'
            '💡 <b>Left:</b> impurity-based importance (how often a feature is used for '
            'tree splits — can be misleading when features are correlated). '
            '<b>Right:</b> permutation importance (how much RMSE actually worsens when '
            'each feature is shuffled — more reliable for correlated features).</p>'
        )

    html = (f'<img src="data:image/png;base64,{b64}" '
            f'style="width:100%;border-radius:10px;margin-top:6px;">{note}')
    return html, png_bytes


def _build_comparison_charts(comparison_df: pd.DataFrame, preds_by_model: dict,
                             y_eval) -> Tuple[str, Optional[bytes]]:
    """Side-by-side comparison of all selected models on MAE / RMSE / R²,
    plus an Actual-vs-Predicted overlay.
    Returns (html, png_bytes). If only one model is in comparison_df,
    returns an empty placeholder (the per-model dashboard already covers it).
    """
    if comparison_df is None or len(comparison_df) < 2:
        return _empty_html(
            "Select two or more models to see a side-by-side comparison."
        ), None

    plt.rcParams.update({
        "figure.facecolor": DARK_BG, "axes.facecolor": DARK_AX,
        "axes.edgecolor": DARK_GRD, "axes.labelcolor": DARK_FG,
        "xtick.color": DARK_FG, "ytick.color": DARK_FG,
        "text.color": DARK_FG, "grid.color": DARK_GRD,
        "grid.linewidth": 0.5, "font.family": "monospace",
    })

    cmp = comparison_df.copy()
    fig = plt.figure(figsize=(14, 10), facecolor=DARK_BG)
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.55, wspace=0.40)

    model_labels = cmp["Model"].tolist()
    n = len(model_labels)
    palette = [PURPLE, CYAN, GREEN, AMBER, RED, "#f472b6", "#60a5fa",
               "#fbbf24", "#34d399", "#c084fc", "#fb7185", "#22d3ee",
               "#a3e635", "#fde047"]
    colors = [palette[i % len(palette)] for i in range(n)]

    # Bar chart: MAE
    ax1 = fig.add_subplot(gs[0, 0])
    bars = ax1.bar(range(n), cmp["MAE"].values, color=colors, alpha=0.85)
    for bar, v in zip(bars, cmp["MAE"].values):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                 f"{v:.3f}", ha="center", va="bottom",
                 fontsize=8, color=DARK_FG)
    ax1.set_xticks(range(n))
    ax1.set_xticklabels(model_labels, rotation=35, ha="right", fontsize=8)
    ax1.set_title("MAE (lower is better)", fontsize=11, pad=8)
    ax1.grid(True, axis="y", alpha=0.4)

    # Bar chart: RMSE
    ax2 = fig.add_subplot(gs[0, 1])
    bars = ax2.bar(range(n), cmp["RMSE"].values, color=colors, alpha=0.85)
    for bar, v in zip(bars, cmp["RMSE"].values):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                 f"{v:.3f}", ha="center", va="bottom",
                 fontsize=8, color=DARK_FG)
    ax2.set_xticks(range(n))
    ax2.set_xticklabels(model_labels, rotation=35, ha="right", fontsize=8)
    ax2.set_title("RMSE (lower is better)", fontsize=11, pad=8)
    ax2.grid(True, axis="y", alpha=0.4)

    # Bar chart: R²
    ax3 = fig.add_subplot(gs[0, 2])
    r2_vals = cmp["R2"].values
    bar_colors = [GREEN if v >= 0.7 else AMBER if v >= 0.4 else RED
                  for v in r2_vals]
    bars = ax3.bar(range(n), r2_vals, color=bar_colors, alpha=0.85)
    for bar, v in zip(bars, r2_vals):
        ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                 f"{v:.3f}", ha="center", va="bottom",
                 fontsize=8, color=DARK_FG)
    ax3.set_xticks(range(n))
    ax3.set_xticklabels(model_labels, rotation=35, ha="right", fontsize=8)
    ax3.set_title("R² (higher is better)", fontsize=11, pad=8)
    ax3.set_ylim(min(0, r2_vals.min() * 1.1), 1.05)
    ax3.axhline(0, color=DARK_FG, lw=0.5, ls=":")
    ax3.grid(True, axis="y", alpha=0.4)

    # Actual vs predicted overlay (one scatter per model)
    ax4 = fig.add_subplot(gs[1, 0:2])
    y_arr = np.asarray(y_eval, dtype=float)
    for i, m in enumerate(model_labels):
        if m not in preds_by_model:
            continue
        p = np.asarray(preds_by_model[m], dtype=float)
        ax4.scatter(y_arr, p, alpha=0.45, s=14, color=colors[i],
                    edgecolors="none", label=m)
    mn, mx = float(y_arr.min()), float(y_arr.max())
    ax4.plot([mn, mx], [mn, mx], color=DARK_FG, lw=1.2, ls="--",
             alpha=0.7, label="Perfect fit")
    ax4.set_xlabel("Actual"); ax4.set_ylabel("Predicted")
    ax4.set_title("Actual vs Predicted — all selected models", fontsize=11, pad=8)
    ax4.legend(fontsize=8, ncol=2, loc="best")
    ax4.grid(True, alpha=0.4)

    # Residual distribution overlay
    ax5 = fig.add_subplot(gs[1, 2])
    for i, m in enumerate(model_labels):
        if m not in preds_by_model:
            continue
        residuals = y_arr - np.asarray(preds_by_model[m], dtype=float)
        ax5.hist(residuals, bins=25, color=colors[i], alpha=0.45,
                 label=m, edgecolor=DARK_BG, linewidth=0.4)
    ax5.axvline(0, color=DARK_FG, lw=1.0, ls="--", alpha=0.7)
    ax5.set_xlabel("Residual"); ax5.set_ylabel("Count")
    ax5.set_title("Residual Distribution Overlay", fontsize=11, pad=8)
    ax5.legend(fontsize=7)
    ax5.grid(True, alpha=0.4)

    fig.suptitle(f"Cross-Model Comparison ({n} models)",
                 color=DARK_FG, fontsize=13, y=1.01, fontweight="bold")

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight",
                facecolor=DARK_BG, edgecolor="none")
    buf.seek(0)
    png_bytes = buf.read()
    b64 = base64.b64encode(png_bytes).decode()
    plt.close(fig)
    html = (f'<img src="data:image/png;base64,{b64}" '
            f'style="width:100%;border-radius:10px;margin-top:6px;">')
    return html, png_bytes


def _render_sample(pipeline, X_eval_raw, y_eval, target_col, row_idx: int = 0):
    if X_eval_raw is None or len(X_eval_raw) == 0:
        return _empty_html("Sample prediction unavailable.")
    row_idx = max(0, min(int(row_idx), len(X_eval_raw) - 1))
    row = X_eval_raw.iloc[[row_idx]]
    actual = float(y_eval.iloc[row_idx])
    predicted = float(pipeline.predict(row)[0])
    err_pct = abs(actual - predicted) / (abs(actual) + 1e-9) * 100
    err_col = GREEN if err_pct < 10 else AMBER if err_pct < 25 else RED
    feat_rows = "".join(
        f"<tr><td style='padding:5px 10px;color:#64748b;font-size:11px;font-family:\"DM Mono\",monospace;'>{k}</td>"
        f"<td style='padding:5px 10px;color:#cbd5e1;font-size:11px;font-family:\"DM Mono\",monospace;text-align:right;'>{v}</td></tr>"
        for k, v in row.iloc[0].head(10).items()
    )
    more = (f"<p style='color:#475569;font-size:10px;padding:2px 10px;'>"
            f"…and {row.shape[1] - 10} more</p>") if row.shape[1] > 10 else ""
    return (
        "<div style='background:#13131f;border:1px solid #2a2a3e;border-radius:12px;overflow:hidden;'>"
        "<div style='background:#1a1a2e;padding:10px 14px;border-bottom:1px solid #2a2a3e;'>"
        f"<span style='color:#06b6d4;font-size:11px;font-weight:600;letter-spacing:2px;text-transform:uppercase;'>"
        f"Sample Prediction · Row {row_idx} · Target: {target_col}</span></div>"
        "<div style='display:flex;border-bottom:1px solid #2a2a3e;'>"
        f"<div style='flex:1;padding:16px 18px;border-right:1px solid #2a2a3e;'>"
        f"<div style='color:#64748b;font-size:10px;text-transform:uppercase;margin-bottom:6px;'>Actual</div>"
        f"<div style='color:{GREEN};font-size:24px;font-weight:700;font-family:\"DM Mono\",monospace;'>{actual:.4f}</div></div>"
        f"<div style='flex:1;padding:16px 18px;border-right:1px solid #2a2a3e;'>"
        f"<div style='color:#64748b;font-size:10px;text-transform:uppercase;margin-bottom:6px;'>Predicted</div>"
        f"<div style='color:#a78bfa;font-size:24px;font-weight:700;font-family:\"DM Mono\",monospace;'>{predicted:.4f}</div></div>"
        f"<div style='flex:1;padding:16px 18px;'>"
        f"<div style='color:#64748b;font-size:10px;text-transform:uppercase;margin-bottom:6px;'>Error %</div>"
        f"<div style='color:{err_col};font-size:24px;font-weight:700;font-family:\"DM Mono\",monospace;'>{err_pct:.1f}%</div></div>"
        "</div><div style='padding:10px 4px 6px;'>"
        "<div style='color:#64748b;font-size:10px;text-transform:uppercase;margin:0 10px 6px;'>Raw Input Features</div>"
        f"<table style='width:100%;border-collapse:collapse;'>{feat_rows}</table>{more}</div></div>"
    )


def _render_test_download(pipeline, test_df, feature_cols, id_col_choice: Optional[str]):
    """Predict on test set, preserving an id column if user picked one."""
    if test_df is None:
        return _empty_html("No test file was uploaded."), None
    X_test_final = test_df.reindex(columns=feature_cols)
    preds = np.asarray(pipeline.predict(X_test_final)).flatten()

    if id_col_choice and id_col_choice != "(none)" and id_col_choice in test_df.columns:
        out_df = pd.DataFrame({
            id_col_choice: test_df[id_col_choice].values,
            "prediction": preds,
        })
    else:
        out_df = pd.DataFrame({"id": np.arange(len(preds)), "prediction": preds})

    path = os.path.join(tempfile.gettempdir(), "submission.csv")
    out_df.to_csv(path, index=False)
    return _html_table(out_df, max_rows=8), path


def _save_model_to_disk(pipeline) -> str:
    path = os.path.join(
        tempfile.gettempdir(),
        f"fitted_pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.joblib",
    )
    joblib.dump(pipeline, path)
    return path


def _create_code_file(code_text: str) -> str:
    path = os.path.join(
        tempfile.gettempdir(),
        f"generated_pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.py",
    )
    cleaned = code_text.replace("```python", "").replace("```", "")
    with open(path, "w", encoding="utf-8") as f:
        f.write(cleaned)
    return path


def _bundle_artifacts_zip(paths_and_bytes: List[Tuple[str, Any]]) -> Optional[str]:
    """paths_and_bytes is a list of (filename, content). content is either
    raw bytes or a filesystem path (str)."""
    if not paths_and_bytes:
        return None
    zip_path = os.path.join(
        tempfile.gettempdir(),
        f"automl_artifacts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
    )
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for fname, content in paths_and_bytes:
            if content is None:
                continue
            if isinstance(content, (bytes, bytearray)):
                zf.writestr(fname, bytes(content))
            elif isinstance(content, str) and os.path.exists(content):
                zf.write(content, arcname=fname)
    return zip_path


def _format_kwargs(kwargs: Dict[str, Any]) -> str:
    if not kwargs:
        return ""
    return ", ".join(f"{k}={v!r}" for k, v in kwargs.items())


def _build_reproducible_code(
    target_col: str,
    selected_models: List[str],
    numeric_cols: List[str],
    categorical_cols: List[str],
    high_card_cols: List[str],
    datetime_cols: List[str],
    id_cols: List[str],
    has_valid: bool,
    has_test: bool,
    test_size: float,
    shuffle: bool,
    random_state: int,
    id_col_choice: Optional[str],
    train_path: Optional[str],
    valid_path: Optional[str],
    test_path: Optional[str],
) -> str:
    """Generate a complete, dataset-specific, runnable Python script.

    Decisions made by the app's preprocessing agent (which columns are
    numeric, categorical, datetime, id-like) are baked into the generated
    code as concrete lists, so the script reproduces exactly what the app
    did — no auto-detection at runtime, no guessing.

    The script is self-contained: it loads the user's CSV files, builds
    the preprocessing pipeline, trains every selected model, evaluates
    them, prints a comparison table, and saves the best fitted pipeline
    to disk. If a test file was uploaded, it also writes submission.csv.
    """
    # --- collect imports needed for the selected models ---
    import_block_lines = set()
    model_dict_lines = []
    for name in selected_models:
        mod_path, cls_name, kwargs = MODEL_REGISTRY[name]
        import_block_lines.add(f"from {mod_path} import {cls_name}")
        kw_str = _format_kwargs(kwargs)
        model_dict_lines.append(f"    {name!r}: {cls_name}({kw_str}),")
    model_imports = "\n".join(sorted(import_block_lines))
    models_dict = "\n".join(model_dict_lines)

    # --- file path placeholders (use real names if available) ---
    train_p = train_path or "train.csv"
    valid_p = valid_path or "valid.csv"
    test_p = test_path or "test.csv"

    # --- decide which onehot encoder to declare ---
    needs_high_card = bool(high_card_cols)
    high_card_block = ""
    if needs_high_card:
        high_card_block = f"""
# High-cardinality categoricals (>20 unique values in this dataset) use a
# OneHotEncoder with min_frequency=0.01 to group rare categories together.
HIGH_CARDINALITY_COLS = {high_card_cols!r}
LOW_CARDINALITY_COLS = [c for c in CATEGORICAL_COLS if c not in HIGH_CARDINALITY_COLS]
"""
    else:
        high_card_block = """
HIGH_CARDINALITY_COLS = []
LOW_CARDINALITY_COLS = CATEGORICAL_COLS
"""

    # --- preprocessor build block ---
    transformers_lines = []
    if numeric_cols:
        transformers_lines.append("""
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
        ]), NUMERIC_COLS),""")
    transformers_lines.append("""
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
            ('onehot', make_onehot_encoder()),
        ]), LOW_CARDINALITY_COLS),""")
    if needs_high_card:
        transformers_lines.append("""
        ('cat_highcard', Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
            ('onehot', make_high_card_encoder()),
        ]), HIGH_CARDINALITY_COLS),""")
    transformers_block = "".join(transformers_lines)

    # --- evaluation source: validation file vs split ---
    if has_valid:
        eval_block = f"""
# ── Evaluation source: a validation CSV/Excel file you uploaded.
valid_df = _read({valid_p!r})
valid_df = clean_df(valid_df)
X_eval_raw = valid_df.reindex(columns=FEATURE_COLS).copy()
y_eval = pd.to_numeric(valid_df[TARGET_COL], errors='coerce')
keep = y_eval.notna()
X_eval_raw = X_eval_raw.loc[keep]
y_eval = y_eval.loc[keep]
X_train_raw_final = X_train_raw.copy()
y_train_final = y_train.copy()
print(f'Eval rows from validation file: {{len(X_eval_raw):,}}')
"""
    else:
        rs = random_state if shuffle else None
        eval_block = f"""
# ── Evaluation source: hold out {int(float(test_size) * 100)}% of the training data.
X_train_raw_final, X_eval_raw, y_train_final, y_eval = train_test_split(
    X_train_raw, y_train,
    test_size={float(test_size)},
    shuffle={bool(shuffle)},
    random_state={rs},
)
print(f'Train rows: {{len(X_train_raw_final):,}}, Eval rows: {{len(X_eval_raw):,}}')
"""

    # --- test file block ---
    if has_test:
        if id_col_choice and id_col_choice != "(none)":
            test_block = f"""
# ── Predict on the uploaded test CSV/Excel and write submission.csv ──
test_df = _read({test_p!r})
test_df = clean_df(test_df)
X_test = test_df.reindex(columns=FEATURE_COLS).copy()
test_predictions = best_pipeline.predict(X_test)
submission = pd.DataFrame({{
    {id_col_choice!r}: test_df[{id_col_choice!r}].values,
    'prediction': test_predictions,
}})
submission.to_csv('submission.csv', index=False)
print(f'submission.csv written ({{len(submission):,}} rows).')
"""
        else:
            test_block = f"""
# ── Predict on the uploaded test CSV/Excel and write submission.csv ──
test_df = _read({test_p!r})
test_df = clean_df(test_df)
X_test = test_df.reindex(columns=FEATURE_COLS).copy()
test_predictions = best_pipeline.predict(X_test)
submission = pd.DataFrame({{'id': range(len(test_predictions)), 'prediction': test_predictions}})
submission.to_csv('submission.csv', index=False)
print(f'submission.csv written ({{len(submission):,}} rows).')
"""
    else:
        test_block = "# (No final test file was uploaded; skipping submission step.)"

    n_total = len(numeric_cols) + len(categorical_cols)
    return f'''"""
Reproducible regression pipeline — generated by CrewAI AutoML.
Run from a terminal with: python this_script.py

Dataset-specific decisions baked in:
  - Target column          : {target_col!r}
  - Numeric features  ({len(numeric_cols):>2}): {numeric_cols}
  - Categorical features ({len(categorical_cols):>2}): {categorical_cols}
  - High-cardinality cats  : {high_card_cols}
  - Dropped datetime cols  : {datetime_cols}
  - Dropped ID-like cols   : {id_cols}
  - Total kept features    : {n_total}
  - Models trained         : {selected_models}

This script does NOT re-detect column types — those decisions were already
made by the AutoML preprocessing agent. To use a different dataset, run the
app on it; it will produce a new, dataset-specific script.
"""
import os
import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    mean_absolute_percentage_error,
)
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    mean_absolute_percentage_error,
)
from sklearn.model_selection import (
    KFold,
    RandomizedSearchCV,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler

{model_imports}


# ════════════════════════ DATASET SCHEMA (from the app) ═══════════════════════
TARGET_COL       = {target_col!r}
NUMERIC_COLS     = {numeric_cols!r}
CATEGORICAL_COLS = {categorical_cols!r}
DATETIME_COLS    = {datetime_cols!r}   # detected and dropped
ID_LIKE_COLS     = {id_cols!r}          # dropped due to high uniqueness
FEATURE_COLS     = NUMERIC_COLS + CATEGORICAL_COLS
{high_card_block}

# ════════════════════════ HELPERS ═════════════════════════════════════════════
def _read(path):
    """Read a CSV / Excel / Parquet / JSON file into a DataFrame."""
    ext = os.path.splitext(path)[-1].lower()
    if ext == '.csv':
        return pd.read_csv(path)
    if ext in ('.xlsx', '.xls'):
        return pd.read_excel(path)
    if ext == '.parquet':
        return pd.read_parquet(path)
    if ext == '.json':
        return pd.read_json(path)
    raise ValueError(f'Unsupported file format: {{ext}}')


def clean_df(df):
    """Drop pandas-default 'Unnamed: 0' index columns and convert booleans
    to int 0/1 — matches what the AutoML app does at upload time."""
    drop = [c for c in df.columns
            if isinstance(c, str) and c.startswith('Unnamed:')
            and (df[c].isna().all()
                 or (pd.to_numeric(df[c], errors='coerce').notna().all()
                     and (pd.to_numeric(df[c], errors='coerce').diff().dropna() == 1).all()))]
    if drop:
        df = df.drop(columns=drop)
    for c in df.columns:
        if pd.api.types.is_bool_dtype(df[c]):
            df[c] = df[c].astype(int)
    return df


def make_onehot_encoder():
    """Standard OneHotEncoder with handle_unknown='ignore' for unseen
    categories at predict time."""
    try:
        return OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown='ignore', sparse=False)


def make_high_card_encoder():
    """OneHotEncoder for high-cardinality categoricals — uses min_frequency
    so rare categories collapse into a single bucket."""
    try:
        return OneHotEncoder(
            handle_unknown='infrequent_if_exist',
            sparse_output=False,
            min_frequency=0.01,
        )
    except TypeError:
        try:
            return OneHotEncoder(handle_unknown='ignore', sparse_output=False, min_frequency=0.01)
        except TypeError:
            return make_onehot_encoder()


# ════════════════════════ LOAD DATA ═══════════════════════════════════════════
train_df = _read({train_p!r})
train_df = clean_df(train_df)
print(f'Loaded training data: {{len(train_df):,}} rows × {{len(train_df.columns)}} cols.')

# Target -> numeric, drop rows with missing target.
y_train = pd.to_numeric(train_df[TARGET_COL], errors='coerce')
keep = y_train.notna()
X_train_raw = train_df.loc[keep, FEATURE_COLS].copy()
y_train = y_train.loc[keep]
print(f'After dropping rows with missing target: {{len(X_train_raw):,}} rows.')

{eval_block}

# ════════════════════════ PREPROCESSING + MODELS ══════════════════════════════
preprocessor = ColumnTransformer(
    transformers=[{transformers_block}
    ],
    remainder='drop',
    verbose_feature_names_out=False,
)

models = {{
{models_dict}
}}

# ════════════════════════ TRAIN + EVALUATE EACH MODEL ═════════════════════════
results = []
fitted_pipelines = {{}}
for model_label, model in models.items():
    pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('model', model),
    ])
    pipe.fit(X_train_raw_final, y_train_final)
    pred = pipe.predict(X_eval_raw)
    fitted_pipelines[model_label] = pipe
    try:
        mape = mean_absolute_percentage_error(y_eval, pred) * 100
    except Exception:
        mape = float('nan')
    results.append({{
        'Model': model_label,
        'MAE':   mean_absolute_error(y_eval, pred),
        'MSE':   mean_squared_error(y_eval, pred),
        'RMSE':  float(np.sqrt(mean_squared_error(y_eval, pred))),
        'R2':    r2_score(y_eval, pred),
        'MAPE %': mape,
    }})

comparison = pd.DataFrame(results).sort_values('RMSE').reset_index(drop=True)
print()
print('Model comparison (sorted by RMSE):')
print(comparison.to_string(index=False))

# Best model = lowest RMSE.
best_model_name = comparison.iloc[0]['Model']
best_pipeline = fitted_pipelines[best_model_name]
print()
print(f'Best model by RMSE: {{best_model_name}}')

# ════════════════════════ SAMPLE PREDICTION ═══════════════════════════════════
sample = X_eval_raw.iloc[[0]]
print()
print('Sample row (first eval row):')
print(sample.to_string(index=False))
print(f'Actual    : {{y_eval.iloc[0]:.4f}}')
print(f'Predicted : {{best_pipeline.predict(sample)[0]:.4f}}')

# ════════════════════════ SAVE BEST FITTED PIPELINE ═══════════════════════════
joblib.dump(best_pipeline, 'best_pipeline.joblib')
print()
print('Saved fitted preprocessing+model pipeline to best_pipeline.joblib')
print('Re-load with:  joblib.load("best_pipeline.joblib")')

{test_block}
'''


# ── CrewAI review (now with executable code tool) ────────────────────────────────
def _local_agent_report(agent_plan: dict, summary: dict, comparison_df: pd.DataFrame,
                        model_name: str, target_col: str) -> str:
    """Deterministic fallback report so the app remains usable even without an API key."""
    best_row = comparison_df.sort_values("RMSE").iloc[0] if comparison_df is not None and len(comparison_df) else None
    primary_row = None
    if comparison_df is not None and len(comparison_df) and model_name in comparison_df["Model"].values:
        primary_row = comparison_df.loc[comparison_df["Model"] == model_name].iloc[0]

    best_text = "N/A" if best_row is None else (
        f"{best_row['Model']} by lowest RMSE ({best_row['RMSE']:.4f}); "
        f"R²={best_row['R2']:.4f}, MAE={best_row['MAE']:.4f}."
    )
    primary_text = "N/A" if primary_row is None else (
        f"{model_name}: RMSE={primary_row['RMSE']:.4f}, R²={primary_row['R2']:.4f}, "
        f"MAE={primary_row['MAE']:.4f}."
    )

    return (
        "## Multi-Agent Workflow Output\n"
        "### Planner Agent\n"
        f"- Objective: regression modelling for target `{target_col}`.\n"
        f"- Plan: {agent_plan.get('plan_text', 'Prepare data, train selected models, evaluate, review.')}\n"
        f"- Selected models: {', '.join(agent_plan.get('selected_models', []))}.\n\n"
        "### Data Preprocessing Agent\n"
        f"- Numeric features kept: {len(summary.get('numeric_cols', []))}.\n"
        f"- Categorical features kept: {len(summary.get('categorical_cols', []))}.\n"
        f"- Dropped datetime columns: {summary.get('datetime_cols', []) or 'None'}.\n"
        f"- Dropped ID-like columns: {summary.get('id_cols', []) or 'None'}.\n"
        "- Leakage safeguard: imputers, scaler, and encoder are fitted only inside the training pipeline.\n\n"
        "### Modeler Evaluator Agent\n"
        f"- Best model: {best_text}\n"
        f"- Primary model displayed in the UI: {primary_text}\n"
        f"- Evaluation source: {summary.get('eval_source')}.\n\n"
        "### Reviewer Agent\n"
        "- The generated code includes preprocessing, modelling, evaluation, sample prediction, and optional test-set prediction steps.\n"
        "- Concrete next improvement: add cross-validation and hyperparameter search for the top 2-3 models.\n"
    )


def _run_crewai_review(api_key: str, openai_model: str, model_name: str,
                       target_col: str, summary: dict, comparison_df: pd.DataFrame,
                       fitted_pipeline, X_eval_raw: pd.DataFrame, y_eval: pd.Series,
                       agent_plan: Optional[dict] = None,
                       use_executor: bool = True) -> str:
    """Run a detailed four-agent CrewAI workflow.

    Important design choice:
    The deterministic Python functions in this app still execute the actual data
    preparation, model training, chart creation, artifact saving, and prediction.
    The agents then plan, audit, validate, and explain the completed workflow.
    This keeps the app reliable while still making the project clearly agent-based.
    """

    agent_plan = agent_plan or {}
    if not api_key:
        return (
            "ℹ️ CrewAI LLM execution skipped because no API key was provided.\n"
            "The app still used the same four-stage agent workflow locally.\n\n"
            + _local_agent_report(agent_plan, summary, comparison_df, model_name, target_col)
        )

    try:
        from crewai import Agent, Task, Crew, Process, LLM
    except Exception as e:
        return (
            f"❌ CrewAI unavailable: {type(e).__name__}: {e}\n\n"
            + _local_agent_report(agent_plan, summary, comparison_df, model_name, target_col)
        )

    try:
        llm = LLM(model=openai_model, api_key=api_key.strip())

        tools = []
        executor_note = ""
        if use_executor and _HAS_NB_EXECUTOR:
            ns: Dict[str, Any] = {
                "pd": pd,
                "np": np,
                "trained_model": fitted_pipeline,
                "best_pipeline": fitted_pipeline,
                "X_eval": X_eval_raw,
                "y_eval": y_eval,
                "comparison": comparison_df,
                "summary": summary,
            }
            tools.append(NotebookCodeExecutor(namespace=ns))
            executor_note = (
                "\n\nAVAILABLE CODE EXECUTION TOOL:\n"
                "- You have access to the Notebook Code Executor tool.\n"
                "- The shared namespace already contains: `pd`, `np`, `best_pipeline`, "
                "`trained_model`, `X_eval`, `y_eval`, `comparison`, and `summary`.\n"
                "- Use the tool for verification, not for unsupported speculation.\n"
                "- At minimum, the Reviewer Agent must execute one short diagnostic, "
                "for example residual mean, residual standard deviation, residual quantiles, "
                "or re-checking the best row in the comparison table.\n"
                "- Include print() statements in the executed code so the result is visible.\n"
            )

        selected_models_txt = ", ".join(agent_plan.get("selected_models", [])) or model_name
        comparison_txt = comparison_df.to_string(index=False) if comparison_df is not None else "No comparison table available."

        planner_agent = Agent(
            role="Lead Data Science Planner Agent",
            goal=(
                "Create a complete, step-by-step regression analysis plan for the uploaded dataset. "
                "The plan must explicitly guide the downstream agents through objective clarification, "
                "data inspection, preprocessing, train/evaluation design, model training, model comparison, "
                "diagnostic checking, reproducible-code generation, artifact creation, and final review. "
                "The planner must prevent common workflow mistakes, including training before preprocessing, "
                "using the target as a feature, fitting preprocessing on evaluation/test data, ignoring missing targets, "
                "forgetting validation/test handling, and failing to preserve the selected ID column for test predictions."
            ),
            backstory=(
                "You are a senior AutoML project planner with deep experience converting messy tabular datasets "
                "into reliable, reproducible regression pipelines. You are extremely careful about sequencing. "
                "You know that this Gradio app has already loaded data into pandas DataFrames and that the app's "
                "deterministic Python workflow produces UI outputs including preprocessing summary, metrics, charts, "
                "comparison table, generated code, sample prediction, optional submission file, fitted model, and ZIP bundle. "
                "Your job is to make sure every downstream agent understands the exact purpose of its work and does not "
                "skip any critical step. You write instructions that are precise enough for agents to follow without guessing."
            ),
            llm=llm,
            allow_delegation=False,
            verbose=False,
        )

        preprocessing_agent = Agent(
            role="Data Analysis and Preprocessing Agent",
            goal=(
                "Audit and explain the complete preprocessing workflow used before model training. "
                "Confirm the target column, identify usable feature columns, separate numeric and categorical variables, "
                "detect and remove datetime/date-like columns from modelling features, detect and remove high-cardinality "
                "ID-like categorical columns based on the configured threshold, remove rows with missing target values, "
                "describe how uploaded validation data is aligned to the training feature schema, and explain how optional "
                "test data is reindexed to the same feature list before prediction. "
                "Verify that numeric features use median imputation followed by StandardScaler, categorical features use "
                "constant 'Unknown' imputation followed by OneHotEncoder(handle_unknown='ignore'), and all preprocessing "
                "steps are kept inside a scikit-learn Pipeline/ColumnTransformer to reduce data leakage."
            ),
            backstory=(
                "You are a meticulous data preprocessing specialist. Your main responsibility is to make the data ready "
                "for machine learning without leakage or inconsistent feature schemas. You understand pandas, scikit-learn, "
                "ColumnTransformer, SimpleImputer, StandardScaler, OneHotEncoder, and Pipeline. You are cautious about "
                "date columns, ID columns, string categories, missing values, unseen categories in validation/test data, "
                "and target leakage. You always check what was dropped, what was kept, what was encoded, and how the final "
                "training and evaluation matrices were created. You must report preprocessing in a way that is visible to "
                "the user so the generated output does not only show model training code but also the preprocessing code/logic."
            ),
            llm=llm,
            tools=tools,
            allow_delegation=False,
            verbose=False,
        )

        modeler_evaluator_agent = Agent(
            role="Machine Learning Modeler and Evaluator Agent",
            goal=(
                "Review the model training and evaluation workflow for all selected regression models. "
                "Confirm that every selected model is instantiated from the model registry, wrapped inside the same "
                "preprocessing pipeline, fitted only on the training split or training file, evaluated on the validation file "
                "or holdout split, and compared using MAE, MSE, RMSE, R², and MAPE percentage where available. "
                "Identify the best model by lowest RMSE, summarize the primary UI model's performance, and explain the "
                "purpose of generated outputs: model comparison table, performance dashboard, residual plots, actual-vs-predicted "
                "plot, error percentage distribution, cumulative error curve, feature importance/coefficient plot when available, "
                "sample prediction card, optional test-set submission, downloadable fitted pipeline, generated code file, and ZIP bundle."
            ),
            backstory=(
                "You are a practical machine-learning engineer who specializes in regression modelling. You know that "
                "a fair comparison requires identical preprocessing for all models and a consistent evaluation dataset. "
                "You care about transparent metrics, residual diagnostics, reproducibility, and user-facing artifacts. "
                "You do not overclaim performance. You interpret RMSE, MAE, and R² carefully and point out when the model "
                "needs cross-validation, hyperparameter tuning, additional feature engineering, or domain review. "
                "You make sure the modelling step does not ignore the preprocessing step."
            ),
            llm=llm,
            tools=tools,
            allow_delegation=False,
            verbose=False,
        )

        reviewer_agent = Agent(
            role="Senior ML Reviewer Agent",
            goal=(
                "Perform the final quality review of the complete agent-based AutoML regression workflow. "
                "Check whether the planner, preprocessing, and modelling outputs are internally consistent. "
                "Verify leakage safeguards, evaluation source, selected/best model distinction, generated code completeness, "
                "artifact completeness, and at least one numerical diagnostic using the Notebook Code Executor when available. "
                "Produce a final user-facing review with four labelled sections: Planner Agent, Data Preprocessing Agent, "
                "Modeler Evaluator Agent, and Reviewer Agent. The final section must include one concrete next improvement "
                "that would most improve the project."
            ),
            backstory=(
                "You are a senior machine-learning reviewer. You are skeptical in a productive way. You check that the "
                "workflow is not just visually impressive but also technically correct, reproducible, and safe from common "
                "tabular modelling mistakes. You know how to inspect residuals, compare models, and evaluate whether "
                "preprocessing has been described thoroughly enough for a user to trust the generated code. You write concise "
                "but complete feedback that can be shown directly in the Gradio Code & Review tab."
            ),
            llm=llm,
            tools=tools,
            allow_delegation=False,
            verbose=False,
        )

        plan_task = Task(
            description=(
                "You are the first agent in a sequential four-agent AutoML workflow.\n\n"
                "PROJECT CONTEXT:\n"
                f"- Regression target column: `{target_col}`.\n"
                f"- Selected model(s): {selected_models_txt}.\n"
                f"- Primary UI model: `{model_name}`.\n"
                f"- Evaluation plan from app configuration: {agent_plan.get('eval_plan', 'use uploaded validation file when provided, otherwise use a train/evaluation split')}.\n"
                f"- User-selected workflow summary: {agent_plan}.\n\n"
                "YOUR TASK:\n"
                "Create a careful step-by-step plan that downstream agents must follow. Your plan must include:\n"
                "1. Objective: state that this is a supervised regression workflow for the selected target.\n"
                "2. Data inspection: require checking shape, column names, target validity, missing values, numeric/categorical columns, date-like columns, and ID-like columns.\n"
                "3. Preprocessing: require numeric median imputation and scaling, categorical constant imputation and one-hot encoding, dropping date/date-like columns, and dropping ID-like high-cardinality categorical columns.\n"
                "4. Leakage prevention: require fitting preprocessing only on training data through a Pipeline/ColumnTransformer.\n"
                "5. Splitting/evaluation: require uploaded validation file use when available, otherwise train_test_split with the configured options.\n"
                "6. Modelling: require training every selected model using the same preprocessing pipeline.\n"
                "7. Evaluation: require MAE, MSE, RMSE, R², and MAPE percentage where valid.\n"
                "8. Outputs: require preprocessing summary, metrics card, comparison table, charts, feature importance if available, sample prediction, optional test submission, reproducible code, model file, ZIP artifact, and final review.\n"
                "9. Failure prevention: mention not to use target as a feature, not to fit on validation/test data, not to ignore missing target rows, and not to omit preprocessing code from generated output.\n\n"
                "Return detailed bullets. Do not invent columns or metrics that are not provided."
            ),
            expected_output=(
                "A detailed planner checklist with explicit instructions for data preprocessing, modelling, evaluation, "
                "generated-code completeness, and final review."
            ),
            agent=planner_agent,
        )

        preprocessing_task = Task(
            description=(
                "You are the second agent. Follow the Planner Agent's checklist and audit the actual preprocessing summary.\n\n"
                "ACTUAL PREPROCESSING SUMMARY FROM THE APP:\n"
                f"{summary}\n\n"
                "YOUR TASK:\n"
                "Explain and verify the preprocessing workflow in enough detail that the user can see exactly what happened before model training. Include:\n"
                "1. Original training shape and evaluation source.\n"
                "2. Target-column handling and missing-target-row removal.\n"
                "3. Numeric features retained and the numeric pipeline: SimpleImputer(strategy='median') then StandardScaler().\n"
                "4. Categorical features retained and the categorical pipeline: SimpleImputer(strategy='constant', fill_value='Unknown') then OneHotEncoder(handle_unknown='ignore').\n"
                "5. Dropped datetime/date-like columns and why they were dropped instead of directly modelled.\n"
                "6. Dropped ID-like columns and how high cardinality can harm generalization.\n"
                "7. Final encoded feature count and train/evaluation matrix shapes.\n"
                "8. Schema alignment for validation/test files through reindexing to the training feature list.\n"
                "9. Leakage safeguards: preprocessing is inside the pipeline and fitted only on training data.\n"
                "10. Any limitation or caution visible from the summary.\n\n"
                "When using the Notebook Code Executor, only run short verification snippets against `summary`; do not mutate the objects."
            ),
            expected_output=(
                "A detailed preprocessing report that explicitly covers retained columns, dropped columns, transformations, "
                "matrix shapes, and leakage safeguards."
            ),
            agent=preprocessing_agent,
            context=[plan_task],
        )

        modelling_task = Task(
            description=(
                "You are the third agent. Follow the Planner Agent's checklist and use the preprocessing report as context.\n\n"
                "MODEL COMPARISON TABLE FROM THE APP:\n"
                f"{comparison_txt}\n\n"
                f"PRIMARY UI MODEL: `{model_name}`\n\n"
                "YOUR TASK:\n"
                "Review the modelling and evaluation workflow. Include:\n"
                "1. Confirm that all selected models were trained with the same preprocessing pipeline.\n"
                "2. Identify the best model by lowest RMSE from the comparison table.\n"
                "3. Summarize the primary UI model separately from the best model if they are different.\n"
                "4. Interpret MAE, MSE, RMSE, R², and MAPE percentage, without overclaiming.\n"
                "5. Explain what the performance dashboard and residual diagnostics are checking.\n"
                "6. Explain when feature importance/coefficient output is available and why it may not exist for every model.\n"
                "7. Confirm that sample prediction and optional test-set prediction use the fitted pipeline.\n"
                "8. Mention any modelling limitation visible from the metrics, such as low/negative R² or high RMSE.\n\n"
                "When using the Notebook Code Executor, you may run a short check against `comparison`; do not retrain models."
            ),
            expected_output=(
                "A detailed modelling/evaluation report identifying the best model, the primary model's metrics, "
                "diagnostic interpretation, and limitations."
            ),
            agent=modeler_evaluator_agent,
            context=[plan_task, preprocessing_task],
        )

        review_task = Task(
            description=(
                "You are the final reviewer agent. Synthesize the prior agents' outputs and verify the completed workflow.\n\n"
                "CONTEXT AVAILABLE:\n"
                f"- Target column: `{target_col}`\n"
                f"- Primary UI model: `{model_name}`\n"
                f"- Preprocessing summary: {summary}\n"
                f"- Model comparison table:\n{comparison_txt}\n"
                f"{executor_note}\n\n"
                "YOUR TASK:\n"
                "Produce the final answer for the Gradio Code & Review tab. It must be detailed but organized. Use exactly these four section headings:\n"
                "### Planner Agent\n"
                "### Data Preprocessing Agent\n"
                "### Modeler Evaluator Agent\n"
                "### Reviewer Agent\n\n"
                "Under each heading, include clear bullets. The final answer must explicitly mention:\n"
                "1. The workflow objective and plan.\n"
                "2. The preprocessing actions and leakage safeguards.\n"
                "3. The best model by RMSE and the primary UI model's performance.\n"
                "4. The generated artifacts: reproducible Python code, fitted pipeline, charts, optional submission, and ZIP bundle.\n"
                "5. A diagnostic computed or verified using the code tool if available.\n"
                "6. One concrete next improvement, such as cross-validation plus hyperparameter tuning for the top models.\n\n"
                "Do not invent facts. If a detail is unavailable, say so briefly. Keep the response user-facing and useful."
            ),
            expected_output=(
                "Four labelled markdown sections with detailed but concise bullets. Include one verified diagnostic "
                "and one concrete next improvement."
            ),
            agent=reviewer_agent,
            context=[plan_task, preprocessing_task, modelling_task],
        )

        crew = Crew(
            agents=[planner_agent, preprocessing_agent, modeler_evaluator_agent, reviewer_agent],
            tasks=[plan_task, preprocessing_task, modelling_task, review_task],
            process=Process.sequential,
            verbose=False,
        )
        result = crew.kickoff()
        return getattr(result, "raw", None) or str(result)
    except Exception as e:
        return (
            f"❌ CrewAI multi-agent review failed: {type(e).__name__}: {e}\n"
            "The deterministic pipeline still trained successfully.\n\n"
            + _local_agent_report(agent_plan, summary, comparison_df, model_name, target_col)
        )

# ── Main run function ───────────────────────────────────────────────────────────
def on_run(train_df, valid_df, test_df,
           train_path, valid_path, test_path,
           target_col, selected_models, test_size, shuffle_split, random_state,
           id_threshold, enable_cv, cv_folds, enable_tuning,
           api_key, openai_model, id_col_choice, run_review,
           progress=gr.Progress()):

    empty = _empty_html()
    # 21 outputs; index 0 = status banner, index 18 = log, indices 19,20 = dropdown/slider updates.
    no_outputs_tail = (
        empty, empty, empty, empty, empty, empty, empty, empty,  # 1..8
        empty,                                                   # 9 (cross-model comparison)
        "", "",                                                  # 10 code, 11 review
        None, None, None, None,                                  # 12..15 file outputs
        pd.DataFrame(), None,                                    # 16 custom_inputs, 17 state
        "",                                                      # 18 log
        gr.update(choices=[], value=None),                       # 19 results-model dropdown
        gr.update(maximum=0, value=0),                           # 20 row-idx slider
    )

    # ── Validation ──
    if train_df is None:
        return (_info_banner("❌ Upload a training dataset.", "error"),) + no_outputs_tail
    if not target_col:
        return (_info_banner("❌ Select a target column.", "error"),) + no_outputs_tail
    if not selected_models:
        return (_info_banner(
            "❌ Pick at least one model in the Configure tab.", "error"),) + no_outputs_tail
    log_lines: List[str] = []
    def log(msg: str) -> None:
        ts = datetime.now().strftime("%H:%M:%S")
        log_lines.append(f"[{ts}] {msg}")

    try:
        log("Planner Agent: validating configuration and creating the execution plan…")
        agent_plan = {
            "selected_models": selected_models or [],
            "eval_plan": "Use uploaded validation file if available; otherwise create a train/evaluation split.",
            "plan_text": "Validate inputs, detect usable features, build a leakage-safe preprocessing pipeline, train selected regressors, evaluate, generate artifacts, and review."
        }
        progress(0.05, desc="Planner agent…")
        log("Data Preprocessing Agent: detecting columns and preparing data…")
        progress(0.08, desc="Data preprocessing agent…")
        schema = _detect_schema(train_df, target_col, float(id_threshold))
        numeric_cols = schema["numeric_cols"]
        categorical_cols = schema["categorical_cols"]
        feature_cols = schema["feature_cols"]
        high_card_cols = schema.get("high_card_cols", [])
        text_cols = schema.get("text_cols", [])
        log(f"Found {len(numeric_cols)} numeric, {len(categorical_cols)} categorical features"
            + (f" ({len(high_card_cols)} high-cardinality)" if high_card_cols else "")
            + (f", {len(text_cols)} text" if text_cols else "")
            + ".")

        X_all = train_df[feature_cols].copy()
        y_all = _coerce_numeric_series(train_df[target_col])
        valid_target_rows = y_all.notna()
        X_all = X_all.loc[valid_target_rows]
        y_all = y_all.loc[valid_target_rows]
        use_log_target = _should_log_transform(y_all)
        y_all = y_all.loc[valid_target_rows]

        has_valid = valid_df is not None
        has_test = test_df is not None

        if has_valid:
            if target_col not in valid_df.columns:
                raise ValueError(
                    f"Validation file must include the target column '{target_col}'."
                )
            X_train_raw, y_train = X_all, y_all
            X_eval_raw = valid_df.reindex(columns=feature_cols).copy()
            y_eval = _coerce_numeric_series(valid_df[target_col])
            keep_eval = y_eval.notna()
            X_eval_raw = X_eval_raw.loc[keep_eval]
            y_eval = y_eval.loc[keep_eval]
            eval_source = "Validation file"
            log(f"Using uploaded validation file ({len(X_eval_raw)} eval rows).")
        else:
            X_train_raw, X_eval_raw, y_train, y_eval = train_test_split(
                X_all, y_all, test_size=float(test_size),
                shuffle=bool(shuffle_split),
                random_state=int(random_state) if bool(shuffle_split) else None,
            )
            eval_source = (f"Train split "
                           f"({int((1 - float(test_size)) * 100)}/"
                           f"{int(float(test_size) * 100)})")
            log(f"Split training data: {len(X_train_raw)} train / {len(X_eval_raw)} eval.")

        if len(X_eval_raw) == 0:
            raise ValueError("Evaluation set has no valid rows after removing missing target values.")

        # Deduplicate while preserving the order the user picked.
        selected_models = list(dict.fromkeys(
            [m for m in (selected_models or []) if m in MODEL_REGISTRY]
        ))
        if not selected_models:
            raise ValueError("No valid models selected. Pick at least one model in the Configure tab.")
        log(f"Modeler Evaluator Agent: will train {', '.join(selected_models)}")

        progress(0.25, desc="Modeler evaluator agent…")
        rows = []
        fitted: Dict[str, Pipeline] = {}
        preds: Dict[str, np.ndarray] = {}
        feat_names: Dict[str, List[str]] = {}

        n_models = len(selected_models)
        for i, m in enumerate(selected_models, 1):
            log(f"  Modeler Evaluator Agent ({i}/{n_models}): training {m}…")
            progress(0.25 + 0.30 * i / n_models, desc=f"Training {m} ({i}/{n_models})")
            preprocessor = _build_preprocessor(schema)
            pipeline = Pipeline(steps=[
                ("preprocessor", preprocessor),
                ("model", _wrap_target(_load_model(m), use_log_target)),
            ])
            pipeline.fit(X_train_raw, y_train)
            y_pred = pipeline.predict(X_eval_raw)
            met = _metrics_dict(y_eval, y_pred)
            rows.append({"Model": m, **met})
            fitted[m] = pipeline
            preds[m] = y_pred
            feat_names[m] = _get_feature_names(pipeline.named_steps["preprocessor"])
            log(f"     {m}: R²={met['R2']:.4f}  RMSE={met['RMSE']:.4f}")

        comparison_df_full = pd.DataFrame(rows).sort_values(
            "RMSE", ascending=True
        ).reset_index(drop=True)

        # The "primary" / default model = first one the user selected.
        # (User can switch in the Results tab without retraining.)
        default_model = selected_models[0]
        default_pipeline = fitted[default_model]
        default_pred = preds[default_model]
        default_metrics = _metrics_dict(y_eval, default_pred)
        default_features = feat_names[default_model]

        summary = {
            "train_rows": len(train_df),
            "train_cols": len(train_df.columns),
            "eval_source": eval_source,
            "numeric_cols": numeric_cols,
            "categorical_cols": categorical_cols,
            "high_card_cols": high_card_cols,
            "cardinality": schema.get("cardinality", {}),
            "datetime_cols": schema.get("datetime_cols", []),
            "text_cols": schema.get("text_cols", []),
            "all_missing_cols": schema.get("all_missing_cols", []),
            "constant_cols": schema.get("constant_cols", []),
            "duplicate_cols": schema.get("duplicate_cols", []),
            "low_card_cols": schema.get("low_card_cols", []),
            "target_transform": "log1p/expm1" if use_log_target else "None",
            "schema": schema,
            "id_cols": schema["id_cols"],
            "encoded_feature_count": len(default_features),
            "X_train_shape": tuple(X_train_raw.shape),
            "X_eval_shape": tuple(X_eval_raw.shape),
        }

        log("Modeler Evaluator Agent: rendering charts and tables…")
        progress(0.62, desc="Rendering outputs…")
        preprocess_html = _render_preprocessing_summary(summary)
        metrics_html = _render_metrics(default_metrics, default_model, target_col, eval_source)

        comparison_display = comparison_df_full[["Model", "MAE", "MSE", "RMSE", "R2"]].copy()
        comparison_display = comparison_display.assign(**{
            "MAE":  comparison_display["MAE"].map(lambda x: f"{x:.4f}"),
            "MSE":  comparison_display["MSE"].map(lambda x: f"{x:.4f}"),
            "RMSE": comparison_display["RMSE"].map(lambda x: f"{x:.4f}"),
            "R2":   comparison_display["R2"].map(lambda x: f"{x:.4f}"),
        })
        comparison_html = _html_table(comparison_display, max_rows=20)

        charts_html, charts_png = _build_charts(
            y_eval, default_pred, default_model, eval_source,
        )
        feat_html, feat_png = _render_importances(
            default_pipeline, default_features, default_model,
            X_eval=X_eval_raw, y_eval=y_eval,
        )
        sample_html = _render_sample(
            default_pipeline, X_eval_raw, y_eval, target_col, 0,
        )

        # Cross-model comparison chart (only shown when ≥2 selected).
        cross_html, cross_png = _build_comparison_charts(
            comparison_df_full, preds, y_eval,
        )

        test_html, submission_path = (
            _render_test_download(default_pipeline, test_df, feature_cols, id_col_choice)
            if has_test else (_empty_html("No test file was uploaded."), None)
        )

        log("Reviewer Agent: generating reproducible code and saving artifacts…")
        progress(0.78, desc="Generating code & saving model…")
        reproducible_code = _build_reproducible_code(
            target_col=target_col,
            selected_models=selected_models,
            numeric_cols=numeric_cols,
            categorical_cols=categorical_cols,
            high_card_cols=high_card_cols,
            datetime_cols=schema["datetime_cols"],
            id_cols=schema["id_cols"],
            has_valid=has_valid,
            has_test=has_test,
            test_size=float(test_size),
            shuffle=bool(shuffle_split),
            random_state=int(random_state),
            id_col_choice=id_col_choice,
            train_path=train_path,
            valid_path=valid_path,
            test_path=test_path,
        )
        code_file_path = _create_code_file(reproducible_code)
        model_file_path = _save_model_to_disk(default_pipeline)
        log(f"Saved fitted pipeline → {os.path.basename(model_file_path)}")

        if run_review:
            log(f"Reviewer Agent: running four-agent CrewAI review with {openai_model}…")
            progress(0.88, desc="Running CrewAI review…")
            crewai_review = _run_crewai_review(
                api_key, openai_model, default_model, target_col, summary,
                comparison_df_full, default_pipeline, X_eval_raw, y_eval,
                agent_plan=agent_plan,
                use_executor=True,
            )
        else:
            crewai_review = _local_agent_report(agent_plan, summary, comparison_df_full, default_model, target_col)
        log("Done.")

        # Bundle all artifacts
        bundle_items = [
            ("reproducible_pipeline.py", code_file_path),
            ("fitted_pipeline.joblib", model_file_path),
            ("performance_charts.png", charts_png),
        ]
        if feat_png is not None:
            bundle_items.append(("feature_importance.png", feat_png))
        if cross_png is not None:
            bundle_items.append(("cross_model_comparison.png", cross_png))
        if submission_path is not None:
            bundle_items.append(("submission.csv", submission_path))
        zip_path = _bundle_artifacts_zip(bundle_items)

        # Default DataFrame for the custom predict tab. Pick a row with no
        # missing values across the feature columns so editing it always
        # produces a valid prediction; fall back to the first row if no
        # complete row exists. Categorical values are preserved as strings.
        if feature_cols:
            full = train_df[feature_cols]
            complete_rows = full.dropna()
            if len(complete_rows) > 0:
                custom_default = complete_rows.head(1).reset_index(drop=True).copy()
            else:
                custom_default = full.head(1).reset_index(drop=True).copy()
            # Ensure categorical columns stay as strings so the Gradio
            # Dataframe displays/accepts text values rather than coercing
            # them to NaN/int when the user edits them.
            for c in categorical_cols:
                if c in custom_default.columns:
                    custom_default[c] = custom_default[c].astype(str)
        else:
            custom_default = pd.DataFrame()

        # Full state — keeps every fitted model so the Results-tab dropdown
        # can switch between them without retraining.
        state_dict = {
            "fitted": fitted,
            "preds": preds,
            "feat_names": feat_names,
            "selected_models": selected_models,
            "active_model": default_model,
            "feature_cols": feature_cols,
            "X_eval_raw": X_eval_raw,
            "y_eval": y_eval,
            "target_col": target_col,
            "eval_source": eval_source,
            "comparison_df": comparison_df_full,
        }

        progress(1.0, desc="Done!")
        status_banner = _info_banner(
            f"✅ Agent workflow completed — best by RMSE: "
            f"<b>{comparison_df_full.iloc[0]['Model']}</b>", "ok"
        )

        return (
            status_banner,                              # 0  status_out
            preprocess_html,                            # 1  preprocessing_out
            metrics_html,                               # 2  metrics_out
            comparison_html,                            # 3  comparison_out
            charts_html,                                # 4  charts_out
            feat_html,                                  # 5  feat_out
            sample_html,                                # 6  sample_out
            test_html,                                  # 7  test_dl_out
            cross_html,                                 # 8  cross_model_out
            reproducible_code,                          # 9  code_out
            crewai_review,                              # 10 review_out
            code_file_path,                             # 11 code_file
            submission_path,                            # 12 submission_file
            model_file_path,                            # 13 model_file
            zip_path,                                   # 14 bundle_file
            custom_default,                             # 15 custom_inputs
            state_dict,                                 # 16 pipeline_state
            "\n".join(log_lines),                       # 17 log_out
            gr.update(choices=selected_models, value=default_model),  # 18 results_model_dd
            gr.update(maximum=max(len(X_eval_raw) - 1, 0), value=0),  # 19 row-idx slider
        )

    except Exception as e:
        log(f"FAILED: {type(e).__name__}: {e}")
        tb = traceback.format_exc()
        err_html = _info_banner(
            f"❌ <b>{type(e).__name__}</b>: {e}<br>"
            f"<details><summary>Stack trace</summary>"
            f"<pre style='font-size:11px;color:#fca5a5;'>{tb}</pre></details>",
            "error",
        )
        return (err_html,) + no_outputs_tail[:-2] + (
            gr.update(choices=[], value=None),
            gr.update(maximum=0, value=0),
        )


# ── Predict / sample-row / model-switch callbacks ───────────────────────────────
def _active_pipeline(state):
    """Return (pipeline, model_name) for the currently active model in state."""
    if state is None:
        return None, None
    fitted = state.get("fitted") or {}
    name = state.get("active_model")
    if name and name in fitted:
        return fitted[name], name
    # Fallbacks for safety.
    if fitted:
        first = next(iter(fitted))
        return fitted[first], first
    return None, None


def do_custom_predict(state, df_input):
    pipeline, name = _active_pipeline(state)
    if pipeline is None:
        return _empty_html("❌ Run the pipeline first.")
    if df_input is None or len(df_input) == 0:
        return _empty_html("❌ Enter one row of raw feature values.")
    try:
        row = df_input.iloc[[0]].copy()

        # Drop any 'Unnamed: 0'-style index column that may sneak in if the
        # user manually pastes data. The pipeline will reindex by feature_cols
        # below, but explicit cleanup makes errors clearer.
        bad = [c for c in row.columns if isinstance(c, str) and c.startswith("Unnamed:")]
        if bad:
            row = row.drop(columns=bad)

        # Reindex to the exact feature columns the pipeline was trained on
        # (any extras get dropped, any missing columns become NaN and are
        # then handled by the imputers in the preprocessing pipeline).
        feature_cols = state.get("feature_cols") or list(row.columns)
        row = row.reindex(columns=feature_cols)

        # Coerce numeric columns from string back to number — Gradio's
        # Dataframe sometimes returns numeric edits as strings.
        # Categorical columns stay as-is (the OneHotEncoder accepts strings
        # and ignores unseen categories thanks to handle_unknown='ignore').
        # We can't reliably detect numeric vs categorical from `state` alone,
        # so the safe move is: any column we can convert to numeric without
        # losing info, we convert; otherwise leave as string.
        for c in row.columns:
            v = row.iloc[0][c]
            if isinstance(v, str):
                try:
                    num = pd.to_numeric([v], errors="raise")[0]
                    row[c] = [num]
                except Exception:
                    pass  # keep as string for categorical cols

        pred = float(pipeline.predict(row)[0])
        rows = "".join(
            f"<tr><td style='padding:5px 12px;color:#64748b;font-size:11px;font-family:\"DM Mono\",monospace;'>{c}</td>"
            f"<td style='padding:5px 12px;color:#cbd5e1;font-size:11px;font-family:\"DM Mono\",monospace;text-align:right;'>{row.iloc[0][c]}</td></tr>"
            for c in row.columns[:20]
        )
        return (
            "<div style='background:#13131f;border:1px solid #2a2a3e;border-radius:12px;overflow:hidden;'>"
            "<div style='background:#0f2030;padding:10px 14px;border-bottom:1px solid #2a2a3e;'>"
            f"<span style='color:#06b6d4;font-size:11px;font-weight:600;letter-spacing:2px;text-transform:uppercase;'>"
            f"Custom Raw Row Prediction · model: {name}</span></div>"
            f"<div style='padding:18px;'><div style='color:#64748b;font-size:10px;text-transform:uppercase;margin-bottom:8px;'>Predicted Value</div>"
            f"<div style='color:#a78bfa;font-size:36px;font-weight:700;font-family:\"DM Mono\",monospace;'>{pred:.4f}</div></div>"
            f"<div style='border-top:1px solid #2a2a3e;padding:4px;'><table style='width:100%;border-collapse:collapse;'>{rows}</table></div></div>"
        )
    except Exception as e:
        return _info_banner(f"⚠️ Prediction error: {type(e).__name__}: {e}", "error")


def update_sample_row(state, row_idx):
    pipeline, _ = _active_pipeline(state)
    if pipeline is None:
        return _empty_html("Run the pipeline first.")
    return _render_sample(
        pipeline, state["X_eval_raw"], state["y_eval"],
        state["target_col"], int(row_idx),
    )


def select_results_model(state, model_name):
    """Re-render metrics, dashboard charts, importances and sample for
    the chosen model. Updates state in place so subsequent custom-row
    predictions and sample-row slider use the same model.
    Returns: (state, metrics_html, charts_html, feat_html, sample_html)."""
    if state is None or not state.get("fitted"):
        return (state, _empty_html("Run the pipeline first."),
                _empty_html(), _empty_html(), _empty_html())
    if not model_name or model_name not in state["fitted"]:
        # Fall back to the first selected model.
        model_name = state.get("active_model") or next(iter(state["fitted"]))

    pipeline = state["fitted"][model_name]
    pred = state["preds"][model_name]
    feat_n = state["feat_names"][model_name]
    y_eval = state["y_eval"]
    X_eval_raw = state["X_eval_raw"]
    target_col = state["target_col"]
    eval_source = state.get("eval_source", "")

    metrics = _metrics_dict(y_eval, pred)
    metrics_html = _render_metrics(metrics, model_name, target_col, eval_source)
    charts_html, _ = _build_charts(y_eval, pred, model_name, eval_source)
    feat_html, _ = _render_importances(pipeline, feat_n, model_name,
                                       X_eval=X_eval_raw, y_eval=y_eval)
    sample_html = _render_sample(pipeline, X_eval_raw, y_eval, target_col, 0)

    new_state = dict(state)
    new_state["active_model"] = model_name
    return new_state, metrics_html, charts_html, feat_html, sample_html
>>>>>>> 6292a482cf1e2f547f6a880fb59fdb116f3f1bc2
