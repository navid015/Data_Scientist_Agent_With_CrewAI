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
