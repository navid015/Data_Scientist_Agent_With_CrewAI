"""
ml_tools.py — CrewAI tools that agents use to do actual ML work.

Each tool wraps a specific capability (inspect data, run code, save charts,
save files, make predictions) and reads/writes from the shared STATE object.

No sklearn Pipeline or ColumnTransformer is used here.  Agents write raw
pandas/numpy/xgboost/lightgbm code which is executed by CodeRunnerTool.
"""
import base64, io, os, sys, tempfile, traceback, zipfile
from contextlib import redirect_stdout
from datetime import datetime
from typing import Any, Dict, List, Optional, Type

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from crewai.tools import BaseTool
from pydantic import BaseModel, Field, PrivateAttr

from ml_state import STATE


# ══════════════════════════════════════════════════════════════════════════════
# 1. DataInspectionTool
#    Agents use this to understand the raw dataset before writing any code.
# ══════════════════════════════════════════════════════════════════════════════
class _InspectInput(BaseModel):
    target_col: str = Field(description="Name of the target column to predict.")
    max_cats: int = Field(default=20, description="Max unique values to list for categorical columns.")


class DataInspectionTool(BaseTool):
    name: str = "Data Inspection Tool"
    description: str = (
        "Inspect the uploaded training and evaluation DataFrames. "
        "Returns a structured text report with: shape, dtypes, missing values, "
        "numeric statistics, cardinality of object columns, and target distribution. "
        "Call this FIRST before writing any preprocessing code."
    )
    args_schema: Type[BaseModel] = _InspectInput

    def _run(self, target_col: str, max_cats: int = 20) -> str:
        train_df = STATE.get("train_df")
        eval_df  = STATE.get("eval_df")
        if train_df is None:
            return "ERROR: No training data loaded. Upload a file first."

        lines = ["=== DATA INSPECTION REPORT ===\n"]

        def inspect_df(df, label):
            lines.append(f"--- {label} ---")
            lines.append(f"Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
            lines.append(f"Columns: {list(df.columns)}\n")
            lines.append("DTYPES & MISSING VALUES:")
            for col in df.columns:
                s = df[col]
                n_miss = int(s.isna().sum())
                pct    = n_miss / len(df) * 100
                dtype  = str(s.dtype)
                nu     = int(s.nunique(dropna=True))
                lines.append(
                    f"  {col!r:<30} dtype={dtype:<12} "
                    f"missing={n_miss} ({pct:.1f}%)  unique={nu}"
                )

            lines.append("\nNUMERIC COLUMNS — statistics:")
            num = df.select_dtypes(include=[np.number]).columns.tolist()
            if target_col in num:
                num.remove(target_col)
            for col in num[:20]:
                s = df[col].dropna()
                if len(s) == 0:
                    continue
                lines.append(
                    f"  {col!r:<30} "
                    f"min={s.min():.3g}  max={s.max():.3g}  "
                    f"mean={s.mean():.3g}  std={s.std():.3g}  "
                    f"skew={s.skew():.2f}"
                )

            lines.append("\nCATEGORICAL / OBJECT COLUMNS:")
            cat = df.select_dtypes(include=["object", "string", "category"]).columns.tolist()
            for col in cat:
                s   = df[col].dropna()
                nu  = s.nunique()
                top = s.value_counts().head(5).to_dict()
                lines.append(
                    f"  {col!r:<30} unique={nu}  "
                    f"top5={list(top.keys())[:5]}"
                )

            if target_col in df.columns:
                t = pd.to_numeric(df[target_col], errors="coerce").dropna()
                lines.append(f"\nTARGET '{target_col}':")
                lines.append(
                    f"  min={t.min():.4g}  max={t.max():.4g}  "
                    f"mean={t.mean():.4g}  median={t.median():.4g}  "
                    f"std={t.std():.4g}  skew={t.skew():.2f}"
                )
                neg = (t < 0).sum()
                zero = (t == 0).sum()
                lines.append(
                    f"  negative={neg}  zero={zero}  "
                    f"use_log_transform={'YES (skew>1.5 and all>=0)' if t.skew()>1.5 and neg==0 else 'NO'}"
                )
            lines.append("")

        inspect_df(train_df, "TRAINING DATA")
        if eval_df is not None:
            inspect_df(eval_df, "EVALUATION / VALIDATION DATA")
        else:
            lines.append("--- EVALUATION DATA: will be split from training ---\n")

        report = "\n".join(lines)
        STATE.log("DataInspectionTool: report generated")
        return report


# ══════════════════════════════════════════════════════════════════════════════
# 2. CodeRunnerTool
#    The most important tool: agents write Python code as a string and this
#    tool executes it.  The namespace is pre-loaded with the DataFrames and
#    common libraries.  Agents store results in special result_* variables.
# ══════════════════════════════════════════════════════════════════════════════
class _CodeInput(BaseModel):
    code: str = Field(description=(
        "Python code to execute.  The namespace already contains:\n"
        "  train_df, eval_df (or None), test_df (or None)\n"
        "  y_train, y_eval (pandas Series)\n"
        "  X_train, X_eval (DataFrames — available after preprocessing step)\n"
        "  np, pd, plt\n"
        "  XGBRegressor (from xgboost if installed)\n"
        "  LGBMRegressor (from lightgbm if installed)\n"
        "  RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor,\n"
        "  Ridge, Lasso, LinearRegression, HuberRegressor (from sklearn — models only, no Pipeline)\n\n"
        "To SAVE results, assign these special variables:\n"
        "  result_X_train          — preprocessed training features (DataFrame)\n"
        "  result_X_eval           — preprocessed eval features (DataFrame)\n"
        "  result_feature_names    — list[str] of feature names\n"
        "  result_preprocess_fn    — callable(raw_df)->np.ndarray for new-row prediction\n"
        "  result_model_NAME       — trained model (e.g. result_model_xgboost)\n"
        "  result_pred_NAME        — np.ndarray predictions  (e.g. result_pred_xgboost)\n"
        "  result_metrics_NAME     — dict with keys MAE,RMSE,R2,MAPE (e.g. result_metrics_xgboost)\n"
        "  result_preprocessing_summary — str description of what was done\n"
        "  result_generated_code   — str: clean reproducible script\n"
        "Include print() calls for visibility."
    ))


class CodeRunnerTool(BaseTool):
    name: str = "Python Code Runner"
    description: str = (
        "Execute Python code for preprocessing, training, or evaluation. "
        "The namespace contains the DataFrames and major ML libraries. "
        "Save results by assigning to result_* variables. "
        "Do NOT use sklearn Pipeline or ColumnTransformer — use raw pandas/numpy instead."
    )
    args_schema: Type[BaseModel] = _CodeInput

    def _build_namespace(self) -> Dict[str, Any]:
        ns: Dict[str, Any] = {
            "np": np, "pd": pd, "plt": plt,
            "train_df": STATE.get("train_df"),
            "eval_df":  STATE.get("eval_df"),
            "test_df":  STATE.get("test_df"),
            "y_train":  STATE.get("y_train"),
            "y_eval":   STATE.get("y_eval"),
            "X_train":  STATE.get("X_train"),
            "X_eval":   STATE.get("X_eval"),
            "feature_names": STATE.get("feature_names", []),
        }
        # Attempt to import optional ML libraries
        for module, alias, cls_name in [
            ("xgboost",    "XGBRegressor",             "XGBRegressor"),
            ("lightgbm",   "LGBMRegressor",            "LGBMRegressor"),
            ("catboost",   "CatBoostRegressor",        "CatBoostRegressor"),
        ]:
            try:
                mod = __import__(module)
                ns[cls_name] = getattr(mod, cls_name)
            except Exception:
                pass
        # sklearn model classes (no Pipeline / ColumnTransformer)
        try:
            from sklearn.ensemble import (
                RandomForestRegressor, ExtraTreesRegressor,
                GradientBoostingRegressor, HistGradientBoostingRegressor,
                AdaBoostRegressor,
            )
            from sklearn.linear_model import (
                Ridge, Lasso, ElasticNet, LinearRegression,
                BayesianRidge, HuberRegressor,
            )
            from sklearn.tree import DecisionTreeRegressor
            from sklearn.neighbors import KNeighborsRegressor
            from sklearn.metrics import (
                mean_absolute_error, mean_squared_error,
                r2_score, mean_absolute_percentage_error,
            )
            ns.update({
                "RandomForestRegressor": RandomForestRegressor,
                "ExtraTreesRegressor":   ExtraTreesRegressor,
                "GradientBoostingRegressor": GradientBoostingRegressor,
                "HistGradientBoostingRegressor": HistGradientBoostingRegressor,
                "AdaBoostRegressor":     AdaBoostRegressor,
                "Ridge": Ridge, "Lasso": Lasso, "ElasticNet": ElasticNet,
                "LinearRegression":      LinearRegression,
                "BayesianRidge":         BayesianRidge,
                "HuberRegressor":        HuberRegressor,
                "DecisionTreeRegressor": DecisionTreeRegressor,
                "KNeighborsRegressor":   KNeighborsRegressor,
                "mean_absolute_error":   mean_absolute_error,
                "mean_squared_error":    mean_squared_error,
                "r2_score":              r2_score,
                "mean_absolute_percentage_error": mean_absolute_percentage_error,
            })
        except Exception:
            pass
        return ns

    def _extract_results(self, ns: Dict[str, Any]) -> None:
        """After code runs, scan for result_* variables and save to STATE."""
        # Preprocessed features
        if "result_X_train" in ns and ns["result_X_train"] is not None:
            STATE.set("X_train", ns["result_X_train"])
        if "result_X_eval" in ns and ns["result_X_eval"] is not None:
            STATE.set("X_eval", ns["result_X_eval"])
        if "result_feature_names" in ns:
            STATE.set("feature_names", list(ns["result_feature_names"]))
            STATE.set("feature_cols",  list(ns["result_feature_names"]))
        if "result_preprocess_fn" in ns and callable(ns["result_preprocess_fn"]):
            STATE.set("preprocess_fn", ns["result_preprocess_fn"])
        if "result_preprocessing_summary" in ns:
            STATE.set("preprocessing_summary", str(ns["result_preprocessing_summary"]))
        if "result_generated_code" in ns:
            STATE.set("generated_code", str(ns["result_generated_code"]))

        # Models and predictions — scan for result_model_XXX and result_pred_XXX
        for key, val in ns.items():
            if key.startswith("result_model_") and val is not None:
                name = key[len("result_model_"):]
                models = STATE.get("models", {})
                models[name] = val
                STATE.set("models", models)
                if not STATE.get("active_model"):
                    STATE.set("active_model", name)
            if key.startswith("result_pred_") and val is not None:
                name = key[len("result_pred_"):]
                preds = STATE.get("predictions", {})
                preds[name] = np.asarray(val).flatten()
                STATE.set("predictions", preds)
            if key.startswith("result_metrics_") and isinstance(val, dict):
                name = key[len("result_metrics_"):]
                metrics = STATE.get("metrics", {})
                metrics[name] = val
                STATE.set("metrics", metrics)

    def _run(self, code: str) -> str:
        ns  = self._build_namespace()
        buf = io.StringIO()
        try:
            with redirect_stdout(buf):
                exec(code, ns)
            self._extract_results(ns)
            out = buf.getvalue() or "[no print output]"
            STATE.log(f"CodeRunnerTool: executed OK ({len(code)} chars)")
            return f"✅ Code executed successfully.\nOutput:\n{out}"
        except Exception as exc:
            tb  = traceback.format_exc()
            out = buf.getvalue()
            STATE.log(f"CodeRunnerTool: ERROR — {type(exc).__name__}: {exc}")
            return (
                f"❌ Execution error: {type(exc).__name__}: {exc}\n"
                f"Traceback:\n{tb}\n"
                f"Partial output:\n{out}"
            )


# ══════════════════════════════════════════════════════════════════════════════
# 3. ChartTool
#    Agents write matplotlib code; this tool runs it, captures the PNG bytes,
#    and stores them in STATE['charts_png'].
# ══════════════════════════════════════════════════════════════════════════════
class _ChartInput(BaseModel):
    chart_name: str = Field(description="Short key name for this chart, e.g. 'performance' or 'importance'.")
    code: str = Field(description=(
        "Matplotlib code that creates a figure.  Namespace contains:\n"
        "  np, pd, plt, y_eval, predictions (dict name->array), "
        "  models (dict name->model), X_eval, feature_names\n"
        "End with:  fig = plt.gcf()  — do NOT call plt.show() or plt.savefig()."
    ))


class ChartTool(BaseTool):
    name: str = "Chart Generation Tool"
    description: str = (
        "Generate a matplotlib chart and save it as PNG bytes to shared state. "
        "Write matplotlib code ending with `fig = plt.gcf()`. "
        "Chart is stored under the given chart_name key."
    )
    args_schema: Type[BaseModel] = _ChartInput

    def _run(self, chart_name: str, code: str) -> str:
        ns: Dict[str, Any] = {
            "np": np, "pd": pd, "plt": plt,
            "y_eval":       STATE.get("y_eval"),
            "predictions":  STATE.get("predictions", {}),
            "models":       STATE.get("models", {}),
            "X_eval":       STATE.get("X_eval"),
            "feature_names": STATE.get("feature_names", []),
        }
        # dark theme defaults
        plt.rcParams.update({
            "figure.facecolor": "#0d0d1a",
            "axes.facecolor":   "#1a1a2e",
            "axes.edgecolor":   "#2a2a3e",
            "axes.labelcolor":  "#e2e8f0",
            "xtick.color":      "#e2e8f0",
            "ytick.color":      "#e2e8f0",
            "text.color":       "#e2e8f0",
            "grid.color":       "#2a2a3e",
            "font.family":      "monospace",
        })
        buf = io.StringIO()
        try:
            with redirect_stdout(buf):
                exec(code, ns)
            fig = ns.get("fig") or plt.gcf()
            png_buf = io.BytesIO()
            fig.savefig(png_buf, format="png", dpi=130,
                        bbox_inches="tight", facecolor="#0d0d1a")
            png_buf.seek(0)
            png_bytes = png_buf.read()
            plt.close("all")
            charts = STATE.get("charts_png", {})
            charts[chart_name] = png_bytes
            STATE.set("charts_png", charts)
            STATE.log(f"ChartTool: saved chart '{chart_name}' ({len(png_bytes)//1024}KB)")
            return f"✅ Chart '{chart_name}' saved ({len(png_bytes)//1024} KB)."
        except Exception as exc:
            plt.close("all")
            STATE.log(f"ChartTool: ERROR — {exc}")
            return f"❌ Chart error: {type(exc).__name__}: {exc}\n{traceback.format_exc()}"


# ══════════════════════════════════════════════════════════════════════════════
# 4. FileSaverTool
#    Saves trained models to .joblib, code to .py, predictions to .csv,
#    and bundles everything to .zip.
# ══════════════════════════════════════════════════════════════════════════════
class _FileSaveInput(BaseModel):
    save_model:    bool = Field(default=True,  description="Save the active trained model as .joblib")
    save_code:     bool = Field(default=True,  description="Save the generated code as .py")
    save_submission: bool = Field(default=False, description="Save test predictions as submission.csv")
    id_col:        str  = Field(default="(none)", description="ID column to include in submission.csv")
    create_zip:    bool = Field(default=True,  description="Bundle everything into a .zip file")


class FileSaverTool(BaseTool):
    name: str = "File Saver Tool"
    description: str = (
        "Save artifacts to disk: trained model (.joblib), generated code (.py), "
        "test predictions (.csv), and a ZIP bundle.  Call after training and code generation."
    )
    args_schema: Type[BaseModel] = _FileSaveInput

    def _run(self, save_model: bool = True, save_code: bool = True,
             save_submission: bool = False, id_col: str = "(none)",
             create_zip: bool = True) -> str:
        ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
        tmp      = tempfile.gettempdir()
        saved    = []
        zip_items = []

        if save_model:
            active = STATE.get("active_model")
            models = STATE.get("models", {})
            if active and active in models:
                path = os.path.join(tmp, f"fitted_model_{ts}.joblib")
                joblib.dump({"model": models[active],
                             "feature_names": STATE.get("feature_names", []),
                             "active_model": active}, path)
                STATE.set("model_file_path", path)
                saved.append(f"Model → {path}")
                zip_items.append(("fitted_model.joblib", path))

        if save_code:
            code = STATE.get("generated_code", "")
            if code.strip():
                path = os.path.join(tmp, f"pipeline_{ts}.py")
                with open(path, "w", encoding="utf-8") as f:
                    f.write(code.replace("```python", "").replace("```", ""))
                STATE.set("code_file_path", path)
                saved.append(f"Code → {path}")
                zip_items.append(("pipeline.py", path))

        if save_submission:
            test_df = STATE.get("test_df")
            models  = STATE.get("models", {})
            active  = STATE.get("active_model")
            feat    = STATE.get("feature_names", [])
            fn      = STATE.get("preprocess_fn")
            if test_df is not None and active in models and fn is not None:
                try:
                    X_test = fn(test_df)
                    preds  = np.asarray(models[active].predict(X_test)).flatten()
                    if id_col and id_col != "(none)" and id_col in test_df.columns:
                        out = pd.DataFrame({id_col: test_df[id_col].values,
                                            "prediction": preds})
                    else:
                        out = pd.DataFrame({"id": range(len(preds)),
                                            "prediction": preds})
                    path = os.path.join(tmp, "submission.csv")
                    out.to_csv(path, index=False)
                    STATE.set("submission_path", path)
                    saved.append(f"Submission → {path}")
                    zip_items.append(("submission.csv", path))
                except Exception as exc:
                    saved.append(f"Submission FAILED: {exc}")

        # Add charts to zip
        for cname, cbytes in STATE.get("charts_png", {}).items():
            if cbytes:
                zip_items.append((f"{cname}.png", cbytes))

        if create_zip and zip_items:
            zip_path = os.path.join(tmp, f"automl_artifacts_{ts}.zip")
            with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
                for fname, content in zip_items:
                    if isinstance(content, (bytes, bytearray)):
                        zf.writestr(fname, bytes(content))
                    elif isinstance(content, str) and os.path.exists(content):
                        zf.write(content, arcname=fname)
            STATE.set("zip_path", zip_path)
            saved.append(f"ZIP bundle → {zip_path}")

        STATE.log(f"FileSaverTool: {len(saved)} files saved")
        return "✅ Files saved:\n" + "\n".join(f"  {s}" for s in saved)


# ══════════════════════════════════════════════════════════════════════════════
# 5. ComparisonBuilderTool
#    Reads metrics from STATE and builds a comparison DataFrame + HTML table.
# ══════════════════════════════════════════════════════════════════════════════
class _CmpInput(BaseModel):
    dummy: str = Field(default="run", description="Pass 'run' to build the comparison table.")


class ComparisonBuilderTool(BaseTool):
    name: str = "Comparison Builder Tool"
    description: str = (
        "Read all model metrics from shared state and build a comparison "
        "DataFrame sorted by RMSE.  Call after all models are trained."
    )
    args_schema: Type[BaseModel] = _CmpInput

    def _run(self, dummy: str = "run") -> str:
        metrics = STATE.get("metrics", {})
        if not metrics:
            return "No metrics found.  Train at least one model first."
        rows = [{"Model": name, **m} for name, m in metrics.items()]
        df   = pd.DataFrame(rows).sort_values("RMSE", ascending=True).reset_index(drop=True)
        STATE.set("comparison_df", df)
        if not STATE.get("active_model"):
            STATE.set("active_model", df.iloc[0]["Model"])
        STATE.log("ComparisonBuilderTool: comparison table built")
        return f"✅ Comparison table built ({len(df)} models):\n{df.to_string(index=False)}"


# ── Convenience export ─────────────────────────────────────────────────────────
def make_tools() -> list:
    return [
        DataInspectionTool(),
        CodeRunnerTool(),
        ChartTool(),
        FileSaverTool(),
        ComparisonBuilderTool(),
    ]
