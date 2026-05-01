"""
ml_state.py — Thread-safe shared state between CrewAI agents and Gradio.

Agents write results here via tools; Gradio reads here to render outputs.
This replaces the sklearn Pipeline object as the session memory.
"""
import threading
from datetime import datetime
from typing import Any, Dict, Optional


class SharedState:
    """Thread-safe key-value store.  Agents write; Gradio reads."""

    def __init__(self):
        self._lock = threading.RLock()
        self._data: Dict[str, Any] = {
            # ── Raw DataFrames (set at upload time) ──────────────────────────
            "train_df":   None,
            "eval_df":    None,
            "test_df":    None,
            "target_col": None,
            "eval_source": "",
            # ── Processed arrays (written by Preprocessing agent) ────────────
            "X_train":       None,
            "X_eval":        None,
            "y_train":       None,
            "y_eval":        None,
            "feature_names": [],
            "feature_cols":  [],
            # ── Models and results (written by ML Engineer agent) ────────────
            "models":      {},   # name -> trained model object
            "predictions": {},   # name -> np.ndarray
            "metrics":     {},   # name -> {MAE, RMSE, R2, MAPE}
            "selected_models": [],
            "active_model": None,
            # ── Charts (written by Evaluator agent) ──────────────────────────
            "charts_png":  {},   # name -> png bytes
            # ── Preprocessing function (for custom predictions) ──────────────
            "preprocess_fn": None,   # callable: raw_df -> np.ndarray
            # ── Artifacts (file paths on disk) ────────────────────────────────
            "model_file_path":  None,
            "code_file_path":   None,
            "submission_path":  None,
            "zip_path":         None,
            # ── Text outputs ─────────────────────────────────────────────────
            "preprocessing_summary": "",
            "generated_code":        "",
            "review_text":           "",
            "comparison_df":         None,
            # ── Logging ──────────────────────────────────────────────────────
            "agent_log": [],
            "status":    "idle",
            "error":     None,
        }

    # ── Public API ─────────────────────────────────────────────────────────────
    def get(self, key: str, default: Any = None) -> Any:
        with self._lock:
            return self._data.get(key, default)

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            self._data[key] = value

    def update(self, d: Dict[str, Any]) -> None:
        with self._lock:
            self._data.update(d)

    def log(self, msg: str) -> None:
        ts = datetime.now().strftime("%H:%M:%S")
        with self._lock:
            self._data["agent_log"].append(f"[{ts}] {msg}")

    def get_log(self) -> str:
        with self._lock:
            return "\n".join(self._data["agent_log"])

    def reset_run(self) -> None:
        """Clear everything except the uploaded DataFrames."""
        with self._lock:
            keep = {k: self._data[k] for k in
                    ("train_df", "eval_df", "test_df", "target_col")}
            self._data.update({
                "eval_source": "", "X_train": None, "X_eval": None,
                "y_train": None, "y_eval": None,
                "feature_names": [], "feature_cols": [],
                "models": {}, "predictions": {}, "metrics": {},
                "selected_models": [], "active_model": None,
                "charts_png": {}, "preprocess_fn": None,
                "model_file_path": None, "code_file_path": None,
                "submission_path": None, "zip_path": None,
                "preprocessing_summary": "", "generated_code": "",
                "review_text": "", "comparison_df": None,
                "agent_log": [], "status": "idle", "error": None,
            })
            self._data.update(keep)


# Module-level singleton — import this everywhere
STATE = SharedState()
