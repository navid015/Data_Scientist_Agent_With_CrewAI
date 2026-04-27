"""Notebook executor tool for CrewAI review agents.

This tool intentionally keeps the shared-namespace behavior because the app
passes already-computed objects such as best_pipeline, X_eval, y_eval, and the
comparison table to the reviewer agent. It captures stdout/stderr-style errors
so diagnostics become visible in the CrewAI review output.
"""

import io
import subprocess
import sys
from contextlib import redirect_stdout
from typing import Any, Dict, List, Optional, Type

import numpy as np
import pandas as pd
from crewai.tools import BaseTool
from pydantic import BaseModel, Field, PrivateAttr


class NotebookCodeExecutorSchema(BaseModel):
    code: str = Field(description="Python code to execute. Include print() calls for visible output.")
    required_libraries: Optional[List[str]] = Field(
        default=None,
        description="Optional pip package names to install before execution.",
    )


class NotebookCodeExecutor(BaseTool):
    name: str = "Notebook Code Executor"
    description: str = (
        "Executes Python diagnostics in a shared namespace. Use it only to verify "
        "already-computed pipeline objects, metrics, residuals, feature lists, and data shapes."
    )
    args_schema: Type[BaseModel] = NotebookCodeExecutorSchema
    _execution_namespace: Dict[str, Any] = PrivateAttr(default_factory=dict)

    def __init__(self, namespace: Optional[Dict[str, Any]] = None, **kwargs):
        super().__init__(**kwargs)
        self._execution_namespace = namespace or {}
        self._execution_namespace.setdefault("pd", pd)
        self._execution_namespace.setdefault("np", np)

    def _run(self, code: str, required_libraries: Optional[List[str]] = None) -> str:
        install_log = ""
        if required_libraries:
            install_log += "--- Installing Libraries ---\n"
            for lib in required_libraries:
                try:
                    proc = subprocess.run(
                        [sys.executable, "-m", "pip", "install", lib],
                        capture_output=True,
                        text=True,
                        check=False,
                        timeout=120,
                    )
                    install_log += f"✓ {lib}\n" if proc.returncode == 0 else f"✗ {lib}: {proc.stderr[:500]}\n"
                except Exception as exc:
                    install_log += f"✗ {lib}: {exc}\n"
            install_log += "---\n\n"

        buf = io.StringIO()
        try:
            with redirect_stdout(buf):
                exec(code, self._execution_namespace)
            out = buf.getvalue() or "[no output]"
            return install_log + f"--- Executing Code ---\n✅ Success.\n```output\n{out}\n```\n"
        except Exception as exc:
            partial = buf.getvalue()
            msg = f"--- Executing Code ---\n❌ {type(exc).__name__}: {exc}\n"
            if partial:
                msg += f"Output before error:\n```output\n{partial}\n```\n"
            return install_log + msg
