<<<<<<< HEAD
# CrewAI AutoML — Fully Agent-Based

This project replaces every scikit-learn Pipeline/ColumnTransformer with five
CrewAI agents that write and execute Python code to accomplish ML tasks.

## Architecture

```
app.py          ← Gradio UI (unchanged from original)
ml_state.py     ← Thread-safe shared state (agents write, Gradio reads)
ml_tools.py     ← CrewAI tools: DataInspection, CodeRunner, Chart, FileSaver, Comparison
ml_agents.py    ← Five agents + crew runner
helpers.py      ← HTML renderers that read from STATE
```

## The Five Agents

| # | Agent | Role |
|---|-------|------|
| 1 | Data Analyst | Calls DataInspectionTool, classifies every column, recommends preprocessing |
| 2 | Preprocessing Engineer | Writes + executes pandas/numpy code (no sklearn preprocessors) |
| 3 | ML Engineer | Trains XGBoost/LightGBM/RandomForest directly on preprocessed arrays |
| 4 | Evaluator | Generates 6-panel performance dashboard + feature importance charts |
| 5 | Code Generator | Writes reproducible script, saves files, writes final review |

## What's NOT used
- `sklearn.pipeline.Pipeline`
- `sklearn.compose.ColumnTransformer`
- `sklearn.preprocessing.*` (StandardScaler, OneHotEncoder, etc.)
- `sklearn.impute.SimpleImputer`

All preprocessing is written by the Preprocessing Engineer agent as plain pandas/numpy code.
sklearn model classes (RandomForestRegressor, Ridge, etc.) are still used for fitting.

## Running

```bash
pip install -r requirements.txt
python app.py
```

Open http://127.0.0.1:7860 in your browser. You will need an OpenAI API key.
=======
<!-- ---
title: CrewAI Supplement Sales ML
emoji: 🤖
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
--- -->

---
title: CrewAI AutoML Regression Pipeline
emoji: ⚡
colorFrom: purple
colorTo: cyan
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
---

# ⚡ CrewAI AutoML — Regression Pipeline

Upload **any tabular dataset** (CSV or Excel), pick a target column, and let 3 AI agents automatically:

1. **Plan** the ML pipeline
2. **Preprocess** & encode the data
3. **Train** a Random Forest, print metrics & feature importances

## How to use

1. Upload your CSV or Excel file
2. Preview the first 10 rows in the interface
3. Select the column you want to predict
4. Enter your OpenAI API key
5. Click **Run Pipeline**

## Requirements

- OpenAI API key (`gpt-4.1-mini`)
- Dataset with at least one numeric target column
- Supported formats: `.csv`, `.xlsx`, `.xls`
>>>>>>> 6292a482cf1e2f547f6a880fb59fdb116f3f1bc2
