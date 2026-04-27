---
title: CrewAI ML Engineer Agent
emoji: 🤖
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: "5.0.0"
python_version: "3.10"
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