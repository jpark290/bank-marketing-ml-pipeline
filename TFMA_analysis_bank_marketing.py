#!/usr/bin/env python
# coding: utf-8

# # Bank Marketing — TFMA Analysis Notebook
# 
# This notebook is the deliverable for the **"TFX Model Analysis (TFMA)"** section of the
# project requirements (separate from the main pipeline walkthrough notebook, which covers
# Steps 1–10 of the pipeline itself).
# 
# It assumes the interactive walkthrough has generated evaluation artifacts
# from at least two runs, such as a baseline run and a modified configuration
# (learning rate, train steps, model architecture, etc.) — so that the `Evaluator` component
# has written more than one evaluation result to `PIPELINE_ROOT/Evaluator/evaluation/<run_id>/`.
# 
# This notebook does **not** re-run the pipeline. It only reads the `EvalResult` artifacts
# that the Evaluator component already wrote to disk and renders them with TFMA's
# visualization widgets.
# 
# Covers:
# - **Step 1** — Load and render TFMA results (`tfma.load_eval_result`, `tfma.view.render_slicing_metrics`)
# - **Step 2** — Fairness indicators (`tfma.view.render_fairness_indicator`)
# - **Step 3** — Compare two pipeline runs side by side
# 
# > **Note:** TFMA's widgets are Jupyter widgets — run this in classic Jupyter Notebook or
# > JupyterLab with `tensorflow_model_analysis` and its widget extensions installed and
# > enabled (the same environment you used for the main walkthrough notebook is fine).

# In[1]:


import os
import glob

# Must match the constants used in the main pipeline notebook / pipeline.py
PIPELINE_NAME = "bank_marketing_pipeline"
PIPELINE_ROOT = os.path.join(
    os.path.expanduser("~"), "COMP315", "interactive_pipeline_outputs", PIPELINE_NAME
)

EVALUATOR_OUTPUT_DIR = os.path.join(PIPELINE_ROOT, "Evaluator", "evaluation")
print("Looking for evaluation runs under:", EVALUATOR_OUTPUT_DIR)


# In[2]:


# Each subfolder here is one Evaluator execution (one pipeline run), named by its
# numeric MLMD execution id. Sort them so the *last* entry is the most recent run.
run_dirs = sorted(
    [
        d
        for d in glob.glob(os.path.join(EVALUATOR_OUTPUT_DIR, "*"))
        if os.path.isdir(d) and os.path.basename(d).isdigit()
    ],
    key=lambda d: int(os.path.basename(d)),
)

print(f"Found {len(run_dirs)} evaluation run(s):")
for d in run_dirs:
    print(" -", d)

if not run_dirs:
    raise RuntimeError(
        "No Evaluator runs found. Run the main pipeline notebook first."
    )

LATEST_EVAL_PATH = run_dirs[-1]
print("\nUsing most recent run for Steps 1-2:", LATEST_EVAL_PATH)


# ## Step 1 — Load and render TFMA results
# 
# Load the `EvalResult` written by the Evaluator component for the most recent run, then
# render the overall metrics and the per-slice breakdown for the `marital` slice defined in
# `eval_config` (see `pipeline.py` / the main notebook's `Step 8` cell).

# In[3]:


import tensorflow_model_analysis as tfma

print(tfma.__version__)

eval_result = tfma.load_eval_result(
    output_path=LATEST_EVAL_PATH
)

viewer = tfma.view.render_slicing_metrics(eval_result)

viewer


# In[4]:


# Per-slice breakdown over "marital" — the sensitive/interesting feature slice
# configured in the EvalConfig's second SlicingSpec.
view = tfma.view.render_slicing_metrics(eval_result, slicing_column="marital")
view


# ### Interpreting overall and sliced metrics
#
# Compare overall metrics with each marital-status group's metrics.
# Consider ExampleCount when interpreting differences between groups.
# Recorded results and their interpretation are documented in the report
# and repository README.
#

# ## Step 2 — Inspect fairness metrics
# 
# Render the fairness dashboard on the same evaluation result to compare metric disparities
# across the `marital` slices directly.

# In[5]:


from tensorflow_model_analysis.addons.fairness.view import widget_view

fairness_view = widget_view.render_fairness_indicator(
    eval_result
)

fairness_view


# ### Interpreting subgroup performance
#
# Compare accuracy, AUC, precision, and recall across marital-status groups.
# The lowest-performing group can differ depending on the metric.
# Performance differences alone do not establish their cause or prove
# that the model is fair or unfair.
#
# ## Step 3 — Compare two runs
# 
# Pick the two most recent evaluation runs (e.g. your baseline run and a run after changing
# a hyperparameter such as `train_steps`, learning rate, or model architecture) and render
# them side by side.

# In[6]:


if len(run_dirs) < 2:
    raise RuntimeError(
        "Need at least two Evaluator runs to compare."
    )

RUN_A_PATH, RUN_B_PATH = run_dirs[-2], run_dirs[-1]

print("Run A (older):", RUN_A_PATH)
print("Run B (newer):", RUN_B_PATH)

eval_result_a = tfma.load_eval_result(output_path=RUN_A_PATH)
eval_result_b = tfma.load_eval_result(output_path=RUN_B_PATH)

from IPython.display import display, Markdown

view_a = tfma.view.render_slicing_metrics(
    eval_result_a,
    slicing_column="marital"
)

view_b = tfma.view.render_slicing_metrics(
    eval_result_b,
    slicing_column="marital"
)

display(Markdown("### Model 1: Run A"))
display(view_a)

display(Markdown("### Model 2: Run B"))
display(view_b)


# In[7]:


import pandas as pd
from IPython.display import display

def extract_run_metrics(eval_result, run_name):
    rows = []

    for slice_key, metric_groups in eval_result.slicing_metrics:
        values = metric_groups[""][""]

        if not slice_key:
            slice_name = "Overall"
        else:
            slice_name = ", ".join(
                f"{feature}={value}"
                for feature, value in slice_key
            )

        def get_metric(metric_name):
            return values.get(metric_name, {}).get("doubleValue")

        rows.append({
            "Slice": slice_name,
            f"{run_name} Example Count": get_metric("example_count"),
            f"{run_name} Accuracy": get_metric("binary_accuracy"),
            f"{run_name} AUC": get_metric("auc"),
            f"{run_name} Precision": get_metric("precision"),
            f"{run_name} Recall": get_metric("recall"),
        })

    return pd.DataFrame(rows)


model1_df = extract_run_metrics(eval_result_a, "Model 1")
model2_df = extract_run_metrics(eval_result_b, "Model 2")

comparison_df = model1_df.merge(
    model2_df,
    on="Slice",
    how="outer"
)

comparison_df["Accuracy Change"] = (
    comparison_df["Model 2 Accuracy"]
    - comparison_df["Model 1 Accuracy"]
)

comparison_df["AUC Change"] = (
    comparison_df["Model 2 AUC"]
    - comparison_df["Model 1 AUC"]
)

comparison_df = comparison_df[
    [
        "Slice",
        "Model 1 Example Count",
        "Model 2 Example Count",
        "Model 1 Accuracy",
        "Model 2 Accuracy",
        "Accuracy Change",
        "Model 1 AUC",
        "Model 2 AUC",
        "AUC Change",
        "Model 1 Precision",
        "Model 2 Precision",
        "Model 1 Recall",
        "Model 2 Recall",
    ]
]

print("TFMA comparison of Model 1 and Model 2")
display(comparison_df.round(4))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




