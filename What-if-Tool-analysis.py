#!/usr/bin/env python
# coding: utf-8

# # Bank Marketing — What-If Tool & Partial Dependence Analysis
# 
# This notebook is the deliverable for the **"What-If Tool"** section of the project
# requirements (separate from the main pipeline walkthrough notebook and the TFMA
# analysis notebook). It assumes you have already run the interactive
# walkthrough at least once, so that:
# 
# - `ExampleGen` has written out eval-split `tf.Example`s (TFRecords) under
#   `PIPELINE_ROOT/CsvExampleGen/examples/.../Split-eval/`, and
# - `Pusher` has copied a **blessed** `SavedModel` to `SERVING_MODEL_DIR`.
# 
# This analysis covers:
#
# 1. Prediction inspection using the What-If Tool.
#    Recorded contrasting examples are discussed in the project report.
#    Comparing examples that differ in multiple features does not isolate
#    the effect of a single feature or establish causality.
#
# 2. Feature distributions and partial dependence plots.
#    Static plots summarize changes in mean predicted subscription
#    probability as individual feature values are varied.
#
# 3. Threshold analysis by marital status.
#    Overall and subgroup metrics are compared across classification
#    thresholds to examine false positives, false negatives, and positive
#    prediction rates.
#
# ## Setup — install the What-If Tool widget
# 
# `witwidget` isn't part of the core TFX install. Run this once (skip if already
# installed in your environment).

# In[1]:


# !pip install witwidget
import witwidget
print("witwidget:", witwidget.__version__ if hasattr(witwidget, "__version__") else "installed")


# In[2]:


get_ipython().system('pip install matplotlib')
import os
import glob
import random

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from witwidget.notebook.visualization import WitWidget, WitConfigBuilder

# Must match the constants used in the main pipeline notebook / pipeline.py
PIPELINE_NAME = "bank_marketing_pipeline"
PIPELINE_ROOT = os.path.join(
    os.path.expanduser("~"), "COMP315", "interactive_pipeline_outputs", PIPELINE_NAME
)
SERVING_MODEL_DIR = os.path.join(
    os.path.expanduser("~"), "COMP315", "interactive_pipeline_outputs",
    "serving_model", PIPELINE_NAME,
)

LABEL_KEY = "y"
NUM_DATAPOINTS = 500     # how many eval examples to load into WIT (keep small — WIT is interactive)
RANDOM_SEED = 42

print("Pipeline root:      ", PIPELINE_ROOT)
print("Serving model root: ", SERVING_MODEL_DIR)


# ### Locate the eval-split examples and the latest pushed `SavedModel`
# 
# `ExampleGen` writes eval-split TFRecords under `CsvExampleGen/examples/<exec_id>/Split-eval/`.
# `Pusher` writes one timestamped `SavedModel` directory per push under
# `SERVING_MODEL_DIR`; the highest-numbered directory name is the most recent push.

# In[3]:


# --- eval-split TFRecord files ---
example_gen_root = os.path.join(PIPELINE_ROOT, "CsvExampleGen", "examples")
example_gen_runs = sorted(glob.glob(os.path.join(example_gen_root, "*")))
if not example_gen_runs:
    raise RuntimeError(
        "No CsvExampleGen output found. Run the main pipeline notebook (or "
        "pipeline.py) first so ExampleGen has produced eval-split examples."
    )
latest_example_gen_run = example_gen_runs[-1]
eval_tfrecord_files = sorted(
    glob.glob(os.path.join(latest_example_gen_run, "Split-eval", "*"))
)
print(f"Found {len(eval_tfrecord_files)} eval TFRecord file(s) under:")
print(" ", os.path.join(latest_example_gen_run, "Split-eval"))

# --- most recent pushed SavedModel ---
pushed_model_dirs = sorted(
    d for d in glob.glob(os.path.join(SERVING_MODEL_DIR, "*")) if os.path.isdir(d)
)
if not pushed_model_dirs:
    raise RuntimeError(
        "No pushed SavedModel found under SERVING_MODEL_DIR. Confirm the Pusher "
        "component ran and the model was blessed (see the main pipeline notebook's "
        "Step 8/9 output)."
    )
LATEST_MODEL_PATH = pushed_model_dirs[-1]
print("Loading model from:", LATEST_MODEL_PATH)

loaded_model = tf.saved_model.load(LATEST_MODEL_PATH)
serving_fn = loaded_model.signatures["serving_default"]
print("Serving signature inputs: ", list(serving_fn.structured_input_signature[1].keys()))
print("Serving signature outputs:", list(serving_fn.structured_outputs.keys()))


# ### Load a sample of eval examples as `tf.train.Example` protos
# 
# WIT's counterfactual and PDP panels need the raw (untransformed) `tf.Example`s —
# the same serialized protos `CsvExampleGen` wrote and that the model's
# `serving_default` signature (which wraps the Transform graph) expects directly.

# In[4]:


random.seed(RANDOM_SEED)

def load_examples(tfrecord_files, n):
    """Read tf.Example protos from TFRecords and reservoir-sample n of them."""
    examples = []
    for path in tfrecord_files:
        for raw in tf.data.TFRecordDataset(path, compression_type="GZIP"):
            ex = tf.train.Example()
            ex.ParseFromString(raw.numpy())
            examples.append(ex)
    random.shuffle(examples)
    return examples[:n]

wit_examples = load_examples(eval_tfrecord_files, NUM_DATAPOINTS)
print(f"Loaded {len(wit_examples)} examples for the What-If Tool.")
print("Feature keys on first example:", list(wit_examples[0].features.feature.keys()))


# ### Wrap the `SavedModel` in a `custom_predict_fn`
# 
# `WitConfigBuilder.set_custom_predict_fn` expects a function that takes a list of
# `tf.train.Example` protos and returns, per example, a `[P(no), P(yes)]` pair (WIT's
# convention for binary classification). We serialize each `Example`, feed the
# batch through `serving_default`, and turn the single sigmoid score into that
# two-column format.

# In[5]:


def predict_fn(examples):
    serialized = tf.constant([ex.SerializeToString() for ex in examples])
    outputs = serving_fn(examples=serialized)
    # Adjust the key below if your trainer_module.py names the output tensor
    # something other than the default Keras output name — print
    # `serving_fn.structured_outputs` above to check.
    output_key = list(outputs.keys())[0]
    scores = outputs[output_key].numpy().reshape(-1)  # P(y=1) i.e. P("yes")
    return np.stack([1.0 - scores, scores], axis=1)

# Smoke-test on a handful of examples before handing off to WIT
_test_probs = predict_fn(wit_examples[:5])
print(_test_probs)


# ## Step 1 — Launch the What-If Tool

# In[6]:


config_builder = (
    WitConfigBuilder(wit_examples)
    .set_custom_predict_fn(predict_fn)
    .set_target_feature(LABEL_KEY)
    .set_label_vocab(["no", "yes"])
    .set_model_type("classification")
)

WitWidget(config_builder, height=800)


# ## Step 2 — Feature distributions & partial dependence

# In[7]:


NUMERIC_FEATURES = ["age", "balance", "campaign", "day", "duration", "pdays", "previous"]
CATEGORICAL_FEATURES = ["job", "marital", "education", "default", "housing", "loan",
                         "contact", "month", "poutcome"]

def get_feature_values(examples, feature_name):
    """Pull one feature's raw values out of a list of tf.train.Example protos."""
    vals = []
    for ex in examples:
        f = ex.features.feature[feature_name]
        if f.HasField("int64_list") or len(f.int64_list.value):
            vals.append(f.int64_list.value[0])
        elif len(f.bytes_list.value):
            vals.append(f.bytes_list.value[0].decode("utf-8"))
        else:
            vals.append(f.float_list.value[0])
    return vals

def set_feature(example, feature_name, value):
    """Return a copy of `example` with `feature_name` overwritten to `value`."""
    new_ex = tf.train.Example()
    new_ex.CopyFrom(example)
    f = new_ex.features.feature[feature_name]
    if isinstance(value, (bytes, str)):
        f.Clear()
        f.bytes_list.value.append(value.encode("utf-8") if isinstance(value, str) else value)
    else:
        f.Clear()
        f.int64_list.value.append(int(value))
    return new_ex

def partial_dependence(examples, feature_name, sweep_values, batch_size=256):
    """Mean predicted P(yes) across `examples`, for each value in sweep_values,
    holding every other feature fixed at each example's own observed value."""
    means = []
    for v in sweep_values:
        swept = [set_feature(ex, feature_name, v) for ex in examples]
        probs = []
        for i in range(0, len(swept), batch_size):
            probs.append(predict_fn(swept[i:i + batch_size])[:, 1])
        probs = np.concatenate(probs)
        means.append(probs.mean())
    return np.array(means)


# In[8]:


def plot_pdp_numeric(ax, examples, feature_name, n_sweep=25):
    raw_vals = np.array(get_feature_values(examples, feature_name), dtype=float)
    lo, hi = np.percentile(raw_vals, [1, 99])   # trim extreme outliers (e.g. `balance`)
    sweep = np.linspace(lo, hi, n_sweep)
    pdp = partial_dependence(examples, feature_name, sweep)

    ax.hist(raw_vals, bins=30, range=(lo, hi), color="#c7d9f0", alpha=0.7, label="distribution")
    ax.set_ylabel("count", color="#6f8fbf")
    ax2 = ax.twinx()
    ax2.plot(sweep, pdp, color="#d1495b", linewidth=2, marker="o", markersize=3, label="mean P(yes)")
    ax2.set_ylabel("mean predicted P(yes)", color="#d1495b")
    ax.set_title(feature_name)
    ax.set_xlabel(feature_name)

def plot_pdp_categorical(ax, examples, feature_name):
    raw_vals = get_feature_values(examples, feature_name)
    categories, counts = np.unique(raw_vals, return_counts=True)
    order = np.argsort(-counts)
    categories, counts = categories[order], counts[order]
    pdp = partial_dependence(examples, feature_name, categories)

    x = np.arange(len(categories))
    ax.bar(x, counts, color="#c7d9f0", alpha=0.7, label="distribution")
    ax.set_ylabel("count", color="#6f8fbf")
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=45, ha="right", fontsize=8)
    ax2 = ax.twinx()
    ax2.plot(x, pdp, color="#d1495b", linewidth=2, marker="o", markersize=4, label="mean P(yes)")
    ax2.set_ylabel("mean predicted P(yes)", color="#d1495b")
    ax.set_title(feature_name)


# In[9]:


# Use a smaller subsample for the PDP sweeps — each sweep value re-scores every
# example, so cost scales as (n_examples * n_sweep_values * n_features).
PDP_SAMPLE_SIZE = 200
pdp_examples = wit_examples[:PDP_SAMPLE_SIZE]

all_features = NUMERIC_FEATURES + CATEGORICAL_FEATURES
n_cols = 4
n_rows = int(np.ceil(len(all_features) / n_cols))

fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
axes = axes.flatten()

for i, feature_name in enumerate(all_features):
    ax = axes[i]
    if feature_name in NUMERIC_FEATURES:
        plot_pdp_numeric(ax, pdp_examples, feature_name)
    else:
        plot_pdp_categorical(ax, pdp_examples, feature_name)

for j in range(len(all_features), len(axes)):
    fig.delaxes(axes[j])

fig.suptitle("Feature distribution (blue, left axis) vs. partial dependence on P(yes) (red, right axis)",
             y=1.01, fontsize=13)
fig.tight_layout()
fig.savefig("pdp_all_features.png", dpi=150, bbox_inches="tight")
plt.show()

print("Saved: pdp_all_features.png")


# ## Step 1 — Launch the What-If Tool

# In[10]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
FAIRNESS_SLICE_FEATURE = "marital"
FAIRNESS_THRESHOLDS = np.round(np.arange(0.10, 0.91, 0.05), 2)
REPORT_THRESHOLDS = [0.30, 0.50, 0.70]


def _safe_divide(numerator, denominator):
    """Return a rate without failing when a slice has no matching examples."""
    return float(numerator / denominator) if denominator else np.nan


def _threshold_metrics(labels, scores, threshold):
    """Compute binary-classification metrics at one decision threshold."""
    predictions = (scores >= threshold).astype(np.int64)
    labels = labels.astype(np.int64)

    true_positive = int(np.sum((labels == 1) & (predictions == 1)))
    true_negative = int(np.sum((labels == 0) & (predictions == 0)))
    false_positive = int(np.sum((labels == 0) & (predictions == 1)))
    false_negative = int(np.sum((labels == 1) & (predictions == 0)))

    return {
        "Examples": int(len(labels)),
        "Accuracy": _safe_divide(true_positive + true_negative, len(labels)),
        "False Positive Rate": _safe_divide(
            false_positive, false_positive + true_negative
        ),
        "False Negative Rate": _safe_divide(
            false_negative, false_negative + true_positive
        ),
        "Positive Prediction Rate": float(np.mean(predictions)),
        "Precision": _safe_divide(true_positive, true_positive + false_positive),
        "Recall": _safe_divide(true_positive, true_positive + false_negative),
    }


def build_fairness_threshold_table(
    examples,
    slice_feature=FAIRNESS_SLICE_FEATURE,
    thresholds=FAIRNESS_THRESHOLDS,
):
    """Evaluate overall and per-slice metrics across decision thresholds."""
    labels = np.asarray(get_feature_values(examples, LABEL_KEY), dtype=np.int64)
    scores = np.asarray(predict_fn(examples)[:, 1], dtype=float)
    slice_values = np.asarray(get_feature_values(examples, slice_feature))

    unexpected_labels = sorted(set(labels.tolist()) - {0, 1})
    if unexpected_labels:
        raise ValueError(
            f"Expected binary labels 0/1 in {LABEL_KEY!r}, found {unexpected_labels}."
        )

    rows = []
    for threshold in thresholds:
        overall = _threshold_metrics(labels, scores, threshold)
        rows.append(
            {
                "Feature": slice_feature,
                "Threshold": float(threshold),
                "Slice": "Overall",
                **overall,
            }
        )

        for slice_value in sorted(np.unique(slice_values)):
            mask = slice_values == slice_value
            rows.append(
                {
                    "Feature": slice_feature,
                    "Threshold": float(threshold),
                    "Slice": str(slice_value),
                    **_threshold_metrics(labels[mask], scores[mask], threshold),
                }
            )

    return pd.DataFrame(rows)


fairness_threshold_results = build_fairness_threshold_table(wit_examples)

report_table = fairness_threshold_results[
    fairness_threshold_results["Threshold"].isin(REPORT_THRESHOLDS)
].copy()

metric_columns = [
    "Feature",
    "Threshold",
    "Slice",
    "Examples",
    "Accuracy",
    "False Positive Rate",
    "False Negative Rate",
    "Positive Prediction Rate",
]

print("Fairness threshold analysis by marital status")
display(
    report_table[metric_columns]
    .sort_values(["Threshold", "Slice"])
    .style.format(
        {
            "Threshold": "{:.2f}",
            "Accuracy": "{:.4f}",
            "False Positive Rate": "{:.4f}",
            "False Negative Rate": "{:.4f}",
            "Positive Prediction Rate": "{:.4f}",
        }
    )
)


# In[11]:


group_results = fairness_threshold_results[
    fairness_threshold_results["Slice"] != "Overall"
].copy()

fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharex=True)
plot_metrics = [
    "False Positive Rate",
    "False Negative Rate",
    "Positive Prediction Rate",
]

for ax, metric in zip(axes, plot_metrics):
    for slice_name, slice_rows in group_results.groupby("Slice"):
        slice_rows = slice_rows.sort_values("Threshold")
        ax.plot(
            slice_rows["Threshold"],
            slice_rows[metric],
            marker="o",
            linewidth=2,
            label=slice_name,
        )
    ax.axvline(0.50, color="black", linestyle="--", alpha=0.65, label="0.50")
    ax.set_title(metric)
    ax.set_xlabel("Classification threshold")
    ax.set_ylabel("Rate")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.25)

handles, labels_for_legend = axes[0].get_legend_handles_labels()
fig.legend(
    handles,
    labels_for_legend,
    loc="upper center",
    ncol=len(labels_for_legend),
    bbox_to_anchor=(0.5, 1.03),
)
fig.suptitle(
    "Performance & Fairness threshold analysis by marital status",
    y=1.10,
    fontsize=14,
)
fig.tight_layout()
fig.savefig("wit_fairness_threshold_marital.png", dpi=150, bbox_inches="tight")
plt.show()


# In[12]:


print("Saved: wit_fairness_threshold_marital.png")




threshold_05 = group_results[np.isclose(group_results["Threshold"], 0.50)]

highest_fnr = threshold_05.loc[threshold_05["False Negative Rate"].idxmax()]
lowest_fnr = threshold_05.loc[threshold_05["False Negative Rate"].idxmin()]
highest_fpr = threshold_05.loc[threshold_05["False Positive Rate"].idxmax()]
lowest_fpr = threshold_05.loc[threshold_05["False Positive Rate"].idxmin()]

print("\nInterpretation at threshold 0.50")
print(
    f"Highest FNR: {highest_fnr['Slice']} "
    f"({highest_fnr['False Negative Rate']:.4f}); "
    f"lowest FNR: {lowest_fnr['Slice']} "
    f"({lowest_fnr['False Negative Rate']:.4f}); "
    f"gap: {highest_fnr['False Negative Rate'] - lowest_fnr['False Negative Rate']:.4f}."
)
print(
    f"Highest FPR: {highest_fpr['Slice']} "
    f"({highest_fpr['False Positive Rate']:.4f}); "
    f"lowest FPR: {lowest_fpr['Slice']} "
    f"({lowest_fpr['False Positive Rate']:.4f}); "
    f"gap: {highest_fpr['False Positive Rate'] - lowest_fpr['False Positive Rate']:.4f}."
)
print(
    "Compare thresholds 0.30, 0.50, and 0.70 in the table and figure. "
    "A lower threshold normally reduces false negatives but increases false "
    "positives. This is a trade-off, not proof that one group was intentionally "
    "treated unfairly."
)


# In[ ]:




