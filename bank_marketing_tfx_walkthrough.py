#!/usr/bin/env python
# coding: utf-8

# ##### Bank Marketing — TFX Pipeline (Interactive Walkthrough)
# 
# This notebook builds an end-to-end **TFX** pipeline on the UCI Bank Marketing
# dataset (`bank-full.csv`), predicting whether a client subscribes to a term
# deposit (`y`), step by step, using `InteractiveContext` so you can inspect
# each artifact as it's produced.
# 
# **Environment note:** This project specifies TFX 1.12.0, whose supported
# Python range is >=3.7,<3.10. Use a Python 3.9 environment when preparing
# this version of the project.
#
# This walkthrough uses IPython/Jupyter features, including interactive
# visualizations and TensorBoard magic commands.
#
# Dependency versions are listed in requirements.txt.
# ```
# python3.9 -m venv tfx-env
# source tfx-env/bin/activate
# pip install -r requirements.txt
# ```
# 

# In[1]:


import os
import json

import tensorflow_data_validation as tfdv
from tfx.components import CsvExampleGen, StatisticsGen, SchemaGen, ExampleValidator
from tfx.orchestration.experimental.interactive.interactive_context import InteractiveContext

# Match the constants in bank_marketing_dag.py / pipeline.py
PIPELINE_NAME = "bank_marketing_pipeline"
DATA_ROOT = os.path.join(os.getcwd(), "data", "bank_marketing")
PIPELINE_ROOT = os.path.join(
    os.path.expanduser("~"), "COMP315", "interactive_pipeline_outputs", PIPELINE_NAME
)

context = InteractiveContext(pipeline_root=PIPELINE_ROOT)
print(os.getcwd())


# In[2]:


# ---- One-time data prep (only needed if data/bank_data.csv doesn't exist yet) ----
# The raw UCI file (bank-full.csv or bank.csv) is semicolon-delimited, quoted,
# and has a string label ("yes"/"no"). CsvExampleGen wants plain comma CSV,
# and we binarize the label up front so BinaryAccuracy/AUC work cleanly later.
import pandas as pd

RAW_CSV_PATH = "data/raw/bank-full.csv"  # or bank.csv for the 10% sample
OUT_CSV_PATH = os.path.join(DATA_ROOT, "bank_data.csv")

if not os.path.exists(OUT_CSV_PATH):
    df = pd.read_csv(RAW_CSV_PATH, sep=";", quotechar='"')
    df["y"] = (df["y"] == "yes").astype(int)
    df.to_csv(OUT_CSV_PATH, index=False)

print(pd.read_csv(OUT_CSV_PATH).head())


# ## Step 1 — ExampleGen: Ingest your data
# 
# `CsvExampleGen` reads every CSV under `data_root`, splits rows into
# **train/eval** (default **2:1**), and writes each split out as TFRecord-backed
# `tf.Example`s. No manual splitting needed. Around 66.7% is triain, 33.3% is eval.

# In[3]:


example_gen = CsvExampleGen(input_base=DATA_ROOT)
context.run(example_gen)

examples_artifact = example_gen.outputs["examples"].get()[0]
split_names = json.loads(examples_artifact.split_names)

print("Examples artifact base URI:", examples_artifact.uri)
print("Splits:", split_names)
for split in split_names:
    split_uri = os.path.join(examples_artifact.uri, f"Split-{split}")
    print(f"  {split} split URI: {split_uri}")


# ## Step 2 — StatisticsGen: Compute data statistics
# 
# `StatisticsGen` computes per-feature statistics (min/max/mean, missing-value
# counts, value-distribution histograms) for both splits.

# In[4]:


statistics_gen = StatisticsGen(examples=example_gen.outputs["examples"])
context.run(statistics_gen)

stats_artifact = statistics_gen.outputs["statistics"].get()[0]
print("Statistics artifact URI:", stats_artifact.uri)

# Renders the interactive Facets visualization (train vs eval overlaid)
context.show(statistics_gen.outputs["statistics"])


# ## Step 3 — SchemaGen: Infer the data schema
# 
# `SchemaGen` automatically infers a schema (feature types, expected ranges,
# categorical vocabularies) from the statistics computed above.

# In[5]:


schema_gen = SchemaGen(
    statistics=statistics_gen.outputs["statistics"], infer_feature_shape=True
)
context.run(schema_gen)

context.show(schema_gen.outputs["schema"])


# In[6]:


# Programmatic summary of the inferred schema (for the "short summary" write-up)
schema_artifact_uri = schema_gen.outputs["schema"].get()[0].uri
schema_path = os.path.join(schema_artifact_uri, "schema.pbtxt")
schema = tfdv.load_schema_text(schema_path)

print(f"Schema file: {schema_path}")
print(f"Total features in schema: {len(schema.feature)}\n")
for feature in schema.feature:
    ftype = feature.type
    domain = feature.WhichOneof("domain_info")
    print(f"  - {feature.name:15s} type={ftype!s:10s} domain={domain}")


# ## Statistics Summary (StatisticsGen)
# 
# StatisticsGen computed per-feature statistics over the training split (30,105 rows, ~2/3 of the 45,211-row dataset). All 17 features report 0% missing values, confirming the CSV-to-comma-delimited data prep introduced no parsing errors or nulls. The target label y shows a strong class imbalance: 88.26% of examples are 0 (no subscription), with only ~11.7% positive — a signal that accuracy alone will be a misleading metric and Precision/Recall/AUC should be weighted more heavily during evaluation. The feature previous is 81.67% zeros, mirroring the pdays = -1 sentinel ("never previously contacted") that dominates the same proportion of rows — these two features largely encode the same information. Several numeric features show heavy skew and outliers worth flagging: balance ranges from -8,019 (overdraft) to 102k with a mean of 1,376 and a std dev (3,141) larger than the mean; campaign has a max of 58 contacts against a median of 2. Categorical distributions look reasonable, though contact (~29% "unknown") and poutcome (~82% "unknown") carry large "unknown" buckets that limit their information content. Notably, duration (call length in seconds) is only known after a call takes place and is strongly correlated with the outcome — a likely source of label leakage that limits the model's real-world deployability for pre-call targeting, worth calling out as a known limitation.

# ## Schema Summary (SchemaGen)
# 
# SchemaGen inferred a schema of 17 required features with no domain-of-missingness, consistent with the 0% missing values seen in StatisticsGen. Type inference matched expectations exactly: the 7 numeric fields (age, balance, campaign, day, duration, pdays, previous) plus the label y were inferred as INT, and the 9 categorical fields (job, marital, education, default, housing, loan, contact, month, poutcome) were inferred as STRING with fixed vocabularies. All inferred vocabulary sizes (e.g., 12 for job, 12 for month, 4 for poutcome) fall safely within the embedding-table ceilings hardcoded in the Trainer module, so no out-of-range embedding indices are expected at train time. One limitation worth noting: SchemaGen treats 'unknown' as a legitimate category rather than a semantic missing value across contact, education, job, and poutcome — meaning ExampleValidator will not flag poutcome's ~82% "unknown" rate as anomalous, since it's schema-valid. Any judgment about whether that level of missingness is acceptable requires domain knowledge beyond what automatic schema inference can provide.

# ## Step 4 — ExampleValidator: Detect data anomalies
# 
# `ExampleValidator` compares the actual statistics against the inferred
# schema and flags anomalies (missing values, unexpected types, out-of-range
# values, unseen categories, etc.).

# In[7]:


example_validator = ExampleValidator(
    statistics=statistics_gen.outputs["statistics"],
    schema=schema_gen.outputs["schema"],
)
context.run(example_validator)

context.show(example_validator.outputs["anomalies"])


# In[8]:


# Load every anomalies file under the artifact (one per split) and print + health-check.
anomalies_root = example_validator.outputs["anomalies"].get()[0].uri
print("Anomalies artifact URI:", anomalies_root)

pipeline_healthy = True

for root, _, files in os.walk(anomalies_root):
    for fname in files:
        if not fname.endswith((".pb", ".pbtxt")):
            continue
        fpath = os.path.join(root, fname)
        try:
            anomalies = tfdv.load_anomalies_text(fpath)
        except Exception:
            continue

        split_label = os.path.basename(root)
        print(f"\n--- Split: {split_label} ({fpath}) ---")
        if len(anomalies.anomaly_info) == 0:
            print("  No anomalies detected.")
        else:
            pipeline_healthy = False
            for feature_name, info in anomalies.anomaly_info.items():
                print(f"  Feature '{feature_name}': {info.short_description}")
                print(f"    -> {info.description}")

print("\n==============================")
if pipeline_healthy:
    print("Data pipeline is HEALTHY: no anomalies found in any split.")
else:
    print("Data pipeline has ANOMALIES: review the details above before training.")
print("==============================")


# In[9]:


import tensorflow_model_analysis as tfma
from tfx.components import Transform, Trainer, Evaluator, Pusher
from tfx.components.trainer.executor import GenericExecutor
from tfx.dsl.components.base import executor_spec
from tfx.dsl.components.common.resolver import Resolver
from tfx.dsl.experimental import latest_blessed_model_resolver
from tfx.types import Channel
from tfx.types.standard_artifacts import Model, ModelBlessing
from tfx.proto import trainer_pb2, pusher_pb2

# Module files (same ones bank_marketing_dag.py / pipeline.py point to)
TRANSFORM_MODULE_FILE = os.path.join(os.getcwd(), "transform_module.py")
TRAINER_MODULE_FILE = os.path.join(os.getcwd(), "trainer_module.py")

# Training hyperparameters -- kept small here for a fast interactive smoke
# test. Use TRAIN_NUM_STEPS/EVAL_NUM_STEPS from bank_marketing_dag.py (1000/200)
# for your real graded run; these are just for exploring in this notebook.
TRAIN_NUM_STEPS = 100 # was 200/50
EVAL_NUM_STEPS = 20

SERVING_MODEL_DIR = os.path.join(
    os.path.expanduser("~"), "COMP315", "interactive_pipeline_outputs",
    "serving_model", PIPELINE_NAME,
)

LABEL_KEY = "y"


# ## Step 5 — Transform: Feature engineering
# 
# `Transform` applies a `preprocessing_fn` (TensorFlow Transform) to every
# example: numeric features are scaled to z-scores, `age` is additionally
# bucketized, and categorical features get vocabulary-based integer lookups.
# The module file lives at `modules/transform_module.py` (see repo/output
# files) — the cell below just points to it; open that file to read the full
# `preprocessing_fn`.

# In[10]:


transform = Transform(
    examples=example_gen.outputs["examples"],
    schema=schema_gen.outputs["schema"],
    module_file=TRANSFORM_MODULE_FILE,
)
context.run(transform)

print("Transformed examples URI:", transform.outputs["transformed_examples"].get()[0].uri)
print("Transform graph URI:", transform.outputs["transform_graph"].get()[0].uri)


# ## Step 6 — Trainer: Train your model
# 
# `Trainer` calls `run_fn` from `modules/trainer_module.py`, which builds a
# Keras binary classifier over the transformed features, trains it with a
# `TensorBoard` callback pointed at `fn_args.model_run_dir`, and exports a
# `SavedModel` with a serving signature that accepts *raw* (untransformed)
# `tf.Example`s (the tft graph is baked into the signature).
# 
# After this cell finishes, run the TensorBoard cell below and take your screenshots of the Scalars tab (loss / binary_accuracy / auc, train vs. eval) for the rubric's TensorBoard Interpretation section.

# In[18]:


trainer = Trainer(
    module_file=TRAINER_MODULE_FILE,
    custom_executor_spec=executor_spec.ExecutorClassSpec(GenericExecutor),
    examples=transform.outputs["transformed_examples"],
    transform_graph=transform.outputs["transform_graph"],
    schema=schema_gen.outputs["schema"],
    train_args=trainer_pb2.TrainArgs(num_steps=TRAIN_NUM_STEPS),
    eval_args=trainer_pb2.EvalArgs(num_steps=EVAL_NUM_STEPS),
)

context.run(trainer)

model_run_uri = trainer.outputs["model_run"].get()[0].uri
print(model_run_uri)


# In[19]:


get_ipython().run_line_magic('load_ext', 'tensorboard')
get_ipython().run_line_magic('tensorboard', '--logdir {model_run_uri}')


# ## Step 7 — Resolver: Find the latest blessed model
# 
# `Resolver` (with `LatestBlessedModelStrategy`) looks up the most recently
# **blessed** model in ML Metadata to use as the Evaluator's baseline.
# When no baseline is available, baseline-change validation is skipped.
# The candidate must still satisfy the absolute accuracy threshold.

# In[20]:


model_resolver = Resolver(
    strategy_class=latest_blessed_model_resolver.LatestBlessedModelResolver,
    model=Channel(type=Model),
    model_blessing=Channel(type=ModelBlessing),
)
context.run(model_resolver)

print("Resolver state:", model_resolver.outputs["model"].get())
print("(Empty list is expected on the very first run -- no prior blessed model exists yet.)")


# ## Step 8 — Evaluator: Evaluate with TFMA
# 
# We configure an `EvalConfig` with:
# - a `ModelSpec` pointing to label key `y`,
# - two `SlicingSpec`s — the overall dataset, and a slice over `marital`
#   status (an interesting/sensitive feature for a marketing model),
# - `MetricsSpec`s with `BinaryAccuracy` and `AUC` (plus a value/change
#   threshold on accuracy that gates blessing).

# In[21]:


eval_config = tfma.EvalConfig(
    model_specs=[tfma.ModelSpec(label_key=LABEL_KEY, signature_name="serving_default")],
    slicing_specs=[
        tfma.SlicingSpec(),                          # overall
        tfma.SlicingSpec(feature_keys=["marital"]),  # sensitive/interesting slice
    ],
    metrics_specs=[
        tfma.MetricsSpec(
            metrics=[
                tfma.MetricConfig(class_name="AUC"),
                tfma.MetricConfig(class_name="ExampleCount"),
                tfma.MetricConfig(class_name="Precision"),
                tfma.MetricConfig(class_name="Recall"),
                tfma.MetricConfig(
                    class_name="BinaryAccuracy",
                    threshold=tfma.MetricThreshold(
                        value_threshold=tfma.GenericValueThreshold(
                            lower_bound={"value": 0.70}
                        ),
                        change_threshold=tfma.GenericChangeThreshold(
                            direction=tfma.MetricDirection.HIGHER_IS_BETTER,
                            absolute={"value": -0.01},
                        ),
                    ),
                ),
            ]
        )
    ],
)

evaluator = Evaluator(
    examples=example_gen.outputs["examples"],
    model=trainer.outputs["model"],
    baseline_model=model_resolver.outputs["model"],
    eval_config=eval_config,
)

context.run(evaluator)

context.show(evaluator.outputs["evaluation"])
# evaluator = Evaluator(
#     examples=example_gen.outputs["examples"],
#     model=trainer.outputs["model"],
#     baseline_model=model_resolver.outputs["model"],
#     eval_config=eval_config,
# ).with_id("evaluator_run_3")

# context.run(evaluator)

# context.show(evaluator.outputs["evaluation"])


# In[22]:


# Blessing check + a printed summary of the model this run produced
blessing_uri = evaluator.outputs["blessing"].get()[0].uri
blessed = os.path.isfile(os.path.join(blessing_uri, "BLESSED"))
print("Blessing artifact URI:", blessing_uri)
print("Model blessed:", blessed)
if not blessed and os.path.isfile(os.path.join(blessing_uri, "NOT_BLESSED")):
    print("Reason: did not clear the BinaryAccuracy threshold (>= 0.70, or did not beat baseline by >= -0.01).")


# ## Step 9 — Pusher: Deploy the blessed model
# 
# `Pusher` copies the trained model to `serving_model_dir` **only if** the
# Evaluator blessed it.

# In[23]:


pusher = Pusher(
    model=trainer.outputs["model"],
    model_blessing=evaluator.outputs["blessing"],
    push_destination=pusher_pb2.PushDestination(
        filesystem=pusher_pb2.PushDestination.Filesystem(
            base_directory=SERVING_MODEL_DIR
        )
    ),
)
context.run(pusher)

pushed_model_uri = pusher.outputs["pushed_model"].get()[0].uri
pushed = pusher.outputs["pushed_model"].get()[0].get_custom_property("pushed")
print("Pushed model artifact URI:", pushed_model_uri)
print("Was the model actually pushed to the serving dir?:", bool(pushed))
if bool(pushed):
    print("Serving model directory:", SERVING_MODEL_DIR)
else:
    print("Not pushed -- model was not blessed by the Evaluator (see Step 8 output above).")


# ## Step 10 — `create_pipeline` function
# 
# A separate `create_pipeline(...)` function is defined in `pipeline.py`.
# It returns a `tfx.dsl.Pipeline` containing the same main component types,
# but its evaluation settings differ from this interactive walkthrough.
#
# The Airflow DAG in `bank_marketing_dag.py` defines its own pipeline
# function and does not import the function from `pipeline.py`.
# The cell below inspects the separate function's signature.

# In[24]:


import inspect
from pipeline import create_pipeline

print(inspect.signature(create_pipeline))


# In[ ]:





# In[ ]:





# In[ ]:




