"""
Transform module for the Bank Marketing TFX pipeline.

Defines `preprocessing_fn`, which TFX's Transform component looks for by
convention (the function name is fixed; TFX imports this module and calls it).

Feature engineering performed:
  - Numeric features -> scaled to z-scores (mean 0, std 1)
  - `age` -> also bucketized into deciles as an additional engineered feature
  - Categorical string features -> integer vocabulary lookup
  - Label `y` is already binarized to 0/1 during data prep (see the CSV
    conversion step), so it is passed through unchanged here. Keeping the
    raw label numeric (rather than "yes"/"no") is what lets the Evaluator
    compute BinaryAccuracy/AUC directly against ExampleGen's raw output.
"""

import tensorflow as tf
import tensorflow_transform as tft

# ---- Feature groups (must match the CSV header in data/bank_data.csv) -----
NUMERIC_FEATURE_KEYS = [
    "age", "balance", "day", "duration", "campaign", "pdays", "previous",
]

CATEGORICAL_FEATURE_KEYS = [
    "job", "marital", "education", "default", "housing", "loan",
    "contact", "month", "poutcome",
]

LABEL_KEY = "y"

# TFX convention: transformed feature names should be different from raw
# names so the Transform graph output doesn't collide with raw input.
def _transformed_name(key):
    return key + "_xf"


def preprocessing_fn(inputs):
    """tf.transform's callback function for preprocessing inputs.

    Args:
      inputs: map from feature keys to raw not-yet-transformed features.

    Returns:
      Map from string feature key to transformed feature.
    """
    outputs = {}

    # 1. Scale numeric features to z-scores.
    for key in NUMERIC_FEATURE_KEYS:
        outputs[_transformed_name(key)] = tft.scale_to_z_score(
            tf.cast(inputs[key], tf.float32)
        )

    # 2. Extra engineered feature: bucketize age into 10 quantile buckets.
    outputs[_transformed_name("age_bucket")] = tft.bucketize(
        tf.cast(inputs["age"], tf.float32), num_buckets=10
    )

    # 3. Vocabulary lookup for categorical string features.
    for key in CATEGORICAL_FEATURE_KEYS:
        outputs[_transformed_name(key)] = tft.compute_and_apply_vocabulary(
            inputs[key], num_oov_buckets=1, vocab_filename=key
        )

    # 4. Label is already 0/1 in the raw data; just cast through.
    outputs[_transformed_name(LABEL_KEY)] = tf.cast(inputs[LABEL_KEY], tf.int64)

    return outputs
