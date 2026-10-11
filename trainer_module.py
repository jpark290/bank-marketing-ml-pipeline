"""
Trainer module for the Bank Marketing TFX pipeline.

Defines `run_fn`, which TFX's Trainer component looks for by convention.
Builds a Keras binary classifier on top of the tf.Transform-ed features,
trains it, logs to TensorBoard, and exports a SavedModel with a serving
signature that accepts raw (untransformed) CSV-style examples.
"""

import tensorflow as tf
import tensorflow_transform as tft
from tfx_bsl.public import tfxio
from tensorflow_transform.tf_metadata import schema_utils

# Rough vocab-size ceiling; tft will have written the real vocab files,
# these sizes just need to be >= the true number of distinct values.
_MAX_CATEGORICAL_FEATURE_VALUES = {
    "job": 15, "marital": 5, "education": 6, "default": 3, "housing": 3,
    "loan": 3, "contact": 5, "month": 15, "poutcome": 6,
}
NUMERIC_FEATURE_KEYS = [
    "age", "balance", "day", "duration", "campaign", "pdays", "previous",
]

CATEGORICAL_FEATURE_KEYS = [
    "job", "marital", "education", "default", "housing", "loan",
    "contact", "month", "poutcome",
]

LABEL_KEY = "y"
# _TRAIN_BATCH_SIZE = 64
# _EVAL_BATCH_SIZE = 64
_TRAIN_BATCH_SIZE = 128
_EVAL_BATCH_SIZE = 128
_LEARNING_RATE = 5e-4
_DENSE_UNITS = (128, 64, 16)
_DROPOUT_RATE = 0.2
_CLASS_WEIGHT = {0: 1.0, 1: 4.0}
_EPOCHS = 10

def _transformed_name(key):
    return key + "_xf"


def _build_input_fn(file_pattern, data_accessor, tf_transform_output, batch_size):
    """Builds a tf.data.Dataset of transformed, batched examples."""
    dataset = data_accessor.tf_dataset_factory(
        file_pattern,
        _dataset_options(batch_size, _transformed_name(LABEL_KEY)),
        tf_transform_output.transformed_metadata.schema,
    )
    return dataset.repeat()


def _dataset_options(batch_size, label_key):

    return tfxio.TensorFlowDatasetOptions(
        batch_size=batch_size,
        label_key=label_key,
    )


def _build_keras_model(tf_transform_output):
    """Builds a small feed-forward Keras binary classifier."""
    feature_spec = tf_transform_output.transformed_feature_spec()
    feature_spec.pop(_transformed_name(LABEL_KEY))

    inputs = {}
    numeric_inputs = []
    for key in NUMERIC_FEATURE_KEYS:
        inputs[_transformed_name(key)] = tf.keras.Input(
            shape=(1,), name=_transformed_name(key), dtype=tf.float32
        )
        numeric_inputs.append(inputs[_transformed_name(key)])

    inputs[_transformed_name("age_bucket")] = tf.keras.Input(
        shape=(1,), name=_transformed_name("age_bucket"), dtype=tf.int64
    )
    age_bucket_embed = tf.keras.layers.Embedding(10, 4)(
        inputs[_transformed_name("age_bucket")]
    )
    age_bucket_embed = tf.keras.layers.Flatten()(age_bucket_embed)

    categorical_embeds = []
    for key in CATEGORICAL_FEATURE_KEYS:
        vocab_size = _MAX_CATEGORICAL_FEATURE_VALUES[key] + 1  # +1 for OOV
        inputs[_transformed_name(key)] = tf.keras.Input(
            shape=(1,), name=_transformed_name(key), dtype=tf.int64
        )
        embed_dim = max(2, min(8, vocab_size // 2))
        embed = tf.keras.layers.Embedding(vocab_size, embed_dim)(
            inputs[_transformed_name(key)]
        )
        embed = tf.keras.layers.Flatten()(embed)
        categorical_embeds.append(embed)

    concat = tf.keras.layers.concatenate(
        numeric_inputs + [age_bucket_embed] + categorical_embeds
    )
    dense = tf.keras.layers.Dense(128, activation="relu")(concat)
    dense = tf.keras.layers.Dropout(0.2)(dense)
    dense = tf.keras.layers.Dense(64, activation="relu")(dense)
    dense = tf.keras.layers.Dropout(0.2)(dense)
    dense = tf.keras.layers.Dense(16, activation="relu")(dense)
    dense = tf.keras.layers.Dropout(0.2)(dense)
    output = tf.keras.layers.Dense(1, activation="sigmoid", name="output")(dense)

    model = tf.keras.Model(inputs=inputs, outputs=output)
    model.compile(
        loss="binary_crossentropy",
        optimizer=tf.keras.optimizers.Adam(learning_rate=5e-4), # HERE
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="binary_accuracy"),
            tf.keras.metrics.AUC(name="auc"),
        ],
    )
    model.summary()
    return model


def _get_serve_tf_examples_fn(model, tf_transform_output):
    """Returns a function that parses raw tf.Examples and applies the tft graph."""
    model.tft_layer = tf_transform_output.transform_features_layer()

    @tf.function
    def serve_tf_examples_fn(serialized_tf_examples):
        feature_spec = tf_transform_output.raw_feature_spec()
        feature_spec.pop(LABEL_KEY, None)
        parsed_features = tf.io.parse_example(serialized_tf_examples, feature_spec)
        transformed_features = model.tft_layer(parsed_features)
        return model(transformed_features)

    return serve_tf_examples_fn


def run_fn(fn_args):
    """TFX Trainer entry point.

    Args:
      fn_args: tfx.components.trainer.fn_args_utils.FnArgs holding args for
        training, e.g. train/eval file paths, model dir, hyperparameters.
    """
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_graph_path)

    train_dataset = _build_input_fn(
        fn_args.train_files, fn_args.data_accessor, tf_transform_output,
        _TRAIN_BATCH_SIZE,
    )
    eval_dataset = _build_input_fn(
        fn_args.eval_files, fn_args.data_accessor, tf_transform_output,
        _EVAL_BATCH_SIZE,
    )

    model = _build_keras_model(tf_transform_output)

    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=fn_args.model_run_dir,
        update_freq="epoch"
    )
    print("TRAIN STEPS:", fn_args.train_steps)
    print("EVAL STEPS:", fn_args.eval_steps)
    print("EPOCHS: 10")

    model.fit(
        train_dataset,
        steps_per_epoch=fn_args.train_steps,
        epochs=10,
        validation_data=eval_dataset,
        validation_steps=fn_args.eval_steps,
        callbacks=[tensorboard_callback],
        class_weight=_CLASS_WEIGHT,
    )

    signatures = {
        "serving_default": _get_serve_tf_examples_fn(
            model, tf_transform_output
        ).get_concrete_function(
            tf.TensorSpec(shape=[None], dtype=tf.string, name="examples")
        ),
    }
    model.save(fn_args.serving_model_dir, save_format="tf", signatures=signatures)
