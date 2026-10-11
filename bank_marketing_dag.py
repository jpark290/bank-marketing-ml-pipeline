"""
TFX Pipeline for Bank Marketing Dataset (UCI Bank Marketing / bank-full.csv)
Features: age, job, marital, education, default, balance, housing, loan,
          contact, day, month, duration, campaign, pdays, previous, poutcome
          -> y (0=no subscription, 1=subscribed to term deposit)
"""
import datetime
import os

import absl
from tfx.components import (
    CsvExampleGen,
    Evaluator,
    ExampleValidator,
    Pusher,
    SchemaGen,
    StatisticsGen,
    Trainer,
    Transform,
)
from tfx.components.trainer.executor import GenericExecutor
from tfx.dsl.components.base import executor_spec
from tfx.orchestration import metadata, pipeline
from tfx.proto import pusher_pb2, trainer_pb2
from tfx.v1.dsl import Resolver
from tfx.v1.dsl.experimental import LatestBlessedModelStrategy
from tfx.types import Channel
from tfx.types.standard_artifacts import Model, ModelBlessing
import tensorflow_model_analysis as tfma
from tfx.orchestration.airflow.airflow_dag_runner import AirflowDagRunner
from tfx.orchestration.airflow.airflow_dag_runner import AirflowPipelineConfig


airflow_config = {
    "schedule_interval": None,
    "start_date": datetime.datetime(2026, 1, 1),
}

# Training hyperparameters
TRAIN_NUM_STEPS = 1000
EVAL_NUM_STEPS = 200

# ─────────────────────────────────────────────
# Pipeline constants
# ─────────────────────────────────────────────
PIPELINE_NAME = "bank_marketing_pipeline"

# Root directory for all pipeline artifacts
PIPELINE_ROOT = os.path.join(
    os.path.expanduser("~"), "COMP315", "airflow_pipeline_outputs",
    "bankmarketing_tfx_output",
)

# Where the (comma-delimited, label-binarized) CSV lives. Keep this folder
# containing ONLY bank_data.csv -- CsvExampleGen ingests every file under it.
DATA_ROOT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "data", "bank_marketing"
)
 
# preprocessing_fn and run_fn live in separate module files here.
TRANSFORM_MODULE_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "transform_module.py"
)
TRAINER_MODULE_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "trainer_module.py"
)

# SQLite metadata store
METADATA_PATH = os.path.join(
    os.path.expanduser("~"), "COMP315", "airflow_pipeline_outputs",
    "metadata", PIPELINE_NAME, "metadata.db",
)

# Where the final SavedModel is pushed
SERVING_MODEL_DIR = os.path.join(
    os.path.expanduser("~"), "COMP315", "airflow_pipeline_outputs",
    "tfx_output", "serving_model", PIPELINE_NAME,
)

LABEL_KEY = "y"


# ─────────────────────────────────────────────
# Pipeline definition
# ─────────────────────────────────────────────
def create_pipeline(
    pipeline_name: str,
    pipeline_root: str,
    data_root: str,
    transform_module_file: str,
    trainer_module_file: str,
    metadata_path: str,
    serving_model_dir: str,
    train_num_steps: int,
    eval_num_steps: int,
    enable_cache: bool = True,
) -> pipeline.Pipeline:

    # 1. Ingest CSV data
    example_gen = CsvExampleGen(input_base=data_root)

    # 2. Compute statistics over the dataset
    statistics_gen = StatisticsGen(
        examples=example_gen.outputs["examples"]
    )

    # 3. Infer schema from statistics
    schema_gen = SchemaGen(
        statistics=statistics_gen.outputs["statistics"],
        infer_feature_shape=True,
    )

    # 4. Validate examples against the schema
    example_validator = ExampleValidator(
        statistics=statistics_gen.outputs["statistics"],
        schema=schema_gen.outputs["schema"],
    )

    # 5. Apply preprocessing_fn
    transform = Transform(
        examples=example_gen.outputs["examples"],
        schema=schema_gen.outputs["schema"],
        module_file=transform_module_file,  # preprocessing_fn lives here
    )

    # 6. Train the model
    trainer = Trainer(
        module_file=trainer_module_file,  # run_fn lives here
        custom_executor_spec=executor_spec.ExecutorClassSpec(GenericExecutor),
        examples=transform.outputs["transformed_examples"],
        transform_graph=transform.outputs["transform_graph"],
        schema=schema_gen.outputs["schema"],
        train_args=trainer_pb2.TrainArgs(num_steps=train_num_steps),
        eval_args=trainer_pb2.EvalArgs(num_steps=eval_num_steps),
    )

    # 7. Resolve the latest blessed model for comparison
    model_resolver = Resolver(
        strategy_class=LatestBlessedModelStrategy,
        model=Channel(type=Model),
        model_blessing=Channel(type=ModelBlessing),
    ).with_id("latest_blessed_model_resolver")

    # 8. Evaluate model quality
    eval_config = tfma.EvalConfig(
        model_specs=[
            tfma.ModelSpec(label_key=LABEL_KEY, signature_name="serving_default")
        ],
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

    # 9. Push blessed model to serving directory
    pusher = Pusher(
        model=trainer.outputs["model"],
        model_blessing=evaluator.outputs["blessing"],
        push_destination=pusher_pb2.PushDestination(
            filesystem=pusher_pb2.PushDestination.Filesystem(
                base_directory=serving_model_dir
            )
        ),
    )

    components = [
        example_gen,
        statistics_gen,
        schema_gen,
        example_validator,
        transform,
        trainer,
        model_resolver,
        evaluator,
        pusher,
    ]

    print(f"serving model dir is {SERVING_MODEL_DIR}")
    return pipeline.Pipeline(
        pipeline_name=pipeline_name,
        pipeline_root=pipeline_root,
        components=components,
        enable_cache=enable_cache,
        metadata_connection_config=metadata.sqlite_metadata_connection_config(
            metadata_path
        ),
    )


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

absl.logging.set_verbosity(absl.logging.INFO)

# -----------------------------
# Run
# -----------------------------
DAG = AirflowDagRunner(AirflowPipelineConfig(airflow_dag_config=airflow_config)).run(
    create_pipeline(
        pipeline_name=PIPELINE_NAME,
        pipeline_root=PIPELINE_ROOT,
        data_root=DATA_ROOT,
        transform_module_file=TRANSFORM_MODULE_FILE,
        trainer_module_file=TRAINER_MODULE_FILE,
        metadata_path=METADATA_PATH,
        serving_model_dir=SERVING_MODEL_DIR,
        train_num_steps=TRAIN_NUM_STEPS,
        eval_num_steps=EVAL_NUM_STEPS,
    )
)
