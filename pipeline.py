"""
Reusable TFX pipeline definition for the Bank Marketing project.

This module contains an alternative pipeline configuration.
The Airflow entry point in bank_marketing_dag.py defines its own pipeline
and does not import this function.

Evaluation settings differ between the two implementations:
- This module uses an accuracy lower bound of 0.50 and permits an
  absolute accuracy decrease of 0.001 relative to the baseline.
- The Airflow DAG uses an accuracy lower bound of 0.70 and permits an
  absolute accuracy decrease of 0.01 relative to the baseline.
"""

import tensorflow_model_analysis as tfma
from tfx import v1 as tfx


def create_pipeline(
    pipeline_name: str,
    pipeline_root: str,
    data_root: str,
    transform_module_file: str,
    trainer_module_file: str,
    serving_model_dir: str,
    metadata_path: str,
    train_steps: int = 1000,
    eval_steps: int = 200,
) -> tfx.dsl.Pipeline:
    """Builds the end-to-end Bank Marketing TFX pipeline.

    Args:
      pipeline_name: name of the pipeline, used for MLMD bookkeeping.
      pipeline_root: directory for all pipeline/component outputs.
      data_root: directory containing the input CSV file(s).
      transform_module_file: path to the module with `preprocessing_fn`.
      trainer_module_file: path to the module with `run_fn`.
      serving_model_dir: directory the Pusher deploys blessed models to.
      metadata_path: path to the ML Metadata SQLite database file.
      train_steps: number of training steps for the Trainer.
      eval_steps: number of eval steps for the Trainer.

    Returns:
      A tfx.dsl.Pipeline object ready to be handed to a runner.
    """

    # ---- Step 1: ExampleGen ------------------------------------------------
    # Reads the CSV folder, splits into train/eval (default 2:1), writes
    # TFRecord-backed Example artifacts.
    example_gen = tfx.components.CsvExampleGen(input_base=data_root)

    # ---- Step 2: StatisticsGen ---------------------------------------------
    statistics_gen = tfx.components.StatisticsGen(
        examples=example_gen.outputs["examples"]
    )

    # ---- Step 3: SchemaGen --------------------------------------------------
    schema_gen = tfx.components.SchemaGen(
        statistics=statistics_gen.outputs["statistics"], infer_feature_shape=True
    )

    # ---- Step 4: ExampleValidator --------------------------------------------
    example_validator = tfx.components.ExampleValidator(
        statistics=statistics_gen.outputs["statistics"],
        schema=schema_gen.outputs["schema"],
    )

    # ---- Step 5: Transform ----------------------------------------------------
    transform = tfx.components.Transform(
        examples=example_gen.outputs["examples"],
        schema=schema_gen.outputs["schema"],
        module_file=transform_module_file,
    )

    # ---- Step 6: Trainer --------------------------------------------------------
    trainer = tfx.components.Trainer(
        module_file=trainer_module_file,
        examples=transform.outputs["transformed_examples"],
        transform_graph=transform.outputs["transform_graph"],
        schema=schema_gen.outputs["schema"],
        train_args=tfx.proto.TrainArgs(num_steps=train_steps),
        eval_args=tfx.proto.EvalArgs(num_steps=eval_steps),
    )

    # Looks up the latest blessed model in ML Metadata as the baseline.
    # When no baseline is available, baseline-change validation is skipped.
    # The candidate must still satisfy the absolute accuracy threshold.
    model_resolver = tfx.dsl.Resolver(
        strategy_class=tfx.dsl.experimental.LatestBlessedModelStrategy,
        model=tfx.dsl.Channel(type=tfx.types.standard_artifacts.Model),
        model_blessing=tfx.dsl.Channel(
            type=tfx.types.standard_artifacts.ModelBlessing
        ),
    ).with_id("latest_blessed_model_resolver")

    # ---- Step 8: Evaluator ------------------------------------------------------
    eval_config = tfma.EvalConfig(
        # Evaluator runs against the *raw* ExampleGen output and the model's
        # serving signature (which embeds the tft transform graph), so the
        # label key here is the raw column name, not the "_xf" transformed one.
        model_specs=[tfma.ModelSpec(label_key="y")],
        slicing_specs=[
            # Overall dataset slice.
            tfma.SlicingSpec(),
            # Slice over a sensitive/interesting feature: marital status.
            tfma.SlicingSpec(feature_keys=["marital"]),
        ],
        metrics_specs=[
            tfma.MetricsSpec(
                metrics=[
                    tfma.MetricConfig(class_name="BinaryAccuracy"),
                    tfma.MetricConfig(class_name="AUC"),
                    tfma.MetricConfig(
                        class_name="BinaryAccuracy",
                        threshold=tfma.MetricThreshold(
                            value_threshold=tfma.GenericValueThreshold(
                                lower_bound={"value": 0.5}
                            ),
                            change_threshold=tfma.GenericChangeThreshold(
                                direction=tfma.MetricDirection.HIGHER_IS_BETTER,
                                absolute={"value": -1e-3},
                            ),
                        ),
                    ),
                ]
            )
        ],
    )

    evaluator = tfx.components.Evaluator(
        examples=example_gen.outputs["examples"],
        model=trainer.outputs["model"],
        baseline_model=model_resolver.outputs["model"],
        eval_config=eval_config,
    )

    # ---- Step 9: Pusher --------------------------------------------------------
    pusher = tfx.components.Pusher(
        model=trainer.outputs["model"],
        model_blessing=evaluator.outputs["blessing"],
        push_destination=tfx.proto.PushDestination(
            filesystem=tfx.proto.PushDestination.Filesystem(
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

    return tfx.dsl.Pipeline(
        pipeline_name=pipeline_name,
        pipeline_root=pipeline_root,
        components=components,
        enable_cache=True,
        metadata_connection_config=tfx.orchestration.metadata.sqlite_metadata_connection_config(
            metadata_path
        ),
    )
