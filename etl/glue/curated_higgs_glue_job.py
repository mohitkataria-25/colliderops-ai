"""AWS Glue ETL job for the curated HIGGS dataset.

This script is the Glue runtime wrapper. Reusable constants, S3 path helpers,
row-count validation, and metadata construction live in
`etl.glue.curated_higgs_transform` so they can be tested locally without the
AWS Glue runtime.

Expected Glue arguments:

--JOB_NAME
--s3_bucket
--s3_prefix
--dataset_mode
--total_rows

Example target outputs:

s3://<bucket>/<prefix>/curated/curated_higgs/training_dataset.parquet
s3://<bucket>/<prefix>/curated/curated_higgs/dataset_metadata.json
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from typing import Any

import boto3
from awsglue.context import GlueContext
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, StringType, StructField, StructType

from etl.glue.curated_higgs_transform import (
    DATASET_MODE,
    HIGGS_FEATURE_COLUMNS,
    RAW_COLUMNS,
    build_curated_higgs_paths,
    build_dataset_metadata,
    parse_total_rows,
    rows_per_label,
    validate_dataset_mode,
)


class CuratedHiggsGlueJobError(ValueError):
    """Raised when the curated HIGGS Glue job cannot complete safely."""


def get_job_args() -> dict[str, Any]:
    """Read and validate Glue job arguments."""
    args = getResolvedOptions(
        sys.argv,
        [
            "JOB_NAME",
            "s3_bucket",
            "s3_prefix",
            "dataset_mode",
            "total_rows",
        ],
    )

    return {
        "job_name": args["JOB_NAME"],
        "s3_bucket": args["s3_bucket"],
        "s3_prefix": args["s3_prefix"],
        "dataset_mode": validate_dataset_mode(args.get("dataset_mode", DATASET_MODE)),
        "total_rows": parse_total_rows(args.get("total_rows")),
    }


def build_raw_schema() -> StructType:
    """Build schema for the raw HIGGS CSV/GZIP file."""
    fields = [StructField("raw_label", DoubleType(), nullable=False)]
    fields.extend(
        StructField(feature_column, DoubleType(), nullable=True)
        for feature_column in HIGGS_FEATURE_COLUMNS
    )
    return StructType(fields)


def read_raw_higgs_dataset(
    spark: SparkSession,
    raw_input_uri: str,
) -> DataFrame:
    """Read raw HIGGS CSV/GZIP data from S3."""
    return (
        spark.read.option("header", "false")
        .option("mode", "FAILFAST")
        .schema(build_raw_schema())
        .csv(raw_input_uri)
    )


def normalize_labels(df: DataFrame) -> DataFrame:
    """Map raw HIGGS labels to background/signal strings."""
    return df.withColumn(
        "label",
        F.when(F.col("raw_label") == F.lit(1.0), F.lit("signal"))
        .when(F.col("raw_label") == F.lit(0.0), F.lit("background"))
        .otherwise(F.lit(None).cast(StringType())),
    ).drop("raw_label")


def select_balanced_sample(df: DataFrame, total_rows: int) -> DataFrame:
    """Select a balanced signal/background sample."""
    limit_per_label = rows_per_label(total_rows=total_rows)

    background_df = (
        df.filter(F.col("label") == F.lit("background"))
        .orderBy(F.rand(seed=42))
        .limit(limit_per_label)
    )
    signal_df = (
        df.filter(F.col("label") == F.lit("signal"))
        .orderBy(F.rand(seed=42))
        .limit(limit_per_label)
    )

    return background_df.unionByName(signal_df)


def count_null_features(df: DataFrame) -> int:
    """Count null values across all feature columns."""
    null_count_row = df.select(
        [
            F.sum(F.when(F.col(column).isNull(), F.lit(1)).otherwise(F.lit(0))).alias(
                column
            )
            for column in HIGGS_FEATURE_COLUMNS
        ]
    ).collect()[0]

    return int(sum(null_count_row[column] or 0 for column in HIGGS_FEATURE_COLUMNS))


def build_label_counts(df: DataFrame) -> dict[str, int]:
    """Build label-count metadata from a Spark DataFrame."""
    rows = df.groupBy("label").count().collect()
    return {str(row["label"]): int(row["count"]) for row in rows}


def validate_curated_dataset(df: DataFrame, expected_total_rows: int) -> dict[str, Any]:
    """Validate curated HIGGS data before writing outputs."""
    row_count = df.count()
    if row_count != expected_total_rows:
        raise CuratedHiggsGlueJobError(
            f"Curated row_count mismatch. Expected={expected_total_rows}, "
            f"received={row_count}."
        )

    label_counts = build_label_counts(df=df)
    null_feature_count = count_null_features(df=df)

    return build_dataset_metadata(
        row_count=row_count,
        label_counts=label_counts,
        null_feature_count=null_feature_count,
    )


def write_curated_dataset(df: DataFrame, output_uri: str) -> None:
    """Write curated HIGGS dataset as Parquet."""
    (
        df.select([*HIGGS_FEATURE_COLUMNS, "label"])
        .repartition(1)
        .write.mode("overwrite")
        .parquet(path=output_uri)
    )


def write_metadata_to_s3(
    s3_client: Any,
    bucket: str,
    metadata_key: str,
    metadata: dict[str, Any],
) -> None:
    """Write dataset metadata JSON to S3."""
    s3_client.put_object(
        Bucket=bucket,
        Key=metadata_key,
        Body=json.dumps(metadata, indent=2).encode("utf-8"),
        ContentType="application/json",
    )


def run_curated_higgs_glue_job() -> dict[str, Any]:
    """Run the curated HIGGS Glue ETL job."""
    args = get_job_args()
    spark_context = SparkContext.getOrCreate()
    glue_context = GlueContext(spark_context)
    spark = glue_context.spark_session
    s3_client = boto3.client("s3")

    paths = build_curated_higgs_paths(
        bucket=args["s3_bucket"],
        s3_prefix=args["s3_prefix"],
    )

    raw_df = read_raw_higgs_dataset(
        spark=spark,
        raw_input_uri=paths["raw_input_uri"],
    )
    normalized_df = normalize_labels(df=raw_df)
    curated_df = select_balanced_sample(
        df=normalized_df,
        total_rows=args["total_rows"],
    )

    metadata = validate_curated_dataset(
        df=curated_df,
        expected_total_rows=args["total_rows"],
    )
    metadata.update(
        {
            "job_name": args["job_name"],
            "s3_bucket": args["s3_bucket"],
            "s3_prefix": args["s3_prefix"],
            "raw_columns": RAW_COLUMNS,
            "raw_input_uri": paths["raw_input_uri"],
            "curated_output_uri": paths["curated_output_uri"],
            "curated_metadata_key": paths["curated_metadata_key"],
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
    )

    write_curated_dataset(df=curated_df, output_uri=paths["curated_output_uri"])
    write_metadata_to_s3(
        s3_client=s3_client,
        bucket=args["s3_bucket"],
        metadata_key=paths["curated_metadata_key"],
        metadata=metadata,
    )

    print(json.dumps(metadata, indent=2))
    return metadata


if __name__ == "__main__":
    run_curated_higgs_glue_job()
