"""
Glue ETL job scaffold for ColliderOpsAI.

Goal:
- Learn AWS Glue in depth by writing a Glue-style PySpark job.
- Read raw JSON/CSV collider data from a raw zone.
- Normalize nested/raw records into tabular form.
- Validate schema and data quality.
- Write processed/curated outputs for ML training and inference.

This file is intentionally scaffolded first. Implement each function step-by-step.
"""
#from pyspark.context import SparkContext
#from awsglue.context import GlueContext
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.functions import col, lower, trim
from pyspark.sql.window import Window
from pyspark.sql.dataframe import DataFrame
from etl.validate_schema import validate_collider_schema
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_FORMAT = "csv"
RAW_INPUT_PATH = PROJECT_ROOT / "data" / "raw" / "collider_events.csv"
PROCESSED_OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "collider_events"
CURATED_OUTPUT_PATH = PROJECT_ROOT / "data" / "curated" / "training_dataset"

FEATURE_COLUMNS = [
    "DER_mass_MMC",
    "DER_mass_transverse_met_lep",
    "DER_mass_vis",
    "PRI_tau_pt",
    "PRI_lep_pt",
]

CURATE_COLUMNS = [
    "event_id",
    "label",
    *FEATURE_COLUMNS
]



def create_spark_session()->SparkSession:
    return (
        SparkSession.builder
        .appName("ColliderOpsAI-ETL")
        .getOrCreate()
    )

def read_raw_data(
        spark:SparkSession,
        input_format:str,
        input_path:Path,
):
    
    normalized_input = input_format.lower().strip()

    if normalized_input == "json":
        return spark.read.json(str(input_path))
    elif normalized_input == "csv":
        return spark.read.option("header", "true").option("inferschema", "true").csv(str(input_path))
    else:
        raise ValueError(f"File format: {input_format} is not supported. Supported formats are - JSON, CSV.")


def flatten_collider_json(raw_df)->DataFrame:
    
    flattened_df = raw_df.select(
        "event_id",
        "label",
        "features.DER_mass_MMC",
        "features.DER_mass_transverse_met_lep",
        "features.DER_mass_vis",
        "features.PRI_tau_pt",
        "features.PRI_lep_pt",
    )

    return flattened_df


# Step 5: clean and transform data
# Function: transform_collider_events(df)

def transform_collider_events(df:DataFrame)->DataFrame:


    transformed_df = df

    for column_name in FEATURE_COLUMNS:

        transformed_df = transformed_df.withColumn(
            column_name,
            col(column_name).cast("double")
        )
    
    transformed_df = transformed_df.withColumn(
        "event_id",
        trim(col("event_id").cast("string"))
    )

    transformed_df = transformed_df.withColumn(
            "label",
            lower(trim(col("label")))
        )

    transformed_df = transformed_df.filter(
            col("label").isin("signal", "background")
        )

    transformed_df = transformed_df.dropna(subset=["event_id", "label", *FEATURE_COLUMNS] 
                                    )
    
    window_spec = Window.partitionBy("event_id").orderBy(col("event_id").desc())
    transformed_df = transformed_df.withColumn("rn", F.row_number().over(window=window_spec)) \
                                        .filter(col("rn") == 1) \
                                        .drop("rn") \
                                        .withColumn("ingestion_ts", F.current_timestamp())
                                        

    return transformed_df

def build_curated_training_dataset(df:DataFrame)->DataFrame:

    return df.select(*CURATE_COLUMNS)

def write_curated_data(df:DataFrame, output_path:str)->None:

     (
        df.write
        .mode("overwrite")
        .parquet(str(output_path))
      )

def write_processed_data(df:DataFrame, output_path:Path)->None:

    ( df.write
     .mode("overwrite")
     .parquet(str(output_path))
    )


def main()->None:

    spark = create_spark_session()

    try:
        raw_df = read_raw_data(
            spark=spark,
            input_path=RAW_INPUT_PATH,
            input_format=INPUT_FORMAT,
        )

        if INPUT_FORMAT == "json":
            raw_df = flatten_collider_json(raw_df=raw_df)
        
        validate_collider_schema(df=raw_df)

        processed_df = transform_collider_events(df=raw_df)
        curated_df = build_curated_training_dataset(df=processed_df)

        write_processed_data(
            df=processed_df,
            output_path=PROCESSED_OUTPUT_PATH,
        )

        write_curated_data(
            df=curated_df,
            output_path=CURATED_OUTPUT_PATH,
        )

        print(f"Processed output written to: {PROCESSED_OUTPUT_PATH}")
        print(f"Curated output writen to: {CURATED_OUTPUT_PATH}")

    finally:
        spark.stop()

if __name__ == "__main__":
    main()