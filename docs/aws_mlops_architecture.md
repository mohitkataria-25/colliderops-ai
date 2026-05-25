

# ColliderOpsAI AWS MLOps Architecture

## Purpose

This document describes the target AWS architecture for ColliderOpsAI.

ColliderOpsAI currently runs as a local AI/MLOps research workbench with two dataset tracks:

1. `real_cern` — raw CERN/Open Data ROOT ingestion, local caching, branch probing, lightweight feature extraction, and curated ML dataset generation.
2. `curated_higgs` — an ML-ready HIGGS benchmark dataset path for stronger tabular model benchmarking and evaluation.

The AWS architecture extends the local workflow into a cloud-ready platform that can support reproducible ingestion, ETL, training, evaluation, model registration, API serving, RAG documentation, and agentic workflow orchestration.

This architecture is intended as a production-style blueprint. It does not assume that every component must be deployed immediately.

---

## Architecture Goals

The cloud architecture should support:

- Reproducible dataset ingestion from public scientific sources.
- Clear separation of raw, processed, and curated datasets.
- Schema validation and dataset-quality checks before model training.
- Batch ETL using AWS Glue / PySpark.
- Lightweight metadata and orchestration tasks using AWS Lambda.
- End-to-end orchestration using AWS Step Functions, with Airflow as an optional alternative.
- Model training, evaluation, and artifact tracking.
- Model serving through a FastAPI service hosted on ECS/Fargate.
- RAG-based project documentation and model/dataset explainability.
- MLflow-based experiment tracking.
- Auditability through run metadata, model cards, evaluation reports, and dataset cards.

---

## High-Level Architecture

```text
External data sources
  - CERN Open Data ROOT files
  - Curated HIGGS benchmark dataset
  - Future physics ML datasets
  - Project documentation / model cards / dataset cards
        |
        v
S3 Raw Zone
  s3://colliderops-ai/raw/
        |
        v
AWS Glue / PySpark ETL
        |
        v
S3 Processed Zone
  s3://colliderops-ai/processed/
        |
        v
S3 Curated Zone
  s3://colliderops-ai/curated/
        |
        v
Training + Evaluation Jobs
        |
        +--------------------+
        |                    |
        v                    v
Model Artifacts          Evaluation Reports
S3 / MLflow              S3 / MLflow
        |
        v
Model Registry / Metadata Store
        |
        v
FastAPI Inference Service
ECS/Fargate + ALB/API Gateway
        |
        v
Client / Researcher / Agent Interface
```

---

## Data Lake Layout

Recommended S3 layout:

```text
s3://colliderops-ai/
  raw/
    real_cern/
      dataset_registry.json
      root_files/
        record_id=7901/
        record_id=7779/
    curated_higgs/
      HIGGS.csv.gz
      higgs_sample.csv

  processed/
    real_cern/
      events.parquet
      run_metadata.json
    curated_higgs/
      normalized_sample.parquet
      run_metadata.json

  curated/
    real_cern/
      training_dataset.parquet
    curated_higgs/
      training_dataset.parquet

  models/
    real_cern/
      logistic_regression.joblib
      random_forest.joblib
      hist_gradient_boosting.joblib
    curated_higgs/
      logistic_regression.joblib
      random_forest.joblib
      hist_gradient_boosting.joblib

  evaluation/
    real_cern/
      model_comparison.csv
      evaluation_summary.md
      confusion_matrices/
      classification_reports/
      feature_importance/
    curated_higgs/
      model_comparison.csv
      evaluation_summary.md
      confusion_matrices/
      classification_reports/
      feature_importance/

  docs/
    model_card.md
    evaluation_card.md
    data_dictionary.md
    leakage_review.md
    aws_mlops_architecture.md

  rag/
    source_docs/
    vector_index/
```

---

## Dataset Tracks

### 1. `real_cern`

Purpose:

- Demonstrate raw scientific data ingestion.
- Download/cache CERN ROOT files.
- Inspect ROOT trees and branches.
- Extract lightweight event-level features.
- Preserve lineage from CERN record to ROOT file to curated training row.

Current local workflow:

```text
CERN record metadata
  -> ROOT file URL discovery
  -> local ROOT cache
  -> branch probing
  -> readable feature extraction
  -> processed CSV
  -> curated training CSV
  -> training/evaluation
```

AWS target workflow:

```text
CERN Open Data source
  -> S3 raw/root_files/
  -> Glue ETL reads ROOT-derived or converted intermediate data
  -> S3 processed/events.parquet
  -> S3 curated/training_dataset.parquet
  -> model training and evaluation
```

Important note:

ROOT parsing is not always straightforward inside Glue because CMS AODSIM files can include complex C++/EDM objects. For production-style AWS workflows, the preferred path is:

1. Cache/download ROOT files.
2. Run a controlled extraction/conversion step.
3. Write standardized Parquet outputs to S3.
4. Use Glue/PySpark downstream on Parquet rather than repeatedly parsing complex ROOT files.

---

### 2. `curated_higgs`

Purpose:

- Provide an ML-ready particle-physics benchmark path.
- Support stronger model benchmarking than the lightweight `real_cern` feature set.
- Demonstrate tabular ML workflows with richer physics-inspired features.

Current local workflow:

```text
UCI HIGGS dataset
  -> local compressed source cache
  -> balanced sample extraction
  -> curated training dataset
  -> Logistic Regression / Random Forest / HistGradientBoosting
  -> model comparison, ROC-AUC, confusion matrices, feature importance
```

AWS target workflow:

```text
HIGGS.csv.gz
  -> S3 raw/curated_higgs/
  -> Glue ETL validates schema and normalizes labels
  -> S3 curated/curated_higgs/training_dataset.parquet
  -> training/evaluation jobs
  -> reports and model artifacts
```

---

## Core AWS Services

### Amazon S3

S3 is the system of record for:

- Raw datasets.
- Processed datasets.
- Curated model-ready datasets.
- Model artifacts.
- Evaluation reports.
- Dataset cards and model cards.
- RAG source documents and vector index artifacts.

S3 should follow a medallion-style layout:

```text
raw -> processed -> curated
```

---

### AWS Glue

Glue is used for batch ETL and schema-standardization work.

Recommended Glue jobs:

```text
curated_higgs_glue_job.py
  - Reads HIGGS source data from S3 raw zone.
  - Validates expected 28 feature columns plus label.
  - Maps labels to signal/background.
  - Writes curated Parquet output.

real_cern_glue_job.py
  - Reads extracted CERN event features from S3 processed zone.
  - Validates feature schema.
  - Drops leakage-prone columns.
  - Writes curated Parquet output.
```

Glue should not be the first place where complex CMS ROOT files are parsed. The safer pattern is to parse ROOT files in a controlled extraction step, then use Glue on standardized tabular outputs.

---

### AWS Lambda

Lambda is used for lightweight tasks, not heavy training or ROOT parsing.

Recommended Lambda functions:

```text
validate_dataset_metadata
  - Check that expected files exist in S3.
  - Check dataset mode and row-count expectations.
  - Validate required metadata fields.

check_etl_outputs
  - Verify that Glue wrote expected processed/curated outputs.
  - Validate output size and schema metadata.

register_model_metadata
  - Write model metadata to S3/DynamoDB after evaluation.
  - Store dataset mode, model version, metrics path, and model artifact path.

notify_pipeline_status
  - Send completion/failure messages through SNS, Slack webhook, or email.
```

Avoid Lambda for:

- Large ROOT extraction.
- Heavy ETL.
- Model training.
- Long-running evaluation.

---

### AWS Step Functions

Step Functions orchestrates the end-to-end ML pipeline.

Recommended state machine:

```text
Start
  -> ValidateDatasetMetadata Lambda
  -> Run Glue ETL Job
  -> Check ETL Outputs Lambda
  -> Run Training Job
  -> Run Evaluation Job
  -> Register Model Metadata Lambda
  -> Notify Completion Lambda
End
```

Failure paths should capture:

- Dataset validation failures.
- Glue job failures.
- Missing curated dataset outputs.
- Training failures.
- Evaluation metric thresholds not met.
- Model registration failures.

---

### Amazon ECS / Fargate

ECS/Fargate hosts the FastAPI application.

Recommended service responsibilities:

- `/health` — service health check.
- `/predict` — single event prediction.
- `/batch-predict` — batch prediction from JSON records.
- `/batch-predict-file` — uploaded CSV prediction.
- `/model-metadata` — active model and dataset details.
- `/ask` — RAG-based project/documentation Q&A.
- `/agent` — LangGraph workflow routing endpoint.

Model artifacts can be loaded from:

- container image for simple portfolio deployment, or
- S3/model registry for a more production-like deployment.

---

### MLflow

MLflow is used for experiment tracking.

Recommended logged items:

- Dataset mode.
- Feature columns.
- Train/test split parameters.
- Model type and hyperparameters.
- Accuracy, precision, recall, F1, ROC-AUC.
- Confusion matrix CSVs.
- Classification report CSVs.
- Feature importance and permutation importance CSVs.
- Evaluation summary markdown.
- Model artifacts.

Potential backends:

```text
Local MLflow for development.
Remote MLflow tracking server for cloud.
S3 artifact store.
RDS/Postgres backend store if needed.
```

---

### RAG Documentation Layer

RAG should support the workbench by answering questions about:

- Dataset sources.
- Feature definitions.
- Leakage decisions.
- Evaluation results.
- Model cards.
- How to run ETL/training/evaluation.
- AWS pipeline design.

Recommended RAG documents:

```text
docs/model_card.md
docs/evaluation_card.md
docs/data_dictionary.md
docs/leakage_review.md
docs/aws_mlops_architecture.md
docs/runbooks/run_real_cern_etl.md
docs/runbooks/train_and_evaluate.md
```

Possible vector storage options:

- Chroma for local development.
- OpenSearch Serverless for AWS-native retrieval.
- S3-backed vector artifact for lightweight portfolio deployment.

---

## Pipeline Design

### Local Pipeline

```text
python -m etl.download_curated_higgs_dataset
python -m etl.real_cern_etl_job
python -m training.train --dataset-mode curated_higgs
python -m training.evaluate --dataset-mode curated_higgs
python -m training.train --dataset-mode real_cern
python -m training.evaluate --dataset-mode real_cern
```

### AWS Pipeline

```text
1. Upload/register dataset source in S3 raw zone.
2. Validate dataset metadata with Lambda.
3. Run Glue ETL to produce curated Parquet dataset.
4. Start training job.
5. Save models to S3 and log to MLflow.
6. Run evaluation job.
7. Write metrics and reports to S3.
8. Register model metadata.
9. Deploy or mark model as candidate.
10. Notify pipeline completion.
```

---

## Training and Evaluation Architecture

Training jobs should support dataset modes:

```text
sample_collider
real_cern
curated_higgs
```

Each training job should:

- Load curated dataset from S3 or local path.
- Select feature columns by dataset mode.
- Validate required columns.
- Check leakage-prone columns.
- Split train/test with fixed random seed.
- Train supported models.
- Save models.
- Log metadata and artifacts to MLflow.

Evaluation jobs should:

- Load saved models.
- Recreate test split deterministically.
- Generate predictions.
- Write model comparison metrics.
- Write confusion matrix reports.
- Write classification reports.
- Write feature importance / permutation importance reports.
- Write evaluation summary markdown.
- Log artifacts to MLflow.

---

## Model Serving Architecture

Recommended serving path:

```text
Model artifacts in S3
  -> FastAPI service loads active model
  -> API Gateway or ALB routes requests
  -> ECS/Fargate runs containerized app
  -> CloudWatch captures logs and metrics
```

Serving endpoints:

```text
GET  /health
GET  /model-metadata
POST /predict
POST /batch-predict
POST /batch-predict-file
POST /ask
POST /agent
```

For portfolio v1, serving one active model per dataset mode is enough.

For a more production-like version, add:

- Active model registry.
- Model version selection.
- Input schema validation.
- Confidence thresholds.
- Low-confidence routing.
- Prediction logging.

---

## Monitoring and Observability

Recommended monitoring:

- CloudWatch logs for Lambda, Glue, ECS, and Step Functions.
- Step Functions execution history for orchestration debugging.
- Glue job metrics for ETL runtime and failures.
- API latency and error-rate metrics from ECS/ALB/API Gateway.
- Model prediction logs for monitoring confidence distributions.
- Evaluation reports for offline model quality.

Future production extensions:

- Data drift checks.
- Prediction drift checks.
- Feature distribution monitoring.
- Scheduled retraining triggers.
- Model performance regression alerts.

---

## Security and Governance

Recommended controls:

- Least-privilege IAM roles for Glue, Lambda, ECS, and Step Functions.
- S3 bucket encryption.
- S3 bucket versioning for datasets and models.
- Separate S3 prefixes for raw, processed, curated, models, and reports.
- No hardcoded credentials.
- Environment variables or AWS Secrets Manager for service configuration.
- CloudWatch audit logs.
- Optional VPC endpoints for private AWS service access.

Dataset governance:

- Dataset cards for each dataset mode.
- Model cards for each deployed model.
- Evaluation summaries for each model run.
- Run metadata for each ETL and training job.
- Clear leakage policy for excluded features.

---

## Near-Term Implementation Plan

### Phase 1: Documentation and architecture

- Add this AWS architecture document.
- Add orchestration plan.
- Add Step Functions skeleton.
- Add Glue job skeletons.

### Phase 2: S3-compatible local paths

- Add config layer for local vs S3 paths.
- Use environment variables for bucket/prefix settings.
- Keep local mode as default.

### Phase 3: Glue ETL skeleton

- Build `curated_higgs_glue_job.py`.
- Read raw HIGGS source from S3.
- Validate schema and write curated Parquet.

### Phase 4: Orchestration skeleton

- Add Step Functions ASL definition.
- Add Lambda metadata validator skeleton.
- Add job-status checker skeleton.

### Phase 5: API deployment plan

- Containerize FastAPI.
- Load model artifacts from local path or S3.
- Deploy to ECS/Fargate.

---

## Portfolio Framing

This architecture demonstrates:

- Data lake design using raw/processed/curated zones.
- Batch ETL design using AWS Glue/PySpark.
- Orchestration using Step Functions or Airflow.
- MLOps lifecycle from ingestion to training to evaluation to serving.
- Model explainability through feature importance and evaluation reports.
- RAG-enabled documentation assistance.
- Agentic workflow routing through LangGraph.

The project should be positioned as an AI/MLOps research workbench, not merely a classifier.

Best framing:

```text
ColliderOpsAI is an AI/MLOps research workbench that supports both raw CERN/Open Data ingestion and curated particle-physics benchmark modeling. It demonstrates scientific data ingestion, ML feature validation, model comparison, evaluation reporting, RAG documentation, agentic routing, and an AWS-ready production architecture.
```